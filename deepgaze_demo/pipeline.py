from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.ndimage import zoom
from scipy.special import logsumexp

from .config import DEFAULT_CENTERBIAS_SHAPE
from .content import step_definitions
from .utils import array_to_base64_png, format_shape, image_to_base64_png, normalize_to_uint8

try:
    import torch
except Exception:  # pragma: no cover - optional runtime dependency
    torch = None  # type: ignore[assignment]


@dataclass
class InferenceArtifacts:
    trace: list[dict[str, Any]]
    shape_journey: list[dict[str, str]]
    original_b64: str
    heatmap_b64: str
    overlay_b64: str
    probability_min: float
    probability_max: float
    model_mode: str
    centerbias_source: str
    warnings: list[str]


class DeepGazeRunner:
    """Runs DeepGaze IIE if available, otherwise falls back to a deterministic heuristic map."""

    def __init__(self, device: str = "cpu") -> None:
        self.device = torch.device(device) if torch is not None else device
        self._model = None
        self._mode = "heuristic-fallback"
        self._model_load_error: str | None = None

    @property
    def mode(self) -> str:
        return self._mode

    def _load_model(self) -> None:
        if self._model is not None:
            return
        self._model_load_error = None
        if torch is None:
            self._mode = "heuristic-fallback"
            self._model_load_error = "PyTorch is not available in this environment."
            return
        # Lazy import keeps the app runnable if deepgaze is not installed.
        try:
            import deepgaze_pytorch  # type: ignore

            if hasattr(deepgaze_pytorch, "DeepGazeIIE"):
                self._model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).to(self.device)
                self._model.eval()
                self._mode = "deepgaze_iie"
            else:
                self._mode = "heuristic-fallback"
                self._model_load_error = "deepgaze_pytorch is installed but DeepGazeIIE is missing."
        except Exception as exc:
            self._mode = "heuristic-fallback"
            self._model_load_error = f"Failed to load DeepGaze IIE weights: {exc}"

    def _default_centerbias_template(self, shape: tuple[int, int]) -> np.ndarray:
        h, w = shape
        ys = np.linspace(-1.0, 1.0, h, dtype=np.float32)
        xs = np.linspace(-1.0, 1.0, w, dtype=np.float32)
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
        sigma = 0.55
        gaussian = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        gaussian /= np.maximum(gaussian.sum(), 1e-8)
        return np.log(np.maximum(gaussian, 1e-12)).astype(np.float32)

    def _load_centerbias_template(self) -> tuple[np.ndarray, str]:
        candidates = [
            Path("centerbias_mit1003.npy"),
            Path("DeepGaze/centerbias_mit1003.npy"),
            Path("deepgaze/centerbias_mit1003.npy"),
        ]
        for path in candidates:
            if path.exists():
                try:
                    arr = np.load(path).astype(np.float32)
                    if arr.ndim == 2:
                        return arr, f"file:{path}"
                except Exception:
                    continue
        return self._default_centerbias_template(DEFAULT_CENTERBIAS_SHAPE), "generated-default"

    def _heuristic_log_density(self, image: np.ndarray, centerbias: np.ndarray) -> np.ndarray:
        # Edge-aware luminance contrast plus center bias to keep the educational flow functional.
        img = image.astype(np.float32) / 255.0
        lum = 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]
        gx = np.zeros_like(lum)
        gy = np.zeros_like(lum)
        gx[:, 1:-1] = lum[:, 2:] - lum[:, :-2]
        gy[1:-1, :] = lum[2:, :] - lum[:-2, :]
        contrast = np.sqrt(gx * gx + gy * gy)

        contrast = contrast + 1e-8
        contrast = contrast / contrast.sum()
        log_contrast = np.log(np.maximum(contrast, 1e-12))

        combined = 0.70 * log_contrast + 0.30 * centerbias
        combined -= logsumexp(combined)
        return combined.astype(np.float32)

    def run(self, pil_image: Image.Image, use_centerbias: bool = True, overlay_alpha: float = 0.45) -> InferenceArtifacts:
        self._load_model()
        warnings: list[str] = []
        if self._mode != "deepgaze_iie":
            warnings.append(
                "DeepGaze IIE weights are not loaded. Running educational fallback instead of official model inference."
            )
            if self._model_load_error:
                warnings.append(f"Model load detail: {self._model_load_error}")

        image = np.array(pil_image.convert("RGB"))
        h, w = image.shape[0], image.shape[1]

        centerbias_template, centerbias_source = self._load_centerbias_template()
        centerbias = zoom(
            centerbias_template,
            (h / centerbias_template.shape[0], w / centerbias_template.shape[1]),
            order=0,
            mode="nearest",
        ).astype(np.float32)
        if not use_centerbias:
            centerbias = np.zeros_like(centerbias)

        chw = image.transpose(2, 0, 1)
        img_input = np.array([chw])
        if torch is not None:
            image_tensor = torch.tensor(img_input).to(self.device).float()
            centerbias_tensor = torch.tensor([centerbias]).to(self.device).float()
            image_tensor_shape = tuple(image_tensor.shape)
            centerbias_tensor_shape = tuple(centerbias_tensor.shape)
        else:
            image_tensor = img_input.astype(np.float32)
            centerbias_tensor = np.array([centerbias], dtype=np.float32)
            image_tensor_shape = tuple(image_tensor.shape)
            centerbias_tensor_shape = tuple(centerbias_tensor.shape)

        if self._mode == "deepgaze_iie" and self._model is not None:
            with torch.no_grad():
                log_density_prediction = self._model(image_tensor, centerbias_tensor)
            log_density = log_density_prediction.detach().cpu().numpy()[0, 0]
        else:
            log_density = self._heuristic_log_density(image, centerbias)

        prediction = np.exp(log_density)

        heat_rgb = self._to_heatmap_rgb(prediction)
        overlay_rgb = self._overlay(image, heat_rgb, overlay_alpha)

        trace = self._build_trace(
            image=image,
            centerbias_template=centerbias_template,
            centerbias=centerbias,
            chw=chw,
            image_tensor=image_tensor,
            centerbias_tensor=centerbias_tensor,
            log_density=log_density,
            prediction=prediction,
        )

        shape_journey = [
            {"label": "Original image", "shape": "file"},
            {"label": "NumPy image array", "shape": format_shape(image)},
            {"label": "Transposed array", "shape": format_shape(chw)},
            {"label": "Batched image tensor", "shape": format_shape(image_tensor_shape)},
            {"label": "Center-bias map", "shape": format_shape(centerbias)},
            {"label": "Batched center-bias tensor", "shape": format_shape(centerbias_tensor_shape)},
            {"label": "Log-density output", "shape": format_shape(log_density)},
            {"label": "Exponentiated prediction", "shape": format_shape(prediction)},
        ]

        return InferenceArtifacts(
            trace=trace,
            shape_journey=shape_journey,
            original_b64=image_to_base64_png(Image.fromarray(image)),
            heatmap_b64=array_to_base64_png(heat_rgb),
            overlay_b64=array_to_base64_png(overlay_rgb),
            probability_min=float(prediction.min()),
            probability_max=float(prediction.max()),
            model_mode=self._mode,
            centerbias_source=centerbias_source,
            warnings=warnings,
        )

    def _to_heatmap_rgb(self, prediction: np.ndarray) -> np.ndarray:
        from matplotlib import cm

        pred = np.asarray(prediction, dtype=np.float32)
        finite_mask = np.isfinite(pred)
        if not finite_mask.any():
            return np.zeros((*pred.shape, 3), dtype=np.uint8)

        finite_vals = pred[finite_mask]
        low = float(np.percentile(finite_vals, 1.0))
        high = float(np.percentile(finite_vals, 99.0))
        if high - low < 1e-12:
            low = float(finite_vals.min())
            high = float(finite_vals.max())

        pred = np.nan_to_num(pred, nan=low, posinf=high, neginf=low)
        if high - low < 1e-12:
            norm = np.zeros_like(pred, dtype=np.float32)
        else:
            norm = np.clip((pred - low) / (high - low), 0.0, 1.0)

        rgba = cm.get_cmap("jet")(norm)
        rgb = (rgba[:, :, :3] * 255.0).astype(np.uint8)
        return rgb

    def _overlay(self, image_rgb: np.ndarray, heat_rgb: np.ndarray, alpha: float) -> np.ndarray:
        alpha = float(np.clip(alpha, 0.0, 1.0))
        base = image_rgb.astype(np.float32)
        heat = heat_rgb.astype(np.float32)
        out = ((1.0 - alpha) * base + alpha * heat).clip(0, 255).astype(np.uint8)
        return out

    def _build_trace(
        self,
        *,
        image: np.ndarray,
        centerbias_template: np.ndarray,
        centerbias: np.ndarray,
        chw: np.ndarray,
        image_tensor: Any,
        centerbias_tensor: Any,
        log_density: np.ndarray,
        prediction: np.ndarray,
    ) -> list[dict[str, Any]]:
        step_map = {s["id"]: dict(s) for s in step_definitions()}

        previews = {
            "original_image": image,
            "image_array": image,
            "centerbias_template": normalize_to_uint8(centerbias_template),
            "centerbias_resized": normalize_to_uint8(centerbias),
            "image_transposed": normalize_to_uint8(chw[0]),
            "image_tensor": normalize_to_uint8(chw[1]),
            "centerbias_tensor": normalize_to_uint8(centerbias),
            "model_inference": normalize_to_uint8(log_density),
            "log_density": normalize_to_uint8(log_density),
            "probability_map": normalize_to_uint8(prediction),
            "heatmap_interpretation": self._to_heatmap_rgb(prediction),
        }

        io_details = {
            "original_image": ("file", format_shape(image)),
            "image_array": (format_shape(image), format_shape(image)),
            "centerbias_template": ("template", format_shape(centerbias_template)),
            "centerbias_resized": (format_shape(centerbias_template), format_shape(centerbias)),
            "image_transposed": (format_shape(image), format_shape(chw)),
            "image_tensor": (format_shape(chw), format_shape(tuple(image_tensor.shape))),
            "centerbias_tensor": (format_shape(centerbias), format_shape(tuple(centerbias_tensor.shape))),
            "model_inference": (
                f"image {tuple(image_tensor.shape)} + centerbias {tuple(centerbias_tensor.shape)}",
                "(1, 1, H, W)",
            ),
            "log_density": ("(1, 1, H, W)", format_shape(log_density)),
            "probability_map": (format_shape(log_density), format_shape(prediction)),
            "heatmap_interpretation": (format_shape(prediction), "RGB heatmap (H, W, 3)"),
        }

        trace: list[dict[str, Any]] = []
        for step in step_definitions():
            sid = step["id"]
            merged = step_map[sid]
            merged["preview_b64"] = array_to_base64_png(previews[sid])
            merged["runtime_input_shape"], merged["runtime_output_shape"] = io_details[sid]
            trace.append(merged)
        return trace
