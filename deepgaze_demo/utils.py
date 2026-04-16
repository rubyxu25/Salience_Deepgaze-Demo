from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image


def allowed_file(filename: str, allowed_extensions: set[str]) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def image_to_base64_png(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def array_to_base64_png(arr: np.ndarray) -> str:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = normalize_to_uint8(arr)
        img = Image.fromarray(arr, mode="L").convert("RGB")
    else:
        img = Image.fromarray(arr.astype(np.uint8), mode="RGB")
    return image_to_base64_png(img)


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    finite_mask = np.isfinite(arr)
    if not finite_mask.any():
        return np.zeros(arr.shape, dtype=np.uint8)

    finite_vals = arr[finite_mask]
    min_v = float(finite_vals.min())
    max_v = float(finite_vals.max())

    # Keep rendering robust if upstream tensors contain NaN/Inf.
    arr = np.nan_to_num(arr, nan=min_v, posinf=max_v, neginf=min_v)
    if max_v - min_v < 1e-8:
        return np.zeros(arr.shape, dtype=np.uint8)
    return ((arr - min_v) / (max_v - min_v) * 255.0).clip(0, 255).astype(np.uint8)


def format_shape(data: np.ndarray | tuple[int, ...]) -> str:
    if isinstance(data, tuple):
        shape = data
    else:
        shape = tuple(data.shape)
    return str(shape)
