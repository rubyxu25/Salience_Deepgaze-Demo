from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file, send_from_directory
from PIL import Image
from werkzeug.exceptions import HTTPException
from werkzeug.utils import secure_filename

from deepgaze_demo.config import ALLOWED_EXTENSIONS, MAX_INFERENCE_SIDE, MAX_UPLOAD_MB, UPLOAD_DIR
from deepgaze_demo.content import (
    intro_content,
    interpretation_points,
    probability_journey_explainer,
    saliency_mapping_rows,
    step_definitions,
)
from deepgaze_demo.pipeline import DeepGazeRunner
from deepgaze_demo.utils import allowed_file, ensure_dir

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
ensure_dir(UPLOAD_DIR)

runner = DeepGazeRunner(device="cpu")
SAMPLE_DIR = Path("sample_images")
SAMPLE_LABELS = {
    "Stop_Signs_-_geograph.org.uk_-_857110.jpg": "Street Sign Scene",
    "broken-bone.jpg": "Medical X-Ray",
    "poster.png": "Food Poster",
}


@app.errorhandler(Exception)
def handle_unexpected_error(exc: Exception):
    if isinstance(exc, HTTPException):
        return exc
    app.logger.exception("Unhandled server error")
    return jsonify({"error": "Server error while generating saliency result.", "detail": str(exc)}), 500


def _resize_for_inference(image: Image.Image, max_side: int) -> tuple[Image.Image, tuple[int, int] | None]:
    width, height = image.size
    longest = max(width, height)
    if longest <= max_side:
        return image, None

    scale = max_side / float(longest)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    resized = image.resize(new_size, Image.Resampling.LANCZOS)
    return resized, (width, height)


def _run_image_response(
    pil_img: Image.Image,
    *,
    use_centerbias: bool,
    overlay_alpha: float,
    source_label: str,
) -> dict:
    pil_img, original_size = _resize_for_inference(pil_img, MAX_INFERENCE_SIDE)

    artifacts = runner.run(
        pil_image=pil_img,
        use_centerbias=use_centerbias,
        overlay_alpha=overlay_alpha,
    )
    if original_size is not None and artifacts.model_mode not in {"fast-salience"}:
        artifacts.warnings.append(
            (
                "Input image was resized before inference to reduce memory usage: "
                f"{original_size[0]}x{original_size[1]} -> {pil_img.size[0]}x{pil_img.size[1]}."
            )
        )

    return {
        "model_mode": artifacts.model_mode,
        "centerbias_source": artifacts.centerbias_source,
        "use_centerbias": use_centerbias,
        "source_label": source_label,
        "warnings": artifacts.warnings,
        "results": {
            "original": artifacts.original_b64,
            "heatmap": artifacts.heatmap_b64,
            "overlay": artifacts.overlay_b64,
            "probability_min": artifacts.probability_min,
            "probability_max": artifacts.probability_max,
            "image_width": artifacts.image_width,
            "image_height": artifacts.image_height,
            "heatmap_stats": artifacts.heatmap_stats,
        },
        "trace": artifacts.trace,
        "shape_journey": artifacts.shape_journey,
        "caution": (
            "Heatmap colors indicate relative predicted visual attention, not certainty, not task success, "
            "and not semantic importance."
        ),
    }


def _runtime_options() -> tuple[bool, float]:
    use_centerbias = request.form.get("use_centerbias", "true").lower() != "false"
    try:
        overlay_alpha = float(request.form.get("overlay_alpha", "0.45"))
    except ValueError:
        overlay_alpha = 0.45
    return use_centerbias, overlay_alpha


@app.route("/")
def index():
    return render_template("index.html")


@app.get("/sample-images/<path:filename>")
def sample_image(filename: str):
    safe_name = secure_filename(filename)
    if safe_name not in SAMPLE_LABELS:
        return jsonify({"error": "Unknown sample image."}), 404
    return send_from_directory(SAMPLE_DIR, safe_name)


@app.get("/getting-started")
def getting_started():
    return send_file(
        Path("docs") / "deepgaze_iie_getting_started.md",
        mimetype="text/markdown; charset=utf-8",
        download_name="deepgaze_iie_getting_started.md",
    )


@app.get("/api/content")
def content():
    return jsonify(
        {
            "intro": intro_content(),
            "mapping_rows": saliency_mapping_rows(),
            "probability_journey": probability_journey_explainer(),
            "steps": step_definitions(),
            "interpretation": interpretation_points(),
        }
    )


@app.get("/api/samples")
def samples():
    items = []
    for filename, label in SAMPLE_LABELS.items():
        path = SAMPLE_DIR / filename
        if not path.exists():
            continue
        try:
            with Image.open(path) as img:
                width, height = img.size
        except Exception:
            continue
        items.append(
            {
                "id": filename,
                "label": label,
                "filename": filename,
                "width": width,
                "height": height,
                "url": f"/sample-images/{filename}",
            }
        )
    return jsonify({"samples": items})


@app.post("/api/run")
def run_demo():
    file = request.files.get("image")
    if file is None or file.filename is None or file.filename.strip() == "":
        return jsonify({"error": "Please upload an image."}), 400

    if not allowed_file(file.filename, ALLOWED_EXTENSIONS):
        return jsonify({"error": f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"}), 400

    use_centerbias, overlay_alpha = _runtime_options()

    filename = secure_filename(file.filename)
    save_path = Path(UPLOAD_DIR) / filename
    file.save(save_path)

    try:
        pil_img = Image.open(save_path).convert("RGB")
    except Exception:
        return jsonify({"error": "Could not decode image."}), 400
    return jsonify(_run_image_response(pil_img, use_centerbias=use_centerbias, overlay_alpha=overlay_alpha, source_label=filename))


@app.post("/api/run-sample")
def run_sample():
    sample_id = request.form.get("sample_id", "")
    safe_name = secure_filename(sample_id)
    if safe_name not in SAMPLE_LABELS:
        return jsonify({"error": "Unknown sample image."}), 404

    path = SAMPLE_DIR / safe_name
    if not path.exists():
        return jsonify({"error": "Sample image file is missing."}), 404

    use_centerbias, overlay_alpha = _runtime_options()
    try:
        pil_img = Image.open(path).convert("RGB")
    except Exception:
        return jsonify({"error": "Could not decode sample image."}), 400

    return jsonify(
        _run_image_response(
            pil_img,
            use_centerbias=use_centerbias,
            overlay_alpha=overlay_alpha,
            source_label=SAMPLE_LABELS[safe_name],
        )
    )


if __name__ == "__main__":
    app.run(debug=True)
