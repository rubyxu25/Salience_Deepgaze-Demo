from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify, render_template, request
from PIL import Image
from werkzeug.utils import secure_filename

from deepgaze_demo.config import ALLOWED_EXTENSIONS, MAX_UPLOAD_MB, UPLOAD_DIR
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


@app.route("/")
def index():
    return render_template("index.html")


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


@app.post("/api/run")
def run_demo():
    file = request.files.get("image")
    if file is None or file.filename is None or file.filename.strip() == "":
        return jsonify({"error": "Please upload an image."}), 400

    if not allowed_file(file.filename, ALLOWED_EXTENSIONS):
        return jsonify({"error": f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"}), 400

    use_centerbias = request.form.get("use_centerbias", "true").lower() != "false"
    try:
        overlay_alpha = float(request.form.get("overlay_alpha", "0.45"))
    except ValueError:
        overlay_alpha = 0.45

    filename = secure_filename(file.filename)
    save_path = Path(UPLOAD_DIR) / filename
    file.save(save_path)

    try:
        pil_img = Image.open(save_path).convert("RGB")
    except Exception:
        return jsonify({"error": "Could not decode image."}), 400

    artifacts = runner.run(
        pil_image=pil_img,
        use_centerbias=use_centerbias,
        overlay_alpha=overlay_alpha,
    )

    return jsonify(
        {
            "model_mode": artifacts.model_mode,
            "centerbias_source": artifacts.centerbias_source,
            "warnings": artifacts.warnings,
            "results": {
                "original": artifacts.original_b64,
                "heatmap": artifacts.heatmap_b64,
                "overlay": artifacts.overlay_b64,
                "probability_min": artifacts.probability_min,
                "probability_max": artifacts.probability_max,
            },
            "trace": artifacts.trace,
            "shape_journey": artifacts.shape_journey,
            "caution": (
                "Heatmap colors indicate relative predicted visual attention, not certainty, not task success, "
                "and not semantic importance."
            ),
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
