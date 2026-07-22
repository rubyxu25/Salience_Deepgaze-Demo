# DeepGaze IIE Educational Web Demo

Interactive educational Flask app for visual salience and DeepGaze IIE, designed for both non-technical and technical audiences.

## What This Demo Includes

- Layered explanations (`Simple` / `Technical`) for each pipeline step
- Full step-by-step workflow trace:
  - Input image
  - preprocessing
  - center bias preparation
  - tensor conversion
  - model inference
  - log-density output
  - exponentiation to probability map
  - heatmap interpretation
- Shape journey tracker (e.g., `(H, W, 3)` -> `(1, 3, H, W)`)
- Classical saliency vs DeepGaze conceptual mapping table
- Upload + run interface with:
  - original image
  - predicted heatmap
  - overlay
  - download buttons
- Interpretation and limitation guidance
- Optional center-bias on/off toggle
- Overlay opacity slider
- Tooltip terms (`tensor`, `log-density`, `center bias`, `transpose`)

## Project Structure

- `app.py`: Flask server and API routes
- `deepgaze_demo/pipeline.py`: model execution + data trace generation
- `deepgaze_demo/content.py`: educational copy and step metadata
- `deepgaze_demo/utils.py`: helpers for encoding and validation
- `templates/index.html`: page structure
- `static/css/styles.css`: visual design
- `static/js/app.js`: interactive frontend behavior
- `DeepGazeDemo.py`: standalone reference script

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
python app.py
```

4. Open:

- `http://127.0.0.1:5000`

## Model Mode Note

By default, the app runs a fast educational salience approximation based on image contrast and center bias.

- This keeps uploads responsive for local demos and deployed web services.
- It produces salience-style heatmaps, but it is not the official DeepGaze IIE model prediction.
- To experiment with official DeepGaze IIE locally, install all model dependencies and run with `DEEPGAZE_USE_OFFICIAL=1`.

The simplified mode keeps the learning interface usable, but it is not a replacement for official DeepGaze model predictions.

## Running DeepGaze IIE Yourself

If you want to go beyond the web demo and run DeepGaze IIE in your own Python script, see:

- [`docs/deepgaze_iie_getting_started.md`](docs/deepgaze_iie_getting_started.md)

## Editing Content

Educational text is separated from inference logic:

- Edit `deepgaze_demo/content.py` for section text, step explanations, and mapping rows.
- Keep `deepgaze_demo/pipeline.py` focused on computation and shape tracing.
