from __future__ import annotations

from typing import Any


def intro_content() -> dict[str, str]:
    return {
        "title": "DeepGaze IIE Visual Salience Explorer",
        "saliency_intro": (
            "Saliency is about where human vision is likely to land first in a scene. "
            "A saliency map turns an image into a spatial pattern of predicted attention."
        ),
        "deepgaze_intro": (
            "DeepGaze IIE is a modern probabilistic saliency model that predicts fixation "
            "likelihood over all pixels using deep image features plus a center-bias prior."
        ),
        "why_matters": (
            "For UI and product analysis, this helps teams inspect likely attention hotspots, "
            "compare layouts, and diagnose visual competition before running expensive user studies."
        ),
        "bridge_statement": (
            "Traditional saliency maps estimate what visually stands out. DeepGaze IIE extends "
            "this idea by using deep learning and human gaze data to predict where people are likely to look."
        ),
    }


def saliency_mapping_rows() -> list[dict[str, str]]:
    return [
        {
            "classic": "Image feature contrast (color, orientation, intensity)",
            "deepgaze": "CNN backbone feature representation",
            "meaning": "Both summarize visual structure; DeepGaze learns richer features from data.",
        },
        {
            "classic": "Saliency map",
            "deepgaze": "Fixation probability map",
            "meaning": "Both produce spatial attention estimates, but DeepGaze is probabilistic.",
        },
        {
            "classic": "Center prior / viewing bias",
            "deepgaze": "Center-bias input tensor",
            "meaning": "Both encode the tendency to look near image center.",
        },
        {
            "classic": "Handcrafted feature integration",
            "deepgaze": "Learned network inference",
            "meaning": "DeepGaze learns integration from gaze data instead of fixed rules.",
        },
        {
            "classic": "Attention heatmap visualization",
            "deepgaze": "Exponentiated log-density prediction",
            "meaning": "The heatmap is a view of a learned fixation distribution, not a certainty map.",
        },
    ]


def interpretation_points() -> list[dict[str, str]]:
    return [
        {
            "title": "What it suggests",
            "text": "Hot regions indicate higher predicted fixation probability under free-viewing conditions.",
        },
        {
            "title": "What it does not mean",
            "text": "Attention is not the same as importance, correctness, conversion, or task success.",
        },
        {
            "title": "Seeing is not understanding",
            "text": "A user may look at an element without comprehending it or acting on it.",
        },
        {
            "title": "Use as a diagnostic",
            "text": "Treat saliency as an intermediate signal to guide design iteration, then validate with user testing.",
        },
    ]


def probability_journey_explainer() -> dict[str, Any]:
    return {
        "title": "How The Image Becomes A Probabilistic Saliency Map",
        "subtitle": (
            "Paper-inspired explanation: saliency is treated as a spatial distribution over possible fixation "
            "locations, not a binary object mask."
        ),
        "stages": [
            {
                "name": "1. Visual Signal Encoding",
                "image": "/static/img/probability/01-encoding.svg",
                "from": "RGB image I(x, y)",
                "to": "Feature representation F(x, y)",
                "what_changes": "Raw pixels are converted into higher-level visual features.",
                "math_view": "I -> F via learned feature extractors",
                "why_it_matters": "This moves from appearance values to perception-relevant structure.",
            },
            {
                "name": "2. Add Viewing Prior (Center Bias)",
                "image": "/static/img/probability/02-center-prior.svg",
                "from": "Feature map F(x, y)",
                "to": "Feature map + prior C(x, y)",
                "what_changes": "A center-bias prior is introduced to model human viewing tendency.",
                "math_view": "Combine content term and prior term: score(x, y) <- f(F) + C",
                "why_it_matters": "Prediction uses both image content and known gaze behavior.",
            },
            {
                "name": "3. Produce Log-Density Scores",
                "image": "/static/img/probability/03-log-density.svg",
                "from": "Combined internal representation",
                "to": "Log-density map L(x, y)",
                "what_changes": "Model outputs log-likelihood-style scores per pixel.",
                "math_view": "L(x, y) ≈ log p(fixation at x, y | image)",
                "why_it_matters": "Log space is numerically stable and additive with priors.",
            },
            {
                "name": "4. Convert To Probability Space",
                "image": "/static/img/probability/04-exp.svg",
                "from": "Log-density L(x, y)",
                "to": "Positive map P_raw(x, y)",
                "what_changes": "Exponentiation turns log values into positive probability-like values.",
                "math_view": "P_raw(x, y) = exp(L(x, y))",
                "why_it_matters": "Now values are interpretable as relative fixation likelihoods.",
            },
            {
                "name": "5. Normalize Into A Distribution",
                "image": "/static/img/probability/05-normalize.svg",
                "from": "Positive map P_raw(x, y)",
                "to": "Distribution P(x, y)",
                "what_changes": "Map is normalized so total mass sums to 1.",
                "math_view": "P(x, y) = P_raw(x, y) / Σ_x,y P_raw(x, y)",
                "why_it_matters": "This makes the saliency map a true probability distribution.",
            },
            {
                "name": "6. Visualize As Heatmap",
                "image": "/static/img/probability/06-heatmap.svg",
                "from": "Probability distribution P(x, y)",
                "to": "Colorized saliency heatmap",
                "what_changes": "Distribution values are mapped to colors for human interpretation.",
                "math_view": "colormap(P)",
                "why_it_matters": "The heatmap is a visualization of probabilities, not certainty or importance.",
            },
        ],
        "footnote": (
            "Interpretation note: higher saliency means higher predicted fixation probability under the model's "
            "assumptions; it does not guarantee understanding, preference, or task success."
        ),
    }


def step_definitions() -> list[dict[str, Any]]:
    # Step IDs match the dynamic trace generated by the runtime pipeline.
    return [
        {
            "id": "original_image",
            "title": "1. Original image",
            "input_format": "Uploaded image file",
            "output_format": "RGB image array (H, W, 3)",
            "math_formula": "I in R^(H x W x 3)",
            "simple": "Load the image so the demo has something to analyze.",
            "technical": (
                "PIL opens the file and converts to RGB. Converting to NumPy yields an unsigned integer array "
                "with shape (H, W, 3)."
            ),
            "code": "raw_img = Image.open(image_path).convert('RGB')\nimage = np.array(raw_img)",
        },
        {
            "id": "image_array",
            "title": "2. Image converted into array",
            "input_format": "RGB image",
            "output_format": "NumPy array (H, W, 3)",
            "math_formula": "I[x, y, c] stores pixel intensities for c in {R, G, B}",
            "simple": "This turns pixels into numbers the model pipeline can manipulate.",
            "technical": "Each pixel stores three channels (R, G, B). Shape is height x width x channels.",
            "code": "image = np.array(raw_img)",
        },
        {
            "id": "centerbias_template",
            "title": "3. Center bias template",
            "input_format": "Template grid",
            "output_format": "2D center-bias map",
            "math_formula": "C0 in R^(H0 x W0), prior over fixation tendency",
            "simple": "This is a prior that humans often look near the center.",
            "technical": "A center-bias map encodes log-probability offsets independent of image content.",
            "code": "centerbias_template = load_centerbias_template()",
        },
        {
            "id": "centerbias_resized",
            "title": "4. Center bias resized to match image",
            "input_format": "Template map + image shape",
            "output_format": "Center-bias array (H, W)",
            "math_formula": "C = resize(C0) in R^(H x W)",
            "simple": "Resize the center-bias map so it aligns pixel-by-pixel with the image.",
            "technical": (
                "scipy.ndimage.zoom rescales the template with nearest-neighbor settings to image height and width."
            ),
            "code": (
                "centerbias = zoom(centerbias_template,\n"
                "    (image.shape[0] / centerbias_template.shape[0],\n"
                "     image.shape[1] / centerbias_template.shape[1]),\n"
                "    order=0, mode='nearest')"
            ),
        },
        {
            "id": "image_transposed",
            "title": "5. Image transformed into tensor-ready layout",
            "input_format": "NumPy image (H, W, 3)",
            "output_format": "Array (3, H, W)",
            "math_formula": "X = transpose(I, (2, 0, 1)) in R^(3 x H x W)",
            "simple": "Reorder channels so PyTorch sees channel-first data.",
            "technical": "transpose(2, 0, 1) changes from HWC to CHW, which most vision models expect.",
            "code": "chw = image.transpose(2, 0, 1)",
        },
        {
            "id": "image_tensor",
            "title": "6. Image transformed into tensor",
            "input_format": "Array (3, H, W)",
            "output_format": "Tensor (1, 3, H, W)",
            "math_formula": "Xb = batch(X) in R^(1 x 3 x H x W)",
            "simple": "Add batch dimension and convert to float tensor for model input.",
            "technical": "np.array([chw]) creates batch size 1. torch.tensor(...).float() produces float32 tensor.",
            "code": "image_tensor = torch.tensor(np.array([chw])).to(DEVICE).float()",
        },
        {
            "id": "centerbias_tensor",
            "title": "7. Center bias transformed into tensor",
            "input_format": "Center-bias array (H, W)",
            "output_format": "Tensor (1, H, W)",
            "math_formula": "Cb = batch(C) in R^(1 x H x W)",
            "simple": "Prepare the center-bias prior in tensor form for the model.",
            "technical": "Batch dimension is added so model receives one map per input image.",
            "code": "centerbias_tensor = torch.tensor([centerbias]).to(DEVICE).float()",
        },
        {
            "id": "model_inference",
            "title": "8. Model inference with DeepGaze IIE",
            "input_format": "Image tensor + center-bias tensor",
            "output_format": "Log-density tensor",
            "math_formula": "L = f_theta(Xb, Cb), where L in R^(H x W)",
            "simple": "Run the model to predict where people are likely to look.",
            "technical": "torch.no_grad() disables gradient tracking for faster, lower-memory inference.",
            "code": "with torch.no_grad():\n    log_density_prediction = model(image_tensor, centerbias_tensor)",
        },
        {
            "id": "log_density",
            "title": "9. Log-density prediction",
            "input_format": "Model output tensor",
            "output_format": "2D log-probability map",
            "math_formula": "L(x, y) ~= log p(fixation at (x, y) | I, C)",
            "simple": "The model outputs log values that represent attention likelihood before conversion.",
            "technical": "Output approximates log p(fixation at pixel | image, center bias).",
            "code": "log_density = log_density_prediction.detach().cpu().numpy()[0, 0]",
        },
        {
            "id": "probability_map",
            "title": "10. Exponentiation into probability map",
            "input_format": "Log-density map",
            "output_format": "Positive-valued probability-like map",
            "math_formula": "P_raw(x, y) = exp(L(x, y))",
            "simple": "Convert log values back into regular positive values for visualization.",
            "technical": "np.exp reverses log-space representation. Relative magnitudes correspond to fixation likelihood.",
            "code": "prediction = np.exp(log_density)",
        },
        {
            "id": "heatmap_interpretation",
            "title": "11. Final saliency heatmap interpretation",
            "input_format": "Probability map",
            "output_format": "Visual heatmap + interpretation",
            "math_formula": "P(x, y) = P_raw(x, y) / sum(P_raw), Heatmap = colormap(P)",
            "simple": "Colors show likely attention hotspots: warmer means more predicted gaze density.",
            "technical": "Heatmap is a visualization layer over a probabilistic distribution, not an explanation of intent.",
            "code": "plt.imshow(prediction, cmap='jet')",
        },
    ]
