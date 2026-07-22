# Getting Started With DeepGaze IIE In Your Own Python Script

This guide is for people who used the web demo and now want to run the **official DeepGaze IIE model** directly in Python.

Important distinction:

- The web demo is now a **fast interactive teaching demo**. By default, it uses a lightweight salience approximation so users can upload images and get heatmaps quickly.
- The demo still teaches the important mechanism: image preprocessing, center bias, tensor shapes, log-density style output, heatmap visualization, and interpretation limits.
- The demo is **not the full official DeepGaze IIE model by default**.
- If you need official DeepGaze IIE predictions, use this guide to install the model and run it locally.

DeepGaze IIE predicts a **probability distribution of human fixations** over an image. In simpler words, it estimates which image regions people are more likely to look at under free-viewing conditions.

It does not detect objects, diagnose images, or prove that a region is semantically important. It predicts visual attention.

## How The Web Demo Relates To Official DeepGaze IIE

The web demo and official DeepGaze IIE share the same conceptual workflow:

```text
Image
-> RGB array
-> center bias
-> model-ready tensors
-> salience / fixation-density map
-> heatmap
-> overlay
```

The difference is the prediction step:

```text
Web demo default:
Fast salience approximation
Uses contrast, color distinctiveness, saturation, smoothing, and center bias.

Official DeepGaze IIE:
Deep neural model trained on human fixation data
Uses multiple pretrained visual backbones and learned salience layers.
```

So the demo is useful for learning how salience maps are produced and interpreted. But for research-grade DeepGaze IIE output, run the official model locally.

## What You Need

### Hardware

- **CPU only:** Works, and is the easiest way to start. It can be slow because DeepGaze IIE uses several deep neural network backbones.
- **NVIDIA GPU:** Recommended for faster experiments. You need a CUDA-compatible PyTorch installation.
- **Apple Silicon Mac:** CPU is usually the safest beginner path. Some PyTorch operations may work with MPS, but official DeepGaze dependencies are easier to debug on CPU or CUDA.
- **Memory:** Use images with a longest side around `1024 px` or smaller. Very large images can be slow or memory-heavy.

### Software

- Python `3.10` or `3.11` recommended
- `git`
- A Python virtual environment
- Internet access for the first run, because pretrained model weights are downloaded automatically

The demo is deployed with Python `3.11.9`, but local Python `3.9+` may also work if the dependencies install successfully.

## Install Dependencies

From this project folder:

```bash
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

The important packages are:

```text
torch
torchvision
numpy
Pillow
scipy
matplotlib
deepgaze_pytorch
```

In this project, `deepgaze_pytorch` is installed from GitHub through `requirements.txt`.

## First-Time Weight Downloads

The official model call:

```python
deepgaze_pytorch.DeepGazeIIE(pretrained=True)
```

downloads several pretrained weights the first time it runs. These may include backbone weights such as ResNet/ShapeNet, EfficientNet, DenseNet, ResNeXt, plus the final DeepGaze IIE weights.

On most machines, PyTorch stores them here:

```text
~/.cache/torch/hub/checkpoints/
```

The first run can be slow. If a download is interrupted, PyTorch may leave a partial or corrupted file in that folder.

## The Core Official DeepGaze IIE Workflow

The official model workflow is:

1. Open an image with Pillow.
2. Convert it to an RGB NumPy array with shape `(H, W, 3)`.
3. Resize or generate a center-bias map with shape `(H, W)`.
4. Transpose the image from `(H, W, 3)` to `(3, H, W)`.
5. Add a batch dimension to get `(1, 3, H, W)`.
6. Convert image and center bias to PyTorch tensors.
7. Run `deepgaze_pytorch.DeepGazeIIE(pretrained=True)`.
8. Convert the model output from log-density to probability-like values with `np.exp`.
9. Visualize the result as a heatmap or overlay.

The model output is a **log-density map**, not a normal RGB image.

## Minimal Official DeepGaze IIE Script

Create a file named `run_deepgaze_iie.py` in the project root:

```python
from pathlib import Path

import deepgaze_pytorch
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import zoom
from scipy.special import logsumexp


IMAGE_PATH = Path("test.png")
CENTERBIAS_PATH = Path("centerbias_mit1003.npy")
OUTPUT_PATH = Path("deepgaze_iie_heatmap.png")
MAX_SIDE = 1024


def resize_for_inference(image, max_side=MAX_SIDE):
    width, height = image.size
    longest = max(width, height)
    if longest <= max_side:
        return image

    scale = max_side / float(longest)
    new_size = (round(width * scale), round(height * scale))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def make_default_centerbias(shape):
    h, w = shape
    ys = np.linspace(-1.0, 1.0, h, dtype=np.float32)
    xs = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    sigma = 0.55
    gaussian = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    gaussian = gaussian / np.maximum(gaussian.sum(), 1e-8)
    return np.log(np.maximum(gaussian, 1e-12)).astype(np.float32)


def load_centerbias(shape):
    h, w = shape
    if CENTERBIAS_PATH.exists():
        template = np.load(CENTERBIAS_PATH).astype(np.float32)
    else:
        template = make_default_centerbias((1024, 1024))

    centerbias = zoom(
        template,
        (h / template.shape[0], w / template.shape[1]),
        order=0,
        mode="nearest",
    ).astype(np.float32)

    centerbias = centerbias - logsumexp(centerbias)
    return centerbias


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    pil_image = Image.open(IMAGE_PATH).convert("RGB")
    pil_image = resize_for_inference(pil_image)
    image = np.array(pil_image)
    h, w = image.shape[:2]

    centerbias = load_centerbias((h, w))

    image_chw = image.transpose(2, 0, 1)
    image_tensor = torch.from_numpy(np.array([image_chw])).float().to(device)
    centerbias_tensor = torch.from_numpy(np.array([centerbias])).float().to(device)

    model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).to(device)
    model.eval()

    with torch.no_grad():
        log_density_prediction = model(image_tensor, centerbias_tensor)

    log_density = log_density_prediction.detach().cpu().numpy()[0, 0]
    prediction = np.exp(log_density)

    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.imshow(prediction, cmap="jet", alpha=0.45)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=160, bbox_inches="tight", pad_inches=0)
    print(f"Saved heatmap overlay to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
```

Run it:

```bash
source .venv/bin/activate
python run_deepgaze_iie.py
```

The first run may take a while because the model downloads pretrained weights.

## Understanding The Shapes

If your input image is `H` pixels tall and `W` pixels wide:

```text
Pillow image              image file
NumPy RGB image           (H, W, 3)
Transposed image          (3, H, W)
Batched image tensor      (1, 3, H, W)
Center-bias tensor        (1, H, W)
Model output              (1, 1, H, W)
Final log-density map     (H, W)
Exponentiated prediction  (H, W)
```

DeepGaze IIE expects the image tensor in **channels-first** format: `(batch, channels, height, width)`.

## What Is Center Bias?

Human viewers often look near the center of an image, especially in free-viewing datasets. DeepGaze IIE uses a center-bias map as an explicit input.

In this repo, the code looks for:

```text
centerbias_mit1003.npy
```

If that file is available, the center-bias template is resized to match the input image. If not, you can generate a simple Gaussian center bias for learning.

To test the model without center bias, you can replace it with zeros:

```python
centerbias = np.zeros((h, w), dtype=np.float32)
```

That is useful for learning, but the official model is normally used with a proper center-bias map.

## CPU vs GPU

For CPU:

```python
device = torch.device("cpu")
```

For NVIDIA GPU:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

If you use CUDA, make sure your PyTorch installation matches your CUDA version. The regular `pip install torch torchvision` command may install CPU-only PyTorch depending on your machine and platform.

For beginners, start with CPU first. Once the script works, move to GPU if inference is too slow.

## Using The Web Demo As A Learning Aid

Even though the web demo does not run the full official model by default, its breakdown remains useful for learning the mechanism behind salience outputs:

- **Upload and preprocessing:** how an image becomes an RGB array.
- **Center bias:** why visual attention often has a spatial prior.
- **Tensor preparation:** why model inputs use `(1, 3, H, W)` and `(1, H, W)`.
- **Log-density idea:** why salience models often work with probability-like spatial maps.
- **Heatmap interpretation:** why warm colors mean higher predicted/estimated attention, not certainty or semantic importance.

The demo is best understood as an interactive explanation of the pipeline. This guide is for running the official model yourself.

## Common Problems

### The first official model run is slow

This is expected. DeepGaze IIE downloads multiple pretrained weights and builds several feature backbones.

### `unexpected EOF` or `file might be corrupted`

This usually means a model weight download was interrupted. Remove or rename the corrupted file in:

```text
~/.cache/torch/hub/checkpoints/
```

Then rerun the script so PyTorch can download it again.

For example:

```bash
ls -lh ~/.cache/torch/hub/checkpoints
```

If a specific `.pth` or `.pth.tar` file is named in the error, rename it:

```bash
mv ~/.cache/torch/hub/checkpoints/BAD_FILE.pth ~/.cache/torch/hub/checkpoints/BAD_FILE_corrupted_backup.pth
```

Then rerun:

```bash
python run_deepgaze_iie.py
```

### The image is too large

Resize before inference. This guide uses:

```python
MAX_SIDE = 1024
```

This keeps memory use reasonable and makes CPU runs more practical.

### The heatmap looks different from object detection

That is normal. DeepGaze IIE predicts likely fixation density, not object categories or bounding boxes. A bright region means the model predicts higher visual attention there.

## Recommended Way To Explain This Project

Use this wording if you present the demo:

```text
The web demo illustrates the DeepGaze IIE salience workflow with a fast salience approximation. It is designed for interactive learning and quick image uploads. Users who need official DeepGaze IIE predictions can follow the attached guide to install the model, download pretrained weights, and run the full model locally.
```

