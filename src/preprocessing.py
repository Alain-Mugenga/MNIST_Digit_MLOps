"""Image input: grayscale 8x8 pixels, intensity 0..16, flattened to 64 features."""
import io
import numpy as np
from PIL import Image, UnidentifiedImageError
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def dataset():
    digits = load_digits()
    images = digits.images.astype(np.float32)  # Real 8x8 images, not CSV/tabular data
    labels = digits.target.astype(int)
    return train_test_split(images, labels, test_size=0.2, random_state=42, stratify=labels)


def features(images):
    a = np.asarray(images, dtype=np.float32)
    return a.reshape(-1, 64) / 16.0


def decode_image(raw):
    if not raw or len(raw) > 5_000_000:
        raise ValueError("An image must be provided (maximum size: 5 MB).")
    try:
        with Image.open(io.BytesIO(raw)) as im:
            im.verify()
        with Image.open(io.BytesIO(raw)) as im:
            if im.width > 4000 or im.height > 4000:
                raise ValueError("Image dimensions are too large.")
            im = im.convert('L').resize((8, 8), Image.Resampling.LANCZOS)
            arr = np.asarray(im, dtype=np.float32)
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Upload a valid PNG or JPEG image.") from exc
    # Training images: black background (0) and bright foreground (16).
    # Samples exported by this project use that convention.
    return arr / 255.0 * 16.0


def explain_features(images):
    a = np.asarray(images, dtype=float)
    density = (a > 1).sum(axis=(1, 2))
    brightness = a.mean(axis=(1, 2))
    weights = a.sum(axis=(1, 2))
    xx = np.arange(8)[None, None, :]
    x_center = (a * xx).sum(axis=(1, 2)) / np.maximum(weights, 1e-8)
    return density, brightness, x_center
