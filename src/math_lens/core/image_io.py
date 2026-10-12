"""Small, shared helpers for loading and saving RGB images."""

from io import BytesIO
from pathlib import Path
from typing import TypeAlias

import numpy as np
from PIL import Image

PathLike: TypeAlias = str | Path


def load_rgb_image(path: PathLike, max_size: int | None = None) -> np.ndarray:
    """Load an image as an ``(height, width, 3)`` uint8 RGB array.

    If ``max_size`` is supplied, images larger than that value on either axis
    are reduced while preserving their aspect ratio. Smaller images are not
    enlarged.
    """

    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image does not exist: {image_path}")
    if max_size is not None and max_size <= 0:
        raise ValueError("max_size must be greater than zero")

    with Image.open(image_path) as source:
        return _as_rgb_array(source, max_size)


def load_rgb_bytes(data: bytes, max_size: int | None = None) -> np.ndarray:
    """Load uploaded image bytes using the same RGB conversion and resize rules."""

    if not data:
        raise ValueError("image data is empty")
    if max_size is not None and max_size <= 0:
        raise ValueError("max_size must be greater than zero")
    with Image.open(BytesIO(data)) as source:
        return _as_rgb_array(source, max_size)


def _as_rgb_array(source: Image.Image, max_size: int | None) -> np.ndarray:
    image = source.convert("RGB")
    if max_size is not None and max(image.size) > max_size:
        # Pillow computes the aspect-preserving dimensions for this box.
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return np.array(image, dtype=np.uint8, copy=True)


def save_rgb_image(array: np.ndarray, path: PathLike) -> Path:
    """Save an RGB uint8 array, creating its parent directory if necessary."""

    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("RGB image array must have shape (height, width, 3)")
    if array.dtype != np.uint8:
        raise ValueError("RGB image array must have dtype uint8")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGB").save(output_path)
    return output_path
