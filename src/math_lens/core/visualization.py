"""Small labeled image panels shared by saved analysis workflows."""

from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def make_image_panel(
    images: Sequence[np.ndarray], labels: Sequence[str], caption: str = ""
) -> np.ndarray:
    """Place equal-size RGB images in a row with readable labels and caption."""

    if not images or len(images) != len(labels):
        raise ValueError("one label is required per image")
    height, width, channels = images[0].shape
    if channels != 3 or any(
        image.shape != (height, width, 3) or image.dtype != np.uint8
        for image in images
    ):
        raise ValueError("panel images must share one uint8 RGB shape")
    font_size = max(12, min(28, width // 28))
    font = ImageFont.load_default(size=font_size)
    header = font_size + 20
    footer = font_size + 20 if caption else 0
    gap = 12
    canvas = Image.new(
        "RGB", (width * len(images) + gap * (len(images) - 1), height + header + footer),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for index, (image, label) in enumerate(zip(images, labels)):
        left = index * (width + gap)
        canvas.paste(Image.fromarray(image), (left, header))
        draw.text((left + width / 2, header / 2), label, font=font, fill="#222222", anchor="mm")
    if caption:
        draw.text((12, height + header + footer / 2), caption, font=font, fill="#333333", anchor="lm")
    return np.asarray(canvas, dtype=np.uint8).copy()
