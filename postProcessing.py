"""Utility helpers for post-processing alpha-matted images.

This module mirrors the refinements used inside the PyQt application and can
be used from the command line for scripted workflows (e.g. CI pipelines or
batch servers).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

RGBAColor = Tuple[int, int, int, int]


def smooth_alpha(image: Image.Image, radius: int = 3) -> Image.Image:
    """Feather an image's alpha channel to reduce jagged edges."""

    if radius <= 0:
        return image

    array = np.array(image.convert("RGBA"))
    alpha = array[..., 3]
    kernel = max(1, radius * 2 + 1)
    if kernel % 2 == 0:
        kernel += 1
    alpha = cv2.GaussianBlur(alpha, (kernel, kernel), 0)
    if radius >= 2:
        alpha = cv2.medianBlur(alpha, kernel)
    array[..., 3] = alpha
    return Image.fromarray(array, "RGBA")


def composite_background(
    foreground: Image.Image,
    mode: str = "transparent",
    *,
    color: RGBAColor = (255, 255, 255, 255),
    gradient_start: RGBAColor = (255, 255, 255, 255),
    gradient_end: RGBAColor = (0, 0, 0, 255),
    background_image: Optional[Path] = None,
) -> Image.Image:
    """Composite the `foreground` onto a background defined by ``mode``."""

    foreground = foreground.convert("RGBA")
    width, height = foreground.size

    if mode == "transparent":
        return foreground

    if mode == "solid":
        background = Image.new("RGBA", (width, height), color)
    elif mode == "gradient":
        background = _create_gradient((width, height), gradient_start, gradient_end)
    elif mode == "image" and background_image:
        bg = Image.open(background_image).convert("RGBA")
        background = bg.resize((width, height), Image.LANCZOS)
    else:
        background = Image.new("RGBA", (width, height), (255, 255, 255, 255))

    return Image.alpha_composite(background, foreground)


def _create_gradient(size: Tuple[int, int], start: RGBAColor, end: RGBAColor) -> Image.Image:
    width, height = size
    gradient = np.linspace(0.0, 1.0, width, dtype=np.float32)
    array = np.zeros((height, width, 4), dtype=np.float32)
    for i in range(4):
        array[..., i] = start[i] + (end[i] - start[i]) * gradient
    array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array, "RGBA")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refine alpha mattes for cleaner edges.")
    parser.add_argument("input", type=Path, help="Path to an RGBA image with alpha.")
    parser.add_argument("output", type=Path, help="Destination for the refined image.")
    parser.add_argument("--radius", type=int, default=3, help="Feather radius in pixels (default: 3).")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    image = Image.open(args.input).convert("RGBA")
    refined = smooth_alpha(image, args.radius)
    refined.save(args.output)


if __name__ == "__main__":
    main()
