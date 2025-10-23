"""Image transformation utilities used by the background remover."""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageEnhance

from .state import ImageState

RGBAColor = Tuple[int, int, int, int]


def apply_hue_shift(image: Image.Image, hue_shift: int) -> Image.Image:
    """Apply a hue rotation measured in degrees."""

    if hue_shift == 0:
        return image

    array = np.array(image.convert("RGBA"))
    rgb = array[..., :3]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue = hsv[..., 0].astype(np.int16)
    shift = int(round(hue_shift / 2))
    hsv[..., 0] = np.mod(hue + shift, 180).astype(np.uint8)
    rgb_shifted = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    array[..., :3] = rgb_shifted
    return Image.fromarray(array, "RGBA")


def apply_enhancements(image: Image.Image, state: ImageState) -> Image.Image:
    """Adjust contrast, brightness, shadows, and highlights."""

    contrast = max(0.1, state.contrast)
    brightness = max(0.1, state.brightness)
    shadows = max(0.1, state.blacks)
    highlights = max(0.1, state.whites)

    image = ImageEnhance.Contrast(image).enhance(contrast)
    image = ImageEnhance.Brightness(image).enhance(brightness)

    array = np.array(image.convert("RGBA"), dtype=np.float32)
    rgb = array[..., :3] / 255.0

    if not np.isclose(shadows, 1.0):
        gamma = 1.0 / shadows
        rgb = np.power(np.clip(rgb, 0.0, 1.0), gamma)

    if not np.isclose(highlights, 1.0):
        rgb = 1.0 - np.power(np.clip(1.0 - rgb, 0.0, 1.0), highlights)

    array[..., :3] = np.clip(rgb * 255.0, 0, 255)
    return Image.fromarray(array.astype(np.uint8), "RGBA")


def refine_alpha(image: Image.Image, radius: int) -> Image.Image:
    """Feather the alpha channel with a Gaussian/median blur combo."""

    if radius <= 0:
        return image

    array = np.array(image.convert("RGBA"))
    alpha = array[..., 3]
    ksize = max(1, radius * 2 + 1)
    if ksize % 2 == 0:
        ksize += 1
    alpha = cv2.GaussianBlur(alpha, (ksize, ksize), 0)
    if radius >= 2:
        alpha = cv2.medianBlur(alpha, ksize)
    array[..., 3] = alpha
    return Image.fromarray(array, "RGBA")


def compose_background(
    image: Image.Image,
    *,
    mode: str,
    color: RGBAColor,
    gradient_start: RGBAColor,
    gradient_end: RGBAColor,
    background_image_path: Optional[str],
) -> Image.Image:
    """Composite ``image`` onto a background defined by ``mode``."""

    image = image.convert("RGBA")
    width, height = image.size

    if mode == "transparent":
        return image

    if mode == "solid":
        background = Image.new("RGBA", (width, height), color)
    elif mode == "gradient":
        background = create_gradient_background((width, height), gradient_start, gradient_end)
    elif mode == "image" and background_image_path:
        try:
            bg = Image.open(background_image_path).convert("RGBA")
            background = bg.resize((width, height), Image.LANCZOS)
        except Exception:
            background = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    else:
        background = Image.new("RGBA", (width, height), (255, 255, 255, 255))

    return Image.alpha_composite(background, image)


def create_gradient_background(
    size: Tuple[int, int],
    start_color: RGBAColor,
    end_color: RGBAColor,
) -> Image.Image:
    width, height = size
    gradient = np.linspace(0, 1, width, dtype=np.float32)
    array = np.zeros((height, width, 4), dtype=np.float32)
    for i in range(4):
        array[..., i] = start_color[i] + (end_color[i] - start_color[i]) * gradient
    array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array, "RGBA")
