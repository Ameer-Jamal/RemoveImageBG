"""Command-line helpers mirroring the PyQt application's refinements."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image

from removebg_app.effects import compose_background as _compose_background
from removebg_app.effects import refine_alpha as _refine_alpha

RGBAColor = Tuple[int, int, int, int]


def smooth_alpha(image: Image.Image, radius: int = 3) -> Image.Image:
    """Feather an image's alpha channel to reduce jagged edges."""

    return _refine_alpha(image, radius)


def composite_background(
    foreground: Image.Image,
    mode: str = "transparent",
    *,
    color: RGBAColor = (255, 255, 255, 255),
    gradient_start: RGBAColor = (255, 255, 255, 255),
    gradient_end: RGBAColor = (0, 0, 0, 255),
    background_image: Optional[Path] = None,
) -> Image.Image:
    """Composite the ``foreground`` onto a background defined by ``mode``."""

    return _compose_background(
        foreground,
        mode=mode,
        color=color,
        gradient_start=gradient_start,
        gradient_end=gradient_end,
        background_image_path=str(background_image) if background_image else None,
    )


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
