"""Processing pipeline coordinating background removal and rendering."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from PIL import Image

from .effects import apply_enhancements, apply_hue_shift, compose_background, refine_alpha
from .state import ProcessedImage


class BackgroundRemover:
    """Wrapper around ``rembg`` sessions to ease testing."""

    def __init__(self, session_name: str = "isnet-general-use") -> None:
        from rembg import new_session

        self.session_name = session_name
        self._session = new_session(session_name)

    def remove(self, image: Image.Image) -> Image.Image:
        from rembg import remove

        return remove(image, session=self._session)


class ImageRenderer:
    """Transform a :class:`ProcessedImage` into a rendered :class:`Image`."""

    def render(
        self,
        document: ProcessedImage,
        *,
        preview_size: Optional[Tuple[int, int]] = None,
    ) -> Image.Image:
        state = document.state
        image = document.background_free.copy().convert("RGBA")

        image = apply_hue_shift(image, state.hue_shift)
        image = apply_enhancements(image, state)
        image = refine_alpha(image, state.edge_smooth)

        if state.crop_box:
            image = image.crop(state.crop_box)

        if state.rotation:
            image = image.rotate(state.rotation, expand=True, resample=Image.BICUBIC)

        if state.target_size:
            width, height = state.target_size
            image = image.resize((max(1, width), max(1, height)), Image.LANCZOS)

        image = compose_background(
            image,
            mode=state.background_mode,
            color=state.background_color,
            gradient_start=state.gradient_start,
            gradient_end=state.gradient_end,
            background_image_path=state.background_image_path,
        )

        if preview_size:
            preview = image.copy()
            preview.thumbnail(preview_size, Image.LANCZOS)
            return preview
        return image


class ImageProcessor:
    """High-level service tying together loading, processing, and rendering."""

    def __init__(
        self,
        remover: Optional[BackgroundRemover] = None,
        renderer: Optional[ImageRenderer] = None,
    ) -> None:
        self.remover = remover or BackgroundRemover()
        self.renderer = renderer or ImageRenderer()

    def process_path(self, path: Path) -> ProcessedImage:
        image = Image.open(path).convert("RGBA")
        return self.process_image(image, path)

    def process_image(self, image: Image.Image, source_path: Path) -> ProcessedImage:
        background_free = self.remover.remove(image)
        return self.build_document(background_free, source_path)

    def build_document(
        self,
        background_free: Image.Image,
        source_path: Path,
        *,
        pre_refined: bool = False,
    ) -> ProcessedImage:
        processed = background_free.convert("RGBA")
        if not pre_refined:
            processed = refine_alpha(processed, radius=3)
        else:
            processed = processed.copy()

        document = ProcessedImage(source_path=source_path, background_free=processed)
        document.state.target_size = processed.size
        document.snapshot()
        return document

    def render(self, document: ProcessedImage, *, preview_size: Optional[Tuple[int, int]] = None) -> Image.Image:
        return self.renderer.render(document, preview_size=preview_size)

    def export(self, document: ProcessedImage, target: Path) -> None:
        image = self.render(document)
        extension = target.suffix.lower()
        if extension not in {".png", ".tiff", ".tif"} and image.mode == "RGBA":
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background
        target.parent.mkdir(parents=True, exist_ok=True)
        image.save(target)
