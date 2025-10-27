from pathlib import Path

from PIL import Image

from removebg_app import pipeline
from removebg_app.pipeline import ImageProcessor, ImageRenderer
from removebg_app.state import ProcessedImage


class PassthroughRemover:
    def remove(self, image: Image.Image) -> Image.Image:  # pragma: no cover - simple stub
        return image


def make_document(color=(0, 0, 0, 0)) -> ProcessedImage:
    image = Image.new("RGBA", (4, 4), color)
    document = ProcessedImage(Path("dummy.png"), image)
    document.state.edge_smooth = 0
    document.state.target_size = image.size
    return document


def test_renderer_applies_solid_background():
    renderer = ImageRenderer()
    document = make_document()
    document.state.background_mode = "solid"
    document.state.background_color = (10, 20, 30, 255)

    result = renderer.render(document)
    assert result.size == (4, 4)
    assert result.getpixel((0, 0)) == (10, 20, 30, 255)


def test_renderer_respects_target_size():
    renderer = ImageRenderer()
    document = make_document()
    document.state.target_size = (2, 2)

    result = renderer.render(document)
    assert result.size == (2, 2)


def test_renderer_produces_preview_thumbnail():
    renderer = ImageRenderer()
    document = make_document()
    document.state.target_size = (1000, 1000)

    preview = renderer.render(document, preview_size=(100, 100))
    assert preview.width <= 100
    assert preview.height <= 100


def test_processor_export_flattens_to_rgb(tmp_path):
    renderer = ImageRenderer()
    processor = ImageProcessor(remover=PassthroughRemover(), renderer=renderer)
    document = make_document(color=(255, 0, 0, 128))
    document.state.background_mode = "solid"
    document.state.background_color = (0, 255, 0, 255)

    output_path = tmp_path / "output.jpg"
    processor.export(document, output_path)

    saved = Image.open(output_path)
    assert saved.mode == "RGB"
    assert saved.size == document.state.target_size


def test_build_document_skips_refine_for_preprocessed(monkeypatch):
    renderer = ImageRenderer()
    processor = ImageProcessor(remover=PassthroughRemover(), renderer=renderer)

    calls = []

    def fake_refine(image, radius):  # pragma: no cover - patched behaviour
        calls.append(radius)
        return image

    monkeypatch.setattr(pipeline, "refine_alpha", fake_refine)

    image = Image.new("RGBA", (8, 8), (255, 255, 255, 0))
    document = processor.build_document(image, Path("stub.png"), pre_refined=True)

    assert document.background_free.size == (8, 8)
    assert calls == []
