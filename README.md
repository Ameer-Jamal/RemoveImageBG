# RemoveImageBG

A cross-platform PyQt5 desktop application for intelligent background removal and rapid image cleanup. The app now ships with a modernised processing pipeline (powered by the **IS-Net** model bundled with `rembg`), a productivity-focused GUI, batch automation tooling, and export presets that keep transparency intact.

## Highlights

- **State-of-the-art matting** – Uses the high-quality `isnet-general-use` session from `rembg` for accurate extractions without requiring a GPU.
- **Batch aware** – Drop files, add entire folders, or queue paths via the file picker. A multi-process worker keeps the UI fluid while saturating available CPU cores.
- **Non-destructive editing** – Hue, contrast, brightness, shadow, highlight, edge-feather, rotation, resize, crop, and background settings are tracked per image with undo/redo history.
- **Background blending studio** – Replace the transparent canvas with solid colours, horizontal gradients, or custom images and preview results instantly.
- **Multiple export targets** – Save single images or entire batches as PNG, JPEG, TIFF, or WebP with smart transparency fallbacks.
- **Script-friendly helpers** – `postProcessing.py` exposes reusable alpha-feathering and compositing utilities for CLI or headless pipelines.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> **Python**: 3.12 or newer is recommended for the latest `rembg` wheels, faster standard-library primitives, and ONNX runtime packages.

## Usage

```bash
python main.py
```

### Getting started in the UI

1. **Add imagery** – Click **Add Images** or **Add Folder**, or drag & drop files/folders onto the window.
2. **Wait for processing** – The progress bar reports background removal status. Items appear in the list as they finish.
3. **Fine-tune** – Select an image to reveal its preview and adjustment controls. Hue shifts operate in ±180°, tonal sliders use photographic terminology, and edge feathering eliminates matte halos.
4. **Swap backgrounds** – Choose between transparency, solid colour, gradients (set both endpoints), or a custom background image.
5. **Tidy composition** – Lock aspect ratio when resizing, rotate in 90° increments, or crop via pixel-perfect dialog. Clear buttons revert crop/rotation.
6. **Manage history** – Use **Undo/Redo** to traverse saved states. Sliders snapshot on release while discrete actions (rotate, crop, colour pickers, etc.) are recorded immediately.
7. **Export** – Save the selected item or the full batch. JPEG/WebP exports automatically composite against white when transparency is unavailable.

### Keyboard & convenience features

- **Ctrl/Cmd + Z / Shift + Ctrl/Cmd + Z** map to the Undo/Redo buttons through the operating system window manager.
- Double-clicking a list entry reselects it and recentres the preview.
- Dragging new files onto the window reuses the active processing session for maximum throughput.

## Command-line helpers

For scripted workflows, `postProcessing.py` can be called directly to feather an existing alpha matte:

```bash
python postProcessing.py input.png output.png --radius 5
```

The module also exposes `smooth_alpha` and `composite_background` functions for integration in your own pipelines.

## Recommended libraries and rationale

| Library / API | Why it is included or suggested |
| ------------- | -------------------------------- |
| [`rembg`](https://github.com/danielgatis/rembg) | Provides ONNX-backed foreground extraction with IS-Net, delivering excellent quality on CPUs while remaining open source. |
| [`opencv-python`](https://pypi.org/project/opencv-python/) | Used for efficient alpha-channel refinement (Gaussian and median filters) and colour-space conversions for hue shifts. |
| [`Pillow`](https://pillow.readthedocs.io/) | Handles image I/O, compositing, rotation, resizing, and enhancement APIs in a single dependency. |
| [`PyQt5`](https://www.riverbankcomputing.com/static/Docs/PyQt5/) | Mature, cross-platform desktop GUI toolkit with great widget coverage for productivity apps. |
| [`remove.bg` API](https://www.remove.bg/api) *(optional)* | Commercial SaaS with strong handling of edge cases (e.g. hair, motion blur). Consider for teams needing SLAs or web integrations. |
| [`MODNet`](https://github.com/ZHKKKe/MODNet) *(optional)* | Deep matting model with strong portrait performance; ideal if you plan GPU-accelerated workflows. |
| [`BackgroundRemover`](https://github.com/nadermx/backgroundremover) *(optional)* | Wraps multiple SOTA models (U²-Net, IS-Net, etc.) for experimentation and can be swapped in if you require CLI-first tooling. |

## Architecture notes

- The application initialises a single `rembg` session (`isnet-general-use`) and reuses it for every image to avoid repeated model downloads and warmup costs.
- Folder ingestion now fans out across a `ProcessPoolExecutor`, initialised once per worker process to reuse `rembg` sessions and maximise throughput on multi-core hardware.
- Images are stored alongside an immutable source copy; rendering always starts from the matte produced by `rembg` to keep edits non-destructive.
- History snapshots capture adjustment parameters rather than bitmap copies, greatly reducing memory usage for large batches.
- The processing core now lives in the reusable ``removebg_app`` package with
  ``BackgroundRemover`` (model session), ``ImageRenderer`` (stateful compositing),
  and ``ImageProcessor`` (orchestration/export) classes for clearer separation of
  concerns.
- Alpha refinement uses Gaussian + median filtering to soften boundaries before background compositing.

## Future enhancements

- **Cloud offloading** – Integrate optional upload-to-process flows (AWS Lambda, Azure Functions, or remove.bg) for extremely large batches.
- **GPU acceleration** – Detect CUDA/Metal availability and switch to GPU-enabled `onnxruntime` builds for faster matting.
- **Interactive cropping** – Swap the numeric crop dialog for a rubber-band selection overlay in the preview widget.
- **Mobile companion** – Package the processing core into a REST API or use Qt for Mobile to deliver the same workflow on phones/tablets.
- **Preset manager** – Allow saving/loading adjustment presets for consistent brand outputs across image sets.

## Contributing

1. Fork the repository and create a feature branch.
2. Install dependencies with the pinned `requirements.txt` to ensure consistent models.
3. Run `python -m compileall .` before submitting to catch syntax errors.
4. Submit a pull request summarising user-facing changes and new recommendations.

## License

MIT License © Contributors. See `LICENSE` (if present) or repository metadata.

## Testing

Run the automated suite (including new renderer/export unit tests) with:

```bash
pytest
```
