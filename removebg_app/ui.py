"""Qt UI wiring for the background removal workspace."""

from __future__ import annotations

import io
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
    QComboBox,
    QColorDialog,
    QMessageBox,
    QCheckBox,
    QSpinBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
)

from .pipeline import ImageProcessor
from .state import ImageState, ProcessedImage
from .workers import BackgroundRemovalWorker

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}


class CropDialog(QDialog):
    def __init__(self, width: int, height: int, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Crop Image")
        self._width = width
        self._height = height

        layout = QFormLayout(self)

        self.x_spin = QSpinBox()
        self.x_spin.setRange(0, max(0, width - 1))
        self.y_spin = QSpinBox()
        self.y_spin.setRange(0, max(0, height - 1))
        self.width_spin = QSpinBox()
        self.width_spin.setRange(1, max(1, width))
        self.height_spin = QSpinBox()
        self.height_spin.setRange(1, max(1, height))

        layout.addRow("Left (x)", self.x_spin)
        layout.addRow("Top (y)", self.y_spin)
        layout.addRow("Width", self.width_spin)
        layout.addRow("Height", self.height_spin)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def _on_accept(self) -> None:
        x = self.x_spin.value()
        y = self.y_spin.value()
        w = self.width_spin.value()
        h = self.height_spin.value()

        if x + w > self._width or y + h > self._height:
            QMessageBox.warning(
                self,
                "Invalid Crop",
                "Crop rectangle must stay within the image bounds.",
            )
            return
        self.accept()

    def crop_box(self) -> Tuple[int, int, int, int]:
        x = self.x_spin.value()
        y = self.y_spin.value()
        w = self.width_spin.value()
        h = self.height_spin.value()
        return (x, y, x + w, y + h)


class BackgroundRemoverApp(QWidget):
    def __init__(self, processor: Optional[ImageProcessor] = None):
        super().__init__()
        self.setWindowTitle("Intelligent Background Remover")
        self.setGeometry(80, 80, 1100, 720)
        self.setAcceptDrops(True)

        self.processor = processor or ImageProcessor()
        self.documents: List[ProcessedImage] = []
        self.current_doc: Optional[ProcessedImage] = None
        self.updating_controls = False
        self.updating_resize = False
        self.pending_paths: List[Path] = []
        self.worker: Optional[BackgroundRemovalWorker] = None
        self._current_batch_total: int = 0

        self._build_ui()

    def _build_ui(self) -> None:
        main_layout = QVBoxLayout()

        instructions = QLabel(
            "<b>Workflow:</b> Add images via the buttons below or by dragging files here. "
            "The app automatically removes backgrounds using the state-of-the-art "
            "IS-Net model from the rembg project. Fine-tune colours, replace the background, "
            "and export in multiple formats. Use the history controls to undo/redo edits."
        )
        instructions.setWordWrap(True)
        main_layout.addWidget(instructions)

        button_row = QHBoxLayout()
        self.add_images_button = QPushButton("Add Images")
        self.add_images_button.clicked.connect(self.select_images)
        button_row.addWidget(self.add_images_button)

        self.add_folder_button = QPushButton("Add Folder")
        self.add_folder_button.clicked.connect(self.select_folder)
        button_row.addWidget(self.add_folder_button)

        self.save_image_button = QPushButton("Export Selected…")
        self.save_image_button.clicked.connect(self.save_current_image)
        button_row.addWidget(self.save_image_button)

        self.save_all_button = QPushButton("Export All…")
        self.save_all_button.clicked.connect(self.save_all_images)
        button_row.addWidget(self.save_all_button)

        self.reset_button = QPushButton("Reset Adjustments")
        self.reset_button.clicked.connect(self.reset_adjustments)
        button_row.addWidget(self.reset_button)

        main_layout.addLayout(button_row)

        progress_row = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        progress_row.addWidget(QLabel("Processing progress:"))
        progress_row.addWidget(self.progress_bar)
        main_layout.addLayout(progress_row)

        content_layout = QHBoxLayout()

        list_panel = QVBoxLayout()
        list_panel.addWidget(QLabel("Processed images"))
        self.image_list = QListWidget()
        self.image_list.currentItemChanged.connect(self._on_selection_changed)
        list_panel.addWidget(self.image_list)

        history_row = QHBoxLayout()
        self.undo_button = QPushButton("Undo")
        self.undo_button.clicked.connect(self.undo)
        history_row.addWidget(self.undo_button)
        self.redo_button = QPushButton("Redo")
        self.redo_button.clicked.connect(self.redo)
        history_row.addWidget(self.redo_button)
        list_panel.addLayout(history_row)

        content_layout.addLayout(list_panel, 1)

        right_panel = QVBoxLayout()

        self.preview_label = QLabel("Preview will appear here once an image is processed.")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(400, 400)
        right_panel.addWidget(self.preview_label, 3)

        controls_group = QGroupBox("Adjustments")
        controls_layout = QVBoxLayout()

        colour_group = QGroupBox("Colour & Tone")
        colour_layout = QVBoxLayout()

        self.hue_slider = self._create_slider(-180, 180, 0, "Shift foreground hue")
        colour_layout.addLayout(self._wrap_slider("Hue shift", self.hue_slider))

        self.contrast_slider = self._create_slider(10, 300, 100, "Contrast (100 = original)")
        colour_layout.addLayout(self._wrap_slider("Contrast", self.contrast_slider))

        self.brightness_slider = self._create_slider(10, 300, 100, "Brightness (100 = original)")
        colour_layout.addLayout(self._wrap_slider("Brightness", self.brightness_slider))

        self.blacks_slider = self._create_slider(10, 300, 100, "Shadow lift or crush")
        colour_layout.addLayout(self._wrap_slider("Shadows", self.blacks_slider))

        self.whites_slider = self._create_slider(10, 300, 100, "Highlight boost or roll-off")
        colour_layout.addLayout(self._wrap_slider("Highlights", self.whites_slider))

        colour_group.setLayout(colour_layout)
        controls_layout.addWidget(colour_group)

        edge_group = QGroupBox("Edge refinement")
        edge_layout = QVBoxLayout()
        self.edge_slider = self._create_slider(0, 15, 3, "Feather edges to reduce halos")
        edge_layout.addLayout(self._wrap_slider("Feather radius", self.edge_slider))
        edge_group.setLayout(edge_layout)
        controls_layout.addWidget(edge_group)

        background_group = QGroupBox("Background replacement")
        background_layout = QVBoxLayout()

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode"))
        self.background_mode_combo = QComboBox()
        self.background_mode_combo.addItems([
            "Transparent",
            "Solid colour",
            "Linear gradient",
            "Custom image",
        ])
        self.background_mode_combo.currentIndexChanged.connect(self._on_background_mode_changed)
        self.background_mode_combo.activated.connect(lambda _: self.capture_state())
        mode_row.addWidget(self.background_mode_combo)
        background_layout.addLayout(mode_row)

        color_row = QHBoxLayout()
        self.color_button = QPushButton("Choose colour…")
        self.color_button.clicked.connect(self.choose_background_colour)
        color_row.addWidget(self.color_button)
        self.gradient_start_button = QPushButton("Gradient start…")
        self.gradient_start_button.clicked.connect(lambda: self.choose_gradient_colour(True))
        color_row.addWidget(self.gradient_start_button)
        self.gradient_end_button = QPushButton("Gradient end…")
        self.gradient_end_button.clicked.connect(lambda: self.choose_gradient_colour(False))
        color_row.addWidget(self.gradient_end_button)
        self.background_image_button = QPushButton("Choose image…")
        self.background_image_button.clicked.connect(self.choose_background_image)
        color_row.addWidget(self.background_image_button)
        background_layout.addLayout(color_row)

        background_group.setLayout(background_layout)
        controls_layout.addWidget(background_group)

        geometry_group = QGroupBox("Geometry & canvas")
        geometry_layout = QVBoxLayout()

        resize_row = QHBoxLayout()
        resize_row.addWidget(QLabel("Width"))
        self.resize_width_spin = QSpinBox()
        self.resize_width_spin.setRange(1, 10000)
        self.resize_width_spin.valueChanged.connect(self._on_resize_width_changed)
        resize_row.addWidget(self.resize_width_spin)
        resize_row.addWidget(QLabel("Height"))
        self.resize_height_spin = QSpinBox()
        self.resize_height_spin.setRange(1, 10000)
        self.resize_height_spin.valueChanged.connect(self._on_resize_height_changed)
        resize_row.addWidget(self.resize_height_spin)
        geometry_layout.addLayout(resize_row)

        self.aspect_ratio_checkbox = QCheckBox("Lock aspect ratio")
        self.aspect_ratio_checkbox.setChecked(True)
        self.aspect_ratio_checkbox.toggled.connect(self._on_aspect_ratio_toggled)
        geometry_layout.addWidget(self.aspect_ratio_checkbox)

        rotate_row = QHBoxLayout()
        self.rotate_left_button = QPushButton("⟲ Rotate left")
        self.rotate_left_button.clicked.connect(lambda: self.rotate(-90))
        rotate_row.addWidget(self.rotate_left_button)
        self.rotate_right_button = QPushButton("⟳ Rotate right")
        self.rotate_right_button.clicked.connect(lambda: self.rotate(90))
        rotate_row.addWidget(self.rotate_right_button)
        self.clear_rotation_button = QPushButton("Reset rotation")
        self.clear_rotation_button.clicked.connect(lambda: self.set_rotation(0))
        rotate_row.addWidget(self.clear_rotation_button)
        geometry_layout.addLayout(rotate_row)

        crop_row = QHBoxLayout()
        self.crop_button = QPushButton("Crop…")
        self.crop_button.clicked.connect(self.crop_image)
        crop_row.addWidget(self.crop_button)
        self.clear_crop_button = QPushButton("Clear crop")
        self.clear_crop_button.clicked.connect(self.clear_crop)
        crop_row.addWidget(self.clear_crop_button)
        geometry_layout.addLayout(crop_row)

        geometry_group.setLayout(geometry_layout)
        controls_layout.addWidget(geometry_group)

        controls_group.setLayout(controls_layout)
        right_panel.addWidget(controls_group, 2)

        content_layout.addLayout(right_panel, 2)
        main_layout.addLayout(content_layout)

        self.setLayout(main_layout)

        for slider in (
            self.hue_slider,
            self.contrast_slider,
            self.brightness_slider,
            self.blacks_slider,
            self.whites_slider,
            self.edge_slider,
        ):
            slider.valueChanged.connect(self.update_from_controls)
            slider.sliderReleased.connect(self.capture_state)

        self.background_mode_combo.currentIndexChanged.connect(self.update_from_controls)
        self.update_background_controls()
        self.update_history_buttons()

    def _create_slider(self, minimum: int, maximum: int, value: int, tooltip: str) -> QSlider:
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(minimum)
        slider.setMaximum(maximum)
        slider.setValue(value)
        slider.setSingleStep(1)
        slider.setPageStep(max(1, (maximum - minimum) // 10))
        slider.setToolTip(tooltip)
        return slider

    def _wrap_slider(self, label_text: str, slider: QSlider) -> QHBoxLayout:
        layout = QHBoxLayout()
        label = QLabel(label_text)
        label.setMinimumWidth(110)
        layout.addWidget(label)
        layout.addWidget(slider)
        return layout

    def dragEnterEvent(self, event):  # type: ignore[override]
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):  # type: ignore[override]
        paths = []
        for url in event.mimeData().urls():
            if url.isLocalFile():
                path = Path(url.toLocalFile())
                if path.is_dir():
                    paths.extend(self._collect_images(path))
                elif path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                    paths.append(path)
        if paths:
            self.process_files(paths)

    def select_images(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select image files",
            str(Path.home()),
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.webp)",
        )
        paths = [Path(p) for p in files]
        if paths:
            self.process_files(paths)

    def select_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select folder containing images", str(Path.home()))
        if folder:
            paths = self._collect_images(Path(folder))
            if not paths:
                QMessageBox.information(self, "No images found", "The selected folder does not contain supported image files.")
                return
            self.process_files(paths)

    def _collect_images(self, folder: Path) -> List[Path]:
        return [p for p in folder.rglob("*") if p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS]

    def process_files(self, paths: List[Path]) -> None:
        if not paths:
            return

        self.pending_paths.extend(paths)
        self._start_worker_if_idle()

    def _start_worker_if_idle(self) -> None:
        if self.worker and self.worker.isRunning():
            return

        if not self.pending_paths:
            return

        batch = self.pending_paths
        self.pending_paths = []
        self._current_batch_total = len(batch)
        self.progress_bar.setValue(0)

        session_name = getattr(self.processor.remover, "session_name", "isnet-general-use")
        self.worker = BackgroundRemovalWorker(
            batch,
            session_name=session_name,
        )
        self.worker.item_ready.connect(self._on_worker_item_ready)
        self.worker.progress.connect(self._on_worker_progress)
        self.worker.failed.connect(self._on_worker_failed)
        self.worker.finished.connect(self._on_worker_finished)
        self._set_processing_state(True)
        self.worker.start()

    def _set_processing_state(self, active: bool) -> None:
        if not active and not self.pending_paths:
            self.progress_bar.setValue(0)

    def _on_worker_item_ready(self, path_str: str, payload: bytes) -> None:
        path = Path(path_str)
        with Image.open(io.BytesIO(payload)) as image:
            background_free = image.convert("RGBA").copy()

        document = self.processor.build_document(background_free, path, pre_refined=True)
        self.documents.append(document)
        item = QListWidgetItem(document.display_name)
        item.setData(Qt.UserRole, document)
        self.image_list.addItem(item)

        if self.current_doc is None:
            self.image_list.setCurrentItem(item)

    def _on_worker_progress(self, completed: int, total: int) -> None:
        if total <= 0:
            return
        progress = int((completed / total) * 100)
        self.progress_bar.setValue(progress)

    def _on_worker_failed(self, path_str: str, error: str) -> None:
        QMessageBox.critical(
            self,
            "Processing error",
            f"Failed to process {Path(path_str).name}: {error}",
        )

    def _on_worker_finished(self) -> None:
        if self.worker:
            self.worker.deleteLater()
            self.worker = None

        self.progress_bar.setValue(100 if self._current_batch_total else 0)
        self._set_processing_state(False)
        self._current_batch_total = 0
        self._start_worker_if_idle()

    def capture_state(self) -> None:
        if not self.current_doc:
            return
        self.current_doc.snapshot()
        self.update_history_buttons()

    def undo(self) -> None:
        if not self.current_doc:
            return
        state = self.current_doc.history.undo()
        if not state:
            return
        self.current_doc.state = state
        self._sync_controls_from_state()
        self.update_preview()
        self.update_history_buttons()

    def redo(self) -> None:
        if not self.current_doc:
            return
        state = self.current_doc.history.redo()
        if not state:
            return
        self.current_doc.state = state
        self._sync_controls_from_state()
        self.update_preview()
        self.update_history_buttons()

    def update_history_buttons(self) -> None:
        can_undo = bool(self.current_doc and self.current_doc.history.can_undo)
        can_redo = bool(self.current_doc and self.current_doc.history.can_redo)
        self.undo_button.setEnabled(can_undo)
        self.redo_button.setEnabled(can_redo)

    def _on_selection_changed(self, current: Optional[QListWidgetItem], previous: Optional[QListWidgetItem]) -> None:
        if current is None:
            self.current_doc = None
            self.preview_label.setText("Select an image to begin editing.")
            self.update_history_buttons()
            return
        document = current.data(Qt.UserRole)
        if not isinstance(document, ProcessedImage):
            return
        self.current_doc = document
        self._sync_controls_from_state()
        self.update_preview()
        self.update_history_buttons()

    def _sync_controls_from_state(self) -> None:
        if not self.current_doc:
            return
        self.updating_controls = True
        state = self.current_doc.state
        self.hue_slider.setValue(state.hue_shift)
        self.contrast_slider.setValue(int(state.contrast * 100))
        self.brightness_slider.setValue(int(state.brightness * 100))
        self.blacks_slider.setValue(int(state.blacks * 100))
        self.whites_slider.setValue(int(state.whites * 100))
        self.edge_slider.setValue(state.edge_smooth)

        mode_index = {
            "transparent": 0,
            "solid": 1,
            "gradient": 2,
            "image": 3,
        }.get(state.background_mode, 0)
        self.background_mode_combo.setCurrentIndex(mode_index)

        width, height = state.target_size or self.current_doc.background_free.size
        self.updating_resize = True
        self.resize_width_spin.setValue(width)
        self.resize_height_spin.setValue(height)
        self.updating_resize = False

        self.update_background_controls()
        self.updating_controls = False

    def _on_background_mode_changed(self) -> None:
        self.update_background_controls()
        self.update_from_controls()

    def update_background_controls(self) -> None:
        mode = self.background_mode_combo.currentIndex()
        self.color_button.setEnabled(mode == 1)
        self.gradient_start_button.setEnabled(mode == 2)
        self.gradient_end_button.setEnabled(mode == 2)
        self.background_image_button.setEnabled(mode == 3)

    def update_from_controls(self) -> None:
        if not self.current_doc or self.updating_controls:
            return

        state = self.current_doc.state
        state.hue_shift = self.hue_slider.value()
        state.contrast = self.contrast_slider.value() / 100.0
        state.brightness = self.brightness_slider.value() / 100.0
        state.blacks = self.blacks_slider.value() / 100.0
        state.whites = self.whites_slider.value() / 100.0
        state.edge_smooth = self.edge_slider.value()

        mode_map = {
            0: "transparent",
            1: "solid",
            2: "gradient",
            3: "image",
        }
        state.background_mode = mode_map.get(self.background_mode_combo.currentIndex(), "transparent")

        state.target_size = (
            self.resize_width_spin.value(),
            self.resize_height_spin.value(),
        )

        self.update_background_controls()
        self.update_preview()

    def choose_background_colour(self) -> None:
        color = (
            QColorDialog.getColor(QColor(*self.current_doc.state.background_color), self, "Select background colour")
            if self.current_doc
            else QColor()
        )
        if not color.isValid() or not self.current_doc:
            return
        self.current_doc.state.background_color = (
            color.red(),
            color.green(),
            color.blue(),
            255,
        )
        self.update_preview()
        self.capture_state()

    def choose_gradient_colour(self, start: bool) -> None:
        if not self.current_doc:
            return
        initial = self.current_doc.state.gradient_start if start else self.current_doc.state.gradient_end
        color = QColorDialog.getColor(QColor(*initial), self, "Select gradient colour")
        if not color.isValid():
            return
        rgba = (color.red(), color.green(), color.blue(), 255)
        if start:
            self.current_doc.state.gradient_start = rgba
        else:
            self.current_doc.state.gradient_end = rgba
        self.update_preview()
        self.capture_state()

    def choose_background_image(self) -> None:
        if not self.current_doc:
            return
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select background image",
            str(Path.home()),
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.webp)",
        )
        if not file_path:
            return
        self.current_doc.state.background_image_path = file_path
        self.update_preview()
        self.capture_state()

    def _on_aspect_ratio_toggled(self, checked: bool) -> None:
        if not self.current_doc:
            return
        if checked:
            original_width, original_height = self.current_doc.background_free.size
            if original_width and original_height:
                base_aspect = original_height / original_width
                current_width = self.resize_width_spin.value()
                new_height = max(1, int(round(current_width * base_aspect)))
                self.updating_resize = True
                self.resize_height_spin.setValue(new_height)
                self.updating_resize = False
        self.update_from_controls()
        self.capture_state()

    def _on_resize_width_changed(self, value: int) -> None:
        if not self.current_doc or self.updating_resize:
            return
        state = self.current_doc.state
        if self.aspect_ratio_checkbox.isChecked():
            original_width, original_height = self.current_doc.background_free.size
            aspect = original_height / original_width if original_width else 1
            self.updating_resize = True
            new_height = max(1, int(round(value * aspect)))
            self.resize_height_spin.setValue(new_height)
            self.updating_resize = False
        state.target_size = (value, self.resize_height_spin.value())
        self.update_preview()
        if not self.updating_resize:
            self.capture_state()

    def _on_resize_height_changed(self, value: int) -> None:
        if not self.current_doc or self.updating_resize:
            return
        state = self.current_doc.state
        if self.aspect_ratio_checkbox.isChecked():
            original_width, original_height = self.current_doc.background_free.size
            aspect = original_width / original_height if original_height else 1
            self.updating_resize = True
            new_width = max(1, int(round(value * aspect)))
            self.resize_width_spin.setValue(new_width)
            self.updating_resize = False
        state.target_size = (self.resize_width_spin.value(), value)
        self.update_preview()
        if not self.updating_resize:
            self.capture_state()

    def rotate(self, degrees: int) -> None:
        if not self.current_doc:
            return
        state = self.current_doc.state
        state.rotation = (state.rotation + degrees) % 360
        self.update_preview()
        self.capture_state()

    def set_rotation(self, degrees: int) -> None:
        if not self.current_doc:
            return
        self.current_doc.state.rotation = degrees % 360
        self.update_preview()
        self.capture_state()

    def crop_image(self) -> None:
        if not self.current_doc:
            return
        base_width, base_height = self.current_doc.background_free.size
        dialog = CropDialog(base_width, base_height, self)
        if dialog.exec_() == QDialog.Accepted:
            self.current_doc.state.crop_box = dialog.crop_box()
            self.update_preview()
            self.capture_state()

    def clear_crop(self) -> None:
        if not self.current_doc:
            return
        self.current_doc.state.crop_box = None
        self.update_preview()
        self.capture_state()

    def update_preview(self) -> None:
        if not self.current_doc:
            return
        preview_image = self.processor.render(self.current_doc, preview_size=(600, 600))
        q_image = self._to_qimage(preview_image)
        pixmap = QPixmap.fromImage(q_image)
        self.preview_label.setPixmap(pixmap)

    def _to_qimage(self, image: Image.Image) -> QImage:
        image = image.convert("RGBA")
        data = image.tobytes("raw", "RGBA")
        return QImage(data, image.width, image.height, QImage.Format_RGBA8888)

    def save_current_image(self) -> None:
        if not self.current_doc:
            QMessageBox.information(self, "No selection", "Select an image to export.")
            return
        suggested_name = f"{self.current_doc.source_path.stem}_processed.png"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export image",
            str(Path.home() / suggested_name),
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;TIFF (*.tiff);;WebP (*.webp)",
        )
        if not file_path:
            return
        self.processor.export(self.current_doc, Path(file_path))

    def save_all_images(self) -> None:
        if not self.documents:
            QMessageBox.information(self, "No images", "Add and process images before exporting.")
            return
        target_dir = QFileDialog.getExistingDirectory(
            self,
            "Select target directory",
            str(Path.home()),
        )
        if not target_dir:
            return
        target_path = Path(target_dir)
        for document in self.documents:
            export_name = f"{document.source_path.stem}_processed.png"
            self.processor.export(document, target_path / export_name)
        QMessageBox.information(self, "Export complete", f"Saved {len(self.documents)} images to {target_dir}.")

    def reset_adjustments(self) -> None:
        if not self.current_doc:
            return
        base = self.current_doc.background_free
        self.current_doc.state = ImageState(target_size=base.size)
        self.current_doc.snapshot()
        self._sync_controls_from_state()
        self.update_preview()


def run_app() -> int:
    app = QApplication.instance() or QApplication([])
    app.setApplicationName("Remove Image Background")
    window = BackgroundRemoverApp()
    window.show()
    return app.exec_()
