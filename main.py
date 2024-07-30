import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QProgressBar, QSlider, QHBoxLayout
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from rembg import remove
from PIL import Image, ImageEnhance
import numpy as np
import cv2
import os
from pathlib import Path


class BackgroundRemoverApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Image Background Removal')
        self.setGeometry(100, 100, 800, 800)

        layout = QVBoxLayout()

        self.select_button = QPushButton('Select Image', self)
        self.select_button.clicked.connect(self.select_image)
        layout.addWidget(self.select_button)

        self.progress_bar = QProgressBar(self)
        layout.addWidget(self.progress_bar)

        self.image_label = QLabel(self)
        layout.addWidget(self.image_label)

        self.output_path_label = QLabel(self)
        layout.addWidget(self.output_path_label)

        # Hue slider
        hue_layout = QHBoxLayout()
        hue_label = QLabel('Hue')
        self.hue_slider = QSlider(Qt.Horizontal)
        self.hue_slider.setMinimum(0)
        self.hue_slider.setMaximum(360)
        self.hue_slider.setValue(180)
        self.hue_slider.valueChanged.connect(self.apply_adjustments)
        hue_layout.addWidget(hue_label)
        hue_layout.addWidget(self.hue_slider)
        layout.addLayout(hue_layout)

        # Contrast slider
        contrast_layout = QHBoxLayout()
        contrast_label = QLabel('Contrast')
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setMinimum(0)
        self.contrast_slider.setMaximum(200)
        self.contrast_slider.setValue(100)
        self.contrast_slider.valueChanged.connect(self.apply_adjustments)
        contrast_layout.addWidget(contrast_label)
        contrast_layout.addWidget(self.contrast_slider)
        layout.addLayout(contrast_layout)

        # Brightness slider
        brightness_layout = QHBoxLayout()
        brightness_label = QLabel('Brightness')
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setMinimum(0)
        self.brightness_slider.setMaximum(200)
        self.brightness_slider.setValue(100)
        self.brightness_slider.valueChanged.connect(self.apply_adjustments)
        brightness_layout.addWidget(brightness_label)
        brightness_layout.addWidget(self.brightness_slider)
        layout.addLayout(brightness_layout)

        # Blacks slider
        blacks_layout = QHBoxLayout()
        blacks_label = QLabel('Blacks')
        self.blacks_slider = QSlider(Qt.Horizontal)
        self.blacks_slider.setMinimum(0)
        self.blacks_slider.setMaximum(200)
        self.blacks_slider.setValue(100)
        self.blacks_slider.valueChanged.connect(self.apply_adjustments)
        blacks_layout.addWidget(blacks_label)
        blacks_layout.addWidget(self.blacks_slider)
        layout.addLayout(blacks_layout)

        # Whites slider
        whites_layout = QHBoxLayout()
        whites_label = QLabel('Whites')
        self.whites_slider = QSlider(Qt.Horizontal)
        self.whites_slider.setMinimum(0)
        self.whites_slider.setMaximum(200)
        self.whites_slider.setValue(100)
        self.whites_slider.valueChanged.connect(self.apply_adjustments)
        whites_layout.addWidget(whites_label)
        whites_layout.addWidget(self.whites_slider)
        layout.addLayout(whites_layout)

        self.save_button = QPushButton('Save Edited File', self)
        self.save_button.clicked.connect(self.save_edited_image)
        layout.addWidget(self.save_button)

        self.setLayout(layout)

    def select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Open file', '', 'Image files (*.jpg *.jpeg *.png)')
        if file_path:
            self.process_image(file_path)

    def remove_background_rembg(self, file_path):
        input_image = Image.open(file_path)
        output_dir = str(Path.home() / "Downloads")
        output_file_name = f"{os.path.splitext(os.path.basename(file_path))[0]}-Python-BgRemoved.png"
        output_path = os.path.join(output_dir, output_file_name)

        # Remove the background
        output_image = remove(input_image)

        # Save the result
        output_image.save(output_path)
        return output_path

    def process_image(self, file_path):
        self.progress_bar.setRange(0, 0)

        # Perform background removal using rembg
        self.output_path_rembg = self.remove_background_rembg(file_path)

        self.progress_bar.setRange(0, 1)

        # Load the result image
        self.result_image = Image.open(self.output_path_rembg)
        self.result_image.thumbnail((400, 400))  # Resize to fit in the window

        # Store original image for adjustments
        self.original_image = self.result_image.copy()

        # Update the image label
        self.apply_adjustments()

        # Show the output file path
        self.output_path_label.setText(f"Output saved to: {self.output_path_rembg}")

    def apply_adjustments(self):
        if hasattr(self, 'original_image'):
            image = self.original_image.convert('RGBA')

            # Separate alpha channel
            r, g, b, a = image.split()

            # Apply hue adjustment
            hue_value = self.hue_slider.value()
            hsv_image = Image.merge("RGB", (r, g, b)).convert('HSV')
            np_hsv = np.array(hsv_image)
            np_hsv[..., 0] = np.clip(hue_value, 0, 255)
            hsv_image = Image.fromarray(np_hsv, 'HSV').convert('RGB')
            image = Image.merge('RGBA', (*hsv_image.split(), a))

            # Apply contrast adjustment
            contrast_value = self.contrast_slider.value() / 100.0
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(contrast_value)

            # Apply brightness adjustment
            brightness_value = self.brightness_slider.value() / 100.0
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(brightness_value)

            # Apply blacks adjustment (shadows)
            blacks_value = self.blacks_slider.value() / 100.0
            image = ImageEnhance.Brightness(image).enhance(blacks_value)

            # Apply whites adjustment (highlights)
            whites_value = self.whites_slider.value() / 100.0
            image = ImageEnhance.Brightness(image).enhance(whites_value)

            self.update_image_label(image)

    def update_image_label(self, image=None):
        if image is None:
            image = self.result_image

        # Convert PIL image to QImage
        image = image.convert("RGBA")
        data = image.tobytes("raw", "RGBA")
        q_image = QImage(data, image.size[0], image.size[1], QImage.Format_RGBA8888)
        q_pixmap = QPixmap.fromImage(q_image)

        self.image_label.setPixmap(q_pixmap)

    def refine_edges(self, image_path, output_path):
        # Load the image
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        # Extract the alpha channel
        b, g, r, a = cv2.split(image)

        # Use the alpha channel as a mask
        mask = a.copy()

        # Apply a Gaussian blur to the mask to smooth the edges
        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        # Create a new image with refined edges
        refined_image = cv2.merge([b, g, r, mask])

        # Save the refined image
        cv2.imwrite(output_path, refined_image)

    def save_edited_image(self):
        if hasattr(self, 'original_image'):
            file_path, _ = QFileDialog.getSaveFileName(self, 'Save file', '', 'PNG files (*.png)')
            if file_path:
                # Apply all adjustments to the original image
                image = self.original_image.convert('RGBA')

                # Separate alpha channel
                r, g, b, a = image.split()

                hue_value = self.hue_slider.value()
                hsv_image = Image.merge("RGB", (r, g, b)).convert('HSV')
                np_hsv = np.array(hsv_image)
                np_hsv[..., 0] = np.clip(hue_value, 0, 255)
                hsv_image = Image.fromarray(np_hsv, 'HSV').convert('RGB')
                image = Image.merge('RGBA', (*hsv_image.split(), a))

                contrast_value = self.contrast_slider.value() / 100.0
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(contrast_value)

                brightness_value = self.brightness_slider.value() / 100.0
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(brightness_value)

                # Apply blacks adjustment (shadows)
                blacks_value = self.blacks_slider.value() / 100.0
                image = ImageEnhance.Brightness(image).enhance(blacks_value)

                # Apply whites adjustment (highlights)
                whites_value = self.whites_slider.value() / 100.0
                image = ImageEnhance.Brightness(image).enhance(whites_value)

                temp_path = file_path.replace('.png', '_temp.png')
                image.save(temp_path)

                # Refine edges
                self.refine_edges(temp_path, file_path)

                # Remove temporary file
                os.remove(temp_path)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = BackgroundRemoverApp()
    ex.show()
    sys.exit(app.exec_())
