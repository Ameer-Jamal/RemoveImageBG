# Image Background Removal and Adjustment Tool 
  
  ## Overview  This project provides a user-friendly GUI application for removing backgrounds from images and applying various adjustments like hue, contrast, brightness, shadows, and highlights. The application uses the `rembg` library for background removal and includes post-processing to refine the edges, ensuring high-quality output.  
  
  ## Features  
  - **Background Removal:** Remove backgrounds from images using the state-of-the-art `rembg` library.
  - **Adjustments:** Fine-tune images with sliders for:   - Hue   - Contrast   - Brightness   - Shadows (Blacks) 
  - Highlights (Whites) - **Edge Refinement:** Post-process images to smooth edges and eliminate artifacts.
  - **Save Functionality:** Save the adjusted and refined images with transparency preserved.  ## Requirements
  - Python 3.6+ - PyQt5 - rembg - pillow - opencv-python-headless 
  
  ## Installation  
  1. **Clone the Repository:**    ```bash    git clone https://github.com/Ameer-Jamal/image-bg-removal-tool.git    cd image-bg-removal-tool```
     
2. **Install Dependencies:**
    
    ```bash pip install -r requirements.txt```
    

## Usage

1. **Run the Application:**
    `python main.py`
    
2. **Using the GUI:**
    
    - **Select Image:** Click the "Select Image" button to choose an image file.
    - **Adjust Sliders:** Use the sliders to adjust hue, contrast, brightness, shadows, and highlights.
    - **Save Image:** Click the "Save Edited File" button to save the processed image.

## Screenshot
![image](https://github.com/user-attachments/assets/c9ca76e1-4b3a-416a-9630-49086df11912)


## Edge Refinement Details

The application includes a post-processing step to refine the edges of the processed images. This is achieved by applying a Gaussian blur to the alpha channel, smoothing the edges and eliminating artifacts.

## Code Structure

- **main.py:** Main application code including GUI and image processing functions.
- **postProcessing.py:** Module for edge refinement (integrated within `main.py`).

## Contributing

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -am 'Add some feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Create a new Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [rembg](https://github.com/danielgatis/rembg) - The background removal library used in this project.
- PyQt5 - For the graphical user interface.
