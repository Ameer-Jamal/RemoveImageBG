import cv2
import numpy as np
from PIL import Image, ImageFilter


def refine_edges(image_path, output_path):
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


# Paths to the images
input_path = "/mnt/data/Warp Icon-Python-BgRemoved.png"
output_path = "/mnt/data/Warp Icon-Python-BgRemoved-Refined.png"

# Refine the edges
refine_edges(input_path, output_path)