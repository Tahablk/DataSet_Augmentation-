#ISSUE IN IMPORTING PERTURBATION FUNCTIONS
#from PerturbationDrive.perturbationdrive.perturbationfuncs import (gaussian_noise, motion_blur, fog_filter, frost_filter, contrast, increase_brightness, rotate_image)
# Import the functions from perturbationfuncs.py
#import sys
#sys.path.append(r"C:\Users\boula\PRAKTIKUMSIM2REAL\Practicum_sim2real\PerturbationDrive\perturbationdrive")
#with open("PerturbationDrive\perturbationdrive\perturbationfuncs.py", 'r') as f:
 #   exec(f.read())

import os
import cv2
import json
import numpy as np

import os
import cv2
import json
import numpy as np

# Perturbation Functions (Image and JSON)
import os
import cv2
import json
import numpy as np

# Perturbation Functions (Image and JSON)
def gaussian_noise(scale, img):
    """Apply Gaussian noise to an image."""
    mean = 0
    var = 10 * scale
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, img.shape).astype('uint8')
    noisy_image = cv2.addWeighted(img, 0.75, gauss, 0.25, 0)
    return noisy_image

def motion_blur(scale, img):
    """Apply motion blur to an image."""
    kernel_size = scale * 2 + 1
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    kernel /= kernel_size
    blurred_image = cv2.filter2D(img, -1, kernel)
    return blurred_image

def rotate_image(scale, img):
    """Rotate an image."""
    angle = [10, 20, 45, 90, 180][scale]
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(img, matrix, (w, h))
    return rotated_image

# Augment Image Only
def augment_image(image_path, output_folder):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")
    
    # List of perturbations
    perturbations = [
        ("gaussian_noise", gaussian_noise(2, image)),
        ("motion_blur", motion_blur(2, image)),
        ("rotate", rotate_image(2, image))
    ]

    for suffix, perturbed_image in perturbations:
        # Save perturbed image
        augmented_image_name = f"{os.path.splitext(os.path.basename(image_path))[0]}_{suffix}.png"
        augmented_image_path = os.path.join(output_folder, augmented_image_name)
        cv2.imwrite(augmented_image_path, perturbed_image)

    print(f"Augmented images saved for {image_path}.")

# Augment Dataset
def augment_dataset_with_perturbations(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)

                print(f"Processing image: {image_path}")
                try:
                    augment_image(image_path, output_folder)
                except Exception as e:
                    print(f"Failed to process {file}: {e}")
            else:
                print(f"Skipping non-image file: {file}")

    print(f"Augmented dataset saved to {output_folder}")

# Example Usage
input_folder = r"C:\Users\boula\PRAKTIKUMSIM2REAL\Practicum_sim2real\content\logs\collected_sim_no_obstacles"
output_folder = r"C:\Users\boula\PRAKTIKUMSIM2REAL\Practicum_sim2real\DataSet_Augmentation\outputPNG"

#augment_dataset_with_perturbations(input_folder, output_folder)

from PIL import Image
import os

# Paths
input_dir = "C:/Users/boula/PRAKTIKUMSIM2REAL/Practicum_sim2real/DataSet_Augmentation/outputPNG"
output_dir = "C:/Users/boula/PRAKTIKUMSIM2REAL/Practicum_sim2real/DataSet_Augmentation/outputPNG-converted"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Conversion settings
target_size = (315, 265)  # Match input size
target_mode = "RGBA"      # Match input mode
target_dpi = (120, 120)   # Match input DPI

# Process images
for filename in os.listdir(input_dir):
    if filename.endswith(".png"):
        img_path = os.path.join(input_dir, filename)
        img = Image.open(img_path)

        # Convert image
        # Convert image
        img = img.resize(target_size, Image.LANCZOS)
        img = img.convert(target_mode)


        # Save converted image
        output_path = os.path.join(output_dir, filename)
        img.save(output_path, format="PNG", dpi=target_dpi)

print(f"Converted images saved to {output_dir}")


