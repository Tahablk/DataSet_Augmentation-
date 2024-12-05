#ISSUE IN IMPORTING PERTURBATION FUNCTIONS
#from PerturbationDrive.perturbationdrive.perturbationfuncs import (gaussian_noise, motion_blur, fog_filter, frost_filter, contrast, increase_brightness, rotate_image)

import os
import cv2
import numpy as np
import shutil
import json
from PIL import Image

# Paths
input_folder = r'C:\Users\boula\PRAKTIKUMSIM2REAL\Practicum_sim2real\content\logs\collected_sim_no_obstacles'
output_folder = r'C:\Users\boula\PRAKTIKUMSIM2REAL\Practicum_sim2real\DataSet_Augmentation\BEFOREFILTER'
final_output_folder = r'C:\Users\boula\PRAKTIKUMSIM2REAL\Practicum_sim2real\DataSet_Augmentation\FINALFOLDERDATA'

# Conversion settings
target_size = (320, 240)  # Match input size
target_mode = "RGB"       # Match input mode
target_dpi = (96, 96)     # Match input DPI

# Ensure output folders exist
os.makedirs(output_folder, exist_ok=True)
os.makedirs(final_output_folder, exist_ok=True)

# Perturbation Functions (Image)
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

def contrast(scale, img):
    """Adjust contrast of an image."""
    factor = [1.1, 1.2, 1.3, 1.5, 1.7][scale]
    pivot = 127.5
    return np.clip(pivot + (img - pivot) * factor, 0, 255).astype(np.uint8)

def saturation_filter(scale, img):
    """Adjust saturation of an image."""
    multiplier = [1.05, 1.15, 1.4, 1.65, 1.9][scale]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * multiplier, 0, 255)
    saturated = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return saturated

# Augment Images
def augment_image(image_path, output_folder, name_mapping):
    """
    Applies selected augmentations to an image.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")

    perturbations = [
        ("gaussian_noise", gaussian_noise(2, image)),
        ("motion_blur", motion_blur(2, image)),
        ("contrast", contrast(2, image)),
        ("saturation", saturation_filter(2, image))
    ]

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    for suffix, perturbed_image in perturbations:
        new_name = f"{base_name}_{suffix}.png"
        augmented_image_path = os.path.join(output_folder, new_name)
        cv2.imwrite(augmented_image_path, perturbed_image)
        name_mapping[base_name + ".png"] = new_name

# Copy and Update JSON Files
def update_json_files(input_folder, output_folder, name_mapping):
    """
    Updates JSON files to reflect augmented image names.
    """
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith('.json'):
                input_json_path = os.path.join(root, file)
                output_json_path = os.path.join(output_folder, file)

                # Copy and update JSON
                with open(input_json_path, 'r') as f:
                    data = json.load(f)

                # Update image names in JSON
                if isinstance(data, dict):
                    for key, value in data.items():
                        if value in name_mapping:
                            data[key] = name_mapping[value]

                elif isinstance(data, list):
                    data = [name_mapping.get(item, item) for item in data]

                with open(output_json_path, 'w') as f:
                    json.dump(data, f, indent=4)

                print(f"Updated JSON file: {output_json_path}")

# Resize and Convert Images
def resize_and_convert_images(input_folder, output_folder):
    """
    Resizes and converts images to the specified format.
    """
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                input_image_path = os.path.join(root, file)
                output_image_path = os.path.join(output_folder, file)

                # Resize and convert image
                img = Image.open(input_image_path)
                img = img.resize(target_size, Image.LANCZOS)
                img = img.convert(target_mode)
                img.save(output_image_path, format="PNG", dpi=target_dpi)

                print(f"Processed and saved: {output_image_path}")

# Main Workflow
def augment_and_prepare_dataset(input_folder, output_folder, final_output_folder):
    """
    Orchestrates the augmentation and preparation of the dataset.
    """
    name_mapping = {}

    # Step 1: Augment Images
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                print(f"Augmenting image: {image_path}")
                augment_image(image_path, output_folder, name_mapping)

    # Step 2: Copy and Update JSON Files
    update_json_files(input_folder, output_folder, name_mapping)

    # Step 3: Resize and Convert Images
    resize_and_convert_images(output_folder, final_output_folder)

    # Step 4: Copy JSON Files to Final Output
    for root, _, files in os.walk(output_folder):
        for file in files:
            if file.lower().endswith('.json'):
                shutil.copy(
                    os.path.join(root, file),
                    os.path.join(final_output_folder, file)
                )

    print(f"Final dataset prepared in: {final_output_folder}")

# Run the workflow
augment_and_prepare_dataset(input_folder, output_folder, final_output_folder)
