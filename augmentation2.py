import os
import cv2
import numpy as np
import random
import json

# Paths
input_folder = r'C:\Users\boula\PRAKTIKUMSIM2REAL\Practicum_sim2real\content\logs\collected_sim_no_obstacles'  
output_folder = r'C:\Users\boula\PRAKTIKUMSIM2REAL\Practicum_sim2real\DataSet_Augmentation\2.Training_Augmentation\outputCORRECT'  
json_record_path = os.path.join(output_folder, "augmentation_records.json")  # Path for the JSON record

# Ensure the output directory exists
os.makedirs(output_folder, exist_ok=True)

# Helper Function to Clamp and Convert
def clamp_and_convert(img):
    return np.clip(img, 0, 255).astype(np.uint8)

# Perturbation Functions (15 total)
def gaussian_noise(scale, img):
    factor = [0.03, 0.06, 0.12, 0.18, 0.22][scale]
    x = img.astype(np.float32) / 255.0
    noisy = x + np.random.normal(size=x.shape, scale=factor)
    return clamp_and_convert(noisy * 255)

def poisson_noise(scale, img):
    factor = [120, 105, 87, 55, 30][scale]
    x = img.astype(np.float32) / 255.0
    noisy = np.random.poisson(x * factor) / factor
    return clamp_and_convert(noisy * 255)

def impulse_noise(scale, img):
    factor = [0.01, 0.02, 0.04, 0.065, 0.10][scale]
    img = img.copy()
    num_salt = int(factor * img.size * 0.5)
    coords = [np.random.randint(0, i - 1, num_salt) for i in img.shape[:2]]
    img[tuple(coords)] = 255
    num_pepper = int(factor * img.size * 0.5)
    coords = [np.random.randint(0, i - 1, num_pepper) for i in img.shape[:2]]
    img[tuple(coords)] = 0
    return img

def defocus_blur(scale, img):
    factor = [2, 5, 6, 9, 12][scale]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (factor, factor))
    return clamp_and_convert(cv2.filter2D(img, -1, kernel))

# ... (Include other perturbation functions here as in your current script)

# Validation Function
def is_valid_image(img, min_threshold=10, max_threshold=245):
    """
    Validates the augmented image to ensure it is not black, white, or unusable.
    """
    mean_intensity = np.mean(img)
    std_deviation = np.std(img)

    if mean_intensity < min_threshold or mean_intensity > max_threshold or std_deviation < min_threshold:
        return False
    return True

# Augmentation Pipeline with JSON Records
def augment_images_with_json(input_folder, output_folder, json_record_path):
    perturbations = [
        gaussian_noise, poisson_noise, impulse_noise, defocus_blur, 
        # Add other perturbation functions here...
    ]
    augmentation_records = {}

    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                img = cv2.imread(image_path)
                if img is None:
                    continue
                
                img = clamp_and_convert(img)  # Ensure uint8 format
                selected = random.sample(perturbations, random.randint(3, 5))
                augmented_files = []

                for perturbation in selected:
                    scale = random.randint(0, 4)
                    augmented_img = perturbation(scale, img)

                    # Validate augmented image
                    if not is_valid_image(augmented_img):
                        print(f"Invalid image discarded (black/white): {file}")
                        continue

                    # Save the valid augmented image
                    new_file_name = f"aug_{file}"
                    output_path = os.path.join(output_folder, new_file_name)
                    cv2.imwrite(output_path, augmented_img)
                    augmented_files.append(new_file_name)
                    print(f"Saved valid image: {output_path}")
                
                # Update JSON record
                augmentation_records[file] = augmented_files

    # Save JSON records
    with open(json_record_path, 'w') as json_file:
        json.dump(augmentation_records, json_file, indent=4)
        print(f"JSON record saved at: {json_record_path}")

# Run the Augmentation
augment_images_with_json(input_folder, output_folder, json_record_path)
