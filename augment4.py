import os
import cv2
import numpy as np
import random
import json

# Paths
input_folder = r'C:\collected_sim_no_obstacles' #input folder
output_folder = r'D:\augment4OUTput' #output folder
os.makedirs(output_folder, exist_ok=True)

# Helper Function 
def clamp_and_convert(img):
    return np.clip(img, 0, 255).astype(np.uint8)

# Refined use of Perturbation Functions (PerturbationDrive)

def gaussian_noise(scale, img):
    factor = [0.02, 0.04, 0.08, 0.12, 0.18][scale]
    noisy = img.astype(np.float32) / 255.0 + np.random.normal(size=img.shape, scale=factor)
    return clamp_and_convert(noisy * 255)

def poisson_noise(scale, img):
    factor = [130, 110, 90, 65, 40][scale]
    noisy = np.random.poisson(img.astype(np.float32) / 255.0 * factor) / factor
    return clamp_and_convert(noisy * 255)

def motion_blur(scale, img):
    size = [2, 3, 5, 7, 10][scale]
    kernel = np.zeros((size, size))
    kernel[int(size / 2), :] = np.ones(size)
    kernel /= size
    return clamp_and_convert(cv2.filter2D(img, -1, kernel))

def brightness_adjustment(scale, img):
    factor = [1.05, 1.1, 1.2, 1.3, 1.5][scale]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def contrast_adjustment(scale, img):
    factor = [1.05, 1.1, 1.2, 1.3, 1.5][scale]
    return clamp_and_convert(127.5 + (img - 127.5) * factor)

def add_shadow(scale, img):
    """Adds a random shadow to simulate real-world lighting."""
    h, w = img.shape[:2]
    mask = np.zeros_like(img, dtype=np.uint8)
    shadow_intensity = [0.2, 0.3, 0.4, 0.5, 0.6][scale]
    x1, x2 = np.random.randint(0, w // 2), np.random.randint(w // 2, w)
    y1, y2 = np.random.randint(0, h // 2), np.random.randint(h // 2, h)
    cv2.fillPoly(mask, [np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])], (50, 50, 50))
    return cv2.addWeighted(img, 1, mask, -shadow_intensity, 0)

def radial_distortion(scale, img):
    """Simulates a slight fish-eye effect."""
    h, w = img.shape[:2]
    dist_coeff = [0.015, 0.03, 0.045, 0.06, 0.075][scale]
    K = np.array([[w, 0, w//2], [0, w, h//2], [0, 0, 1]])
    dist = np.array([dist_coeff, 0, 0, 0])
    map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, K, (w, h), 5)
    return cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)

def jpeg_compression(scale, img):
    quality = [90, 80, 60, 40, 20][scale]
    _, encoded = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)

# --- Image Validation ---
def is_valid_image(img, min_threshold=10, max_threshold=245):
    mean_intensity = np.mean(img)
    std_deviation = np.std(img)
    return min_threshold < mean_intensity < max_threshold and std_deviation > min_threshold

# --- Augmentation Execution ---
def augment_images(input_folder, output_folder):
    perturbations = [gaussian_noise, poisson_noise, motion_blur, 
                     brightness_adjustment, contrast_adjustment, add_shadow, 
                     radial_distortion, jpeg_compression]

    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                img = cv2.imread(image_path)
                if img is None:
                    continue

                img = clamp_and_convert(img)  
                original_name = os.path.splitext(file)[0]
                json_records = []

                num_perturbations = random.randint(2, 3)  # Apply fewer perturbations per image
                selected = random.sample(perturbations, num_perturbations)

                for perturbation in selected:
                    scale = random.randint(0, 4)
                    augmented_img = perturbation(scale, img)
                    if not is_valid_image(augmented_img):
                        continue

                    aug_name = f"{original_name}_{perturbation.__name__}.png"
                    output_path = os.path.join(output_folder, aug_name)
                    cv2.imwrite(output_path, augmented_img)
                    json_records.append(aug_name)

                json_path = os.path.join(output_folder, f"{original_name}_record.json")
                with open(json_path, 'w') as json_file:
                    json.dump({file: json_records}, json_file, indent=4)

                print(f"Processed: {file}")

# Run the augmentation process
augment_images(input_folder, output_folder)
