import os
import cv2
import numpy as np
import random
import json

# Paths
input_folder = r'C:\collected_sim_no_obstacles' # Change this to the folder containing your images
output_folder = r'C:\Users\boula\DataSet_Augmentation-\AUGMENT3OUTPUT' # Change this to the folder where you want to save the augmented images

os.makedirs(output_folder, exist_ok=True)

# Helper Function 
def clamp_and_convert(img):
    return np.clip(img, 0, 255).astype(np.uint8)

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

def glass_blur(scale, img):
    factor = [2, 5, 6, 9, 12][scale]
    height, width = img.shape[:2]
    for _ in range(factor):
        rand_x = np.clip(np.random.randint(-1, 2, (height, width)) + np.arange(width), 0, width - 1)
        rand_y = np.clip(np.random.randint(-1, 2, (height, width)) + np.arange(height)[:, None], 0, height - 1)
        img = img[rand_y, rand_x]
    return img

def motion_blur(scale, img):
    size = [2, 4, 6, 8, 10][scale]
    kernel = np.zeros((size, size))
    kernel[int(size / 2), :] = np.ones(size)
    kernel = kernel / size
    return clamp_and_convert(cv2.filter2D(img, -1, kernel))

def zoom_blur(scale, img):
    scale_factors = [1.01, 1.1, 1.2, 1.3, 1.4][scale]
    zoomed = cv2.resize(img, None, fx=scale_factors, fy=scale_factors, interpolation=cv2.INTER_LINEAR)
    center = tuple(np.array(zoomed.shape[:2]) // 2)
    crop = tuple(np.array(img.shape[:2]) // 2)
    return zoomed[center[0] - crop[0]:center[0] + crop[0], center[1] - crop[1]:center[1] + crop[1]]

def increase_brightness(scale, img):
    factor = [1.1, 1.2, 1.3, 1.5, 1.7][scale]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def contrast(scale, img):
    factor = [1.1, 1.2, 1.3, 1.5, 1.7][scale]
    pivot = 127.5
    return clamp_and_convert(pivot + (img - pivot) * factor)

def pixelate(scale, img):
    factor = [0.85, 0.55, 0.35, 0.2, 0.1][scale]
    h, w = img.shape[:2]
    temp = cv2.resize(img, (int(w * factor), int(h * factor)), interpolation=cv2.INTER_AREA)
    return cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)

def jpeg_filter(scale, img):
    quality = [95, 75, 50, 30, 10][scale]
    _, encoded = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)

def elastic(scale, img):
    alpha, sigma = [(2, 0.4), (3, 0.75), (5, 0.9), (7, 1.2), (10, 1.5)][scale]
    dx = cv2.GaussianBlur((np.random.rand(*img.shape[:2]) * 2 - 1) * alpha, (7, 7), sigma)
    dy = cv2.GaussianBlur((np.random.rand(*img.shape[:2]) * 2 - 1) * alpha, (7, 7), sigma)
    x, y = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def shear_image(scale, img):
    shear_factor = [0.1, 0.2, 0.3, 0.4, 0.5][scale]
    M = np.array([[1, shear_factor, 0], [0, 1, 0]], dtype=np.float32)
    return clamp_and_convert(cv2.warpAffine(img, M, (img.shape[1], img.shape[0])))

def grayscale_filter(scale, img):
    severity = [0.1, 0.2, 0.3, 0.4, 0.5][scale]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(img, 1 - severity, gray_rgb, severity, 0)
  
# 5 new perturbations
def radial_distortion(scale, img):
    """Simulates fish-eye effect for distorted perspectives."""
    h, w = img.shape[:2]
    dist_coeff = [0.02, 0.04, 0.06, 0.08, 0.10][scale]
    K = np.array([[w, 0, w//2], [0, w, h//2], [0, 0, 1]])
    dist = np.array([dist_coeff, 0, 0, 0])
    map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, K, (w, h), 5)
    return cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)

def add_shadow(scale, img):
    """Adds random shadows to simulate real-world lighting variations."""
    h, w = img.shape[:2]
    mask = np.zeros_like(img, dtype=np.uint8)
    shadow_intensity = [0.3, 0.4, 0.5, 0.6, 0.7][scale]
    x1, x2 = np.random.randint(0, w//2), np.random.randint(w//2, w)
    y1, y2 = np.random.randint(0, h//2), np.random.randint(h//2, h)
    cv2.fillPoly(mask, [np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])], (50, 50, 50))
    return cv2.addWeighted(img, 1, mask, -shadow_intensity, 0)

def perspective_warp(scale, img):
    """Applies perspective distortion to simulate extreme turning angles."""
    h, w = img.shape[:2]
    shift = [5, 10, 15, 20, 25][scale]
    src_pts = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    dst_pts = np.float32([[shift, shift], [w-shift, 0], [0, h], [w, h-shift]])
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(img, M, (w, h))

def directional_motion_blur(scale, img):
    """Simulates motion blur in sharp turns."""
    size = [5, 10, 15, 20, 25][scale]
    kernel = np.zeros((size, size))
    angle = np.random.choice([0, 45, 90, 135])
    if angle in [45, 135]:
        np.fill_diagonal(kernel, 1)
    else:
        kernel[size // 2, :] = np.ones(size)
    kernel /= kernel.sum()
    return cv2.filter2D(img, -1, kernel)

# --- Image Validation 
def is_valid_image(img, min_threshold=10, max_threshold=245):
    mean_intensity = np.mean(img)
    std_deviation = np.std(img)
    if mean_intensity < min_threshold or mean_intensity > max_threshold or std_deviation < min_threshold:
        return False
    return True

# --- Augmentation Execution (Using Only the 5 New Perturbations) 
#for the training you can apply all of them but first test these 4 new perturbations.
def augment_images_with_individual_json(input_folder, output_folder):
    perturbations = [#gaussian_noise, poisson_noise, impulse_noise, defocus_blur, glass_blur, motion_blur, zoom_blur, increase_brightness, contrast, pixelate, jpeg_filter, elastic, shear_image, grayscale_filter 
      radial_distortion, add_shadow, perspective_warp, directional_motion_blur] 

    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                img = cv2.imread(image_path)
                if img is None:
                    continue

                img = np.clip(img, 0, 255).astype(np.uint8)  
                original_name = os.path.splitext(file)[0]
                json_records = []

                num_perturbations = min(len(perturbations), random.randint(3, 5))
                selected = random.sample(perturbations, num_perturbations)

                for perturbation in selected:
                    scale = random.randint(0, 4)
                    augmented_img = perturbation(scale, img)
                    if not is_valid_image(augmented_img):
                        continue

                    # Save augmented image
                    aug_name = f"{original_name}_{perturbation.__name__}.png"
                    output_path = os.path.join(output_folder, aug_name)
                    cv2.imwrite(output_path, augmented_img)

                    # Add record to JSON
                    json_records.append(aug_name)

                # Ensure the JSON file records all augmentations for this image
                json_path = os.path.join(output_folder, f"{original_name}_record.json")
                existing_records = {}

                # If the JSON file already exists, load it to update records
                if os.path.exists(json_path):
                    with open(json_path, 'r') as json_file:
                        existing_records = json.load(json_file)

                # Update the JSON with new perturbations
                existing_records[file] = json_records

                # Write back to the JSON file
                with open(json_path, 'w') as json_file:
                    json.dump(existing_records, json_file, indent=4)

                print(f"Processed and saved images & records for: {file}")

# Run the augmentation process
augment_images_with_individual_json(input_folder, output_folder)
