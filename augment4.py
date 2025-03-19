import os
import cv2
import numpy as np
import random
import json

# Paths
input_folder = r'C:\collected_sim_no_obstacles' #input folder
output_folder = r'D:\augment4_StrongPerturbations' #output images and records (adapt it to yours as always)
os.makedirs(output_folder, exist_ok=True)

# Helper function to clamp pixel values
def clamp_and_convert(img):
    return np.clip(img, 0, 255).astype(np.uint8)

### --- 18 PERTURBATIONS (some were removed/made stronger) --- ###
def gaussian_noise(scale, img):
    factor = [0.05, 0.1, 0.2, 0.3, 0.4][scale]
    noisy = img.astype(np.float32) / 255.0 + np.random.normal(size=img.shape, scale=factor)
    return clamp_and_convert(noisy * 255)

def poisson_noise(scale, img):
    factor = [150, 120, 100, 80, 50][scale]
    noisy = np.random.poisson(img.astype(np.float32) / 255.0 * factor) / factor
    return clamp_and_convert(noisy * 255)

def impulse_noise(scale, img):
    factor = [0.02, 0.05, 0.08, 0.12, 0.18][scale]
    img = img.copy()
    num_salt = int(factor * img.size * 0.5)
    coords = [np.random.randint(0, i - 1, num_salt) for i in img.shape[:2]]
    img[tuple(coords)] = 255
    num_pepper = int(factor * img.size * 0.5)
    coords = [np.random.randint(0, i - 1, num_pepper) for i in img.shape[:2]]
    img[tuple(coords)] = 0
    return img

def glass_blur(scale, img):
    factor = [2, 5, 8, 12, 15][scale]
    height, width = img.shape[:2]
    for _ in range(factor):
        rand_x = np.clip(np.random.randint(-1, 2, (height, width)) + np.arange(width), 0, width - 1)
        rand_y = np.clip(np.random.randint(-1, 2, (height, width)) + np.arange(height)[:, None], 0, height - 1)
        img = img[rand_y, rand_x]
    return img

def motion_blur(scale, img):
    size = [5, 10, 15, 20, 30][scale]
    kernel = np.zeros((size, size))
    kernel[int(size / 2), :] = np.ones(size)
    kernel /= size
    return clamp_and_convert(cv2.filter2D(img, -1, kernel))

def zoom_blur(scale, img):
    scale_factors = [1.1, 1.3, 1.5, 1.7, 2.0][scale]
    zoomed = cv2.resize(img, None, fx=scale_factors, fy=scale_factors, interpolation=cv2.INTER_LINEAR)
    center = tuple(np.array(zoomed.shape[:2]) // 2)
    crop = tuple(np.array(img.shape[:2]) // 2)
    return zoomed[center[0] - crop[0]:center[0] + crop[0], center[1] - crop[1]:center[1] + crop[1]]

def contrast(scale, img):
    factor = [1.2, 1.4, 1.6, 1.8, 2.2][scale]
    pivot = 127.5
    return clamp_and_convert(pivot + (img - pivot) * factor)

def pixelate(scale, img):
    factor = [0.8, 0.6, 0.4, 0.2, 0.1][scale]
    h, w = img.shape[:2]
    temp = cv2.resize(img, (int(w * factor), int(h * factor)), interpolation=cv2.INTER_AREA)
    return cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)

def jpeg_filter(scale, img):
    quality = [90, 70, 50, 30, 10][scale]
    _, encoded = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)

def elastic(scale, img):
    alpha, sigma = [(2, 0.4), (5, 0.8), (10, 1.2), (15, 1.5), (20, 2.0)][scale]
    dx = cv2.GaussianBlur((np.random.rand(*img.shape[:2]) * 2 - 1) * alpha, (7, 7), sigma)
    dy = cv2.GaussianBlur((np.random.rand(*img.shape[:2]) * 2 - 1) * alpha, (7, 7), sigma)
    x, y = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR)

def shear_image(scale, img):
    shear_factor = [0.1, 0.2, 0.3, 0.4, 0.5][scale]
    M = np.array([[1, shear_factor, 0], [0, 1, 0]], dtype=np.float32)
    return clamp_and_convert(cv2.warpAffine(img, M, (img.shape[1], img.shape[0])))

def grayscale_filter(scale, img):
    severity = [0.2, 0.4, 0.6, 0.8, 1.0][scale]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(img, 1 - severity, gray_rgb, severity, 0)

### --- AUGMENTATION EXECUTION WITH JSON LOGGING + SCALE OF APPLIED PERTURBATION --- ###
def augment_images(input_folder, output_folder):
    perturbations = [
        gaussian_noise, poisson_noise, impulse_noise, glass_blur, motion_blur,
        zoom_blur, contrast, pixelate, jpeg_filter, elastic,
        shear_image, grayscale_filter
    ]

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

                num_perturbations = random.randint(3, 6)
                selected = random.sample(perturbations, num_perturbations)

                for perturbation in selected:
                    scale = random.randint(0, 4)
                    augmented_img = perturbation(scale, img)

                    aug_name = f"{original_name}_{perturbation.__name__}.png"
                    output_path = os.path.join(output_folder, aug_name)
                    cv2.imwrite(output_path, augmented_img)

                    json_records.append({"image": aug_name, "perturbation": perturbation.__name__, "scale": scale})

                json_path = os.path.join(output_folder, f"{original_name}_record.json")
                existing_records = []
                if os.path.exists(json_path):
                    with open(json_path, 'r') as json_file:
                        existing_records = json.load(json_file)

                existing_records.extend(json_records)

                with open(json_path, 'w') as json_file:
                    json.dump(existing_records, json_file, indent=4)

                print(f"Processed and logged: {file}")

# Run Augmentation
augment_images(input_folder, output_folder)
