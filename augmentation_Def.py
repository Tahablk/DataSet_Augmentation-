import json
import os
import random
import cv2
import numpy as np

def augment_image(image_path):
    """
    Apply augmentation to a single image.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")
    
    # Random rotation
    angle = random.uniform(-30, 30)
    center = (image.shape[1] // 2, image.shape[0] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    rotated = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
    
    # Random brightness adjustment
    factor = random.uniform(0.6, 1.4)
    brightened = np.clip(rotated * factor, 0, 255).astype(np.uint8)
    
    # Add random noise
    noise = np.random.normal(0, 10, brightened.shape).astype(np.uint8)
    noisy_image = cv2.add(brightened, noise)
    
    return noisy_image

def augment_obstacles(obstacle_data):
    """
    Randomly modify obstacle positions and orientations.
    """
    augmented_obstacles = obstacle_data.copy()
    for idx, pose in enumerate(augmented_obstacles['obstacle_poses']):
        perturbation = [random.uniform(-1, 1) for _ in range(3)]
        augmented_obstacles['obstacle_poses'][idx] = [
            pose[i] + perturbation[i] for i in range(3)
        ]
    
    for idx, orientation in enumerate(augmented_obstacles['obstacle_orientations']):
        rotation_perturbation = [random.uniform(-10, 10) for _ in range(3)]
        augmented_obstacles['obstacle_orientations'][idx] = [
            orientation[i] + rotation_perturbation[i] for i in range(3)
        ]
    
    return augmented_obstacles

def augment_dataset(image_folder, obstacle_file, output_folder):
    """
    Augment the dataset and save the augmented data.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # Load and augment obstacles
    with open(obstacle_file, 'r') as f:
        obstacle_data = json.load(f)
    augmented_obstacle_data = augment_obstacles(obstacle_data)
    
    augmented_obstacle_file = os.path.join(output_folder, 'augmented_obstacles.json')
    with open(augmented_obstacle_file, 'w') as f:
        json.dump(augmented_obstacle_data, f, indent=4)
    
    # Traverse all subdirectories to find images
    for root, _, files in os.walk(image_folder):
        for image_name in files:
            if image_name.endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, image_name)
                print(f"Processing image: {image_path}")

                try:
                    augmented_image = augment_image(image_path)
                    
                    # Save augmented image
                    augmented_image_name = f"aug_{os.path.relpath(image_path, image_folder).replace(os.sep, '_')}"
                    augmented_image_path = os.path.join(output_folder, augmented_image_name)
                    cv2.imwrite(augmented_image_path, augmented_image)
                except Exception as e:
                    print(f"Failed to process {image_name}: {e}")
            else:
                print(f"Skipping non-image file: {image_name}")

    print(f"Augmented dataset saved to {output_folder}")


# Corrected Paths
image_folder = r"C:\Users\boula\PRAKTIKUMSIM2REAL\Practicum_sim2real\content\logs\collected_sim_no_obstacles"
obstacle_file = r"C:\Users\boula\PRAKTIKUMSIM2REAL\Practicum_sim2real\content\maps\map1_obstacles_4.json"
output_folder = r"C:\Users\boula\PRAKTIKUMSIM2REAL\Practicum_sim2real\DataSet_Augmentation\output_Folder_DEF"

augment_dataset(image_folder, obstacle_file, output_folder)


