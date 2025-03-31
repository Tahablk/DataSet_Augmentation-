# 🧪 Image Augmentations for Sim2Real gap mitigation

This repository contains a set of **image augmentation tools** designed to support domain adaptation in autonomous driving research. These augmentations simulate real-world visual distortions, helping bridge the gap between synthetic and real-world driving data.

---

## 🌍 Project Context

This work is part of a practical course on **Testing of Data-Intensive Software Applications**, focused on training autonomous vehicle models to generalize from simulation to real-world environments. The augmentations developed here are specifically used to:

- Improve the robustness of a DAVE-2-based model trained in Unity simulations
- Increase image diversity without the need for more real-world data
- Simulate real-world noise, lighting conditions, and distortions

---

## 📦 Augmentation Modules

The repository includes the following implementations:

### ✅ Individual Augmentations

Each transformation is implemented in a standalone Python script:

- **`gaussian_noise(image)`**  
  Adds Gaussian sensor-like noise, mimicking low-light or poor sensor quality.

- **`adjust_brightness(image, factor)`**  
  Changes image brightness to simulate lighting variance like glare or dim scenes.

- **`apply_motion_blur(image, kernel_size)`**  
  Adds motion blur artifacts caused by fast movement or vibration.

- **`rotate_image(image, angle)`**  
  Rotates images to handle misaligned or tilted camera inputs.

### 🔀 `augmentDEF`: Composite Perturbation Mixer

This custom module applies a **mix of multiple perturbations**, either sequentially or at random. It can simulate complex real-world scenarios by combining effects like noise + blur + brightness change.

Useful for generating **edge-case images** and stress-testing model robustness.

---

## 🧰 Integration with Hannes Leonhard' Perturbation Drive

This project integrates with the **Perturbation Drive** library developed by Hannes for applying realistic image perturbations.

### Why use Perturbation Drive?

- ✅ 30+ built-in perturbations (fog, frost, shadows, rain, occlusion, etc.)
- ✅ Physically inspired effects for domain randomization
- ✅ Compatible with dataset pipelines

### Example Usage in Code

```python
from perturbation_drive import apply_perturbation
from PIL import Image

img = Image.open("input_image.png")

# Apply fog effect from Perturbation Drive
augmented = apply_perturbation(img, perturbation="fog", severity=3)
augmented.save("augmented_fog.png")
