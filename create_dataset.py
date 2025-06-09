import numpy as np
import os
import random
from PIL import Image
from torchvision.datasets import MNIST
from torchvision import transforms
from tqdm import tqdm

# === Configuration ===
base_colors = {
    "red": (1.0, 0.0, 0.0),
    "green": (0.0, 1.0, 0.0),
    "blue": (0.0, 0.0, 1.0)
}
train_counts = np.geomspace(1000, 10, num=30, dtype=int)
test_samples_per_class = 100
output_dir = "longtailed_colormnist"

# === Setup output folders ===
for split in ["train", "test"]:
    for i in range(30):
        os.makedirs(os.path.join(output_dir, split, f"class_{i:02d}"), exist_ok=True)

# === Load MNIST ===
transform = transforms.ToTensor()
mnist_train = MNIST(root="./data", train=True, download=True, transform=transform)
mnist_test = MNIST(root="./data", train=False, download=True, transform=transform)

# === Create class ID mapping ===
class_map = {(digit, color): digit * 3 + idx for digit in range(10) for idx, color in enumerate(base_colors)}

def apply_color(img_tensor, color_name):
    img_np = img_tensor.squeeze().numpy()
    r, g, b = base_colors[color_name]
    rgb_img = np.stack([img_np * r, img_np * g, img_np * b], axis=-1)
    rgb_img = (rgb_img * 255).astype(np.uint8)
    return Image.fromarray(rgb_img)

# === Generate TRAIN set ===
used_counts = {i: 0 for i in range(30)}

print("Generating training data...")
for img, label in tqdm(mnist_train):
    for color in base_colors:
        class_id = class_map[(label, color)]
        if used_counts[class_id] < train_counts[class_id]:
            colored_img = apply_color(img, color)
            fname = f"{label}_{color}_{used_counts[class_id]}.png"
            save_path = os.path.join(output_dir, "train", f"class_{class_id:02d}", fname)
            colored_img.save(save_path)
            used_counts[class_id] += 1

# === Generate TEST set ===
test_counts = {i: 0 for i in range(30)}
print("Generating test data...")
for img, label in tqdm(mnist_test):
    for color in base_colors:
        class_id = class_map[(label, color)]
        if test_counts[class_id] < test_samples_per_class:
            colored_img = apply_color(img, color)
            fname = f"{label}_{color}_{test_counts[class_id]}.png"
            save_path = os.path.join(output_dir, "test", f"class_{class_id:02d}", fname)
            colored_img.save(save_path)
            test_counts[class_id] += 1
