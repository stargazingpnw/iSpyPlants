python# preview_data.py
# Purpose: Quick sanity check to verify the dataset loaded correctly.
# Run this before training to confirm images and labels are lined up properly.

import os
from scipy.io import loadmat
from PIL import Image
import matplotlib.pyplot as plt

# -------------------------
# Paths to data files
# -------------------------
data_dir = 'data/jpg'
labels_file = 'data/imagelabels.mat'

# -------------------------
# Load labels from .mat file
# The ['labels'][0] extracts the array from the matlab format
# -------------------------
labels = loadmat(labels_file)['labels'][0]

# Print basic dataset stats
print(f"Total images: {len(os.listdir(data_dir))}")
print(f"Total labels: {len(labels)}")
print(f"Number of classes: {len(set(labels))}")

# -------------------------
# Preview the first 5 images
# zfill(5) pads the number with zeros e.g. 1 -> 00001
# -------------------------
fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for i, ax in enumerate(axes):
    img_path = os.path.join(data_dir, f'image_{str(i+1).zfill(5)}.jpg')
    img = Image.open(img_path)
    ax.imshow(img)
    ax.set_title(f'Label: {labels[i]}')
    ax.axis('off')

plt.tight_layout()
plt.show()