python# prepare_data.py
# Purpose: Organizes raw images into train/valid folders by class label.
# PyTorch's ImageFolder expects this folder structure to work correctly.

import os
import shutil
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

data_dir = 'data/jpg'          # folder containing all raw images
labels_file = 'data/imagelabels.mat'  # matlab file containing class labels
output_dir = 'data'            # where train/ and valid/ folders will be created

labels = loadmat(labels_file)['labels'][0]  # extract label array from matlab format
images = sorted(os.listdir(data_dir))       # get sorted list of image filenames

# split into 80% train, 20% validation
# stratify=labels ensures each class is proportionally represented in both splits
# random_state=42 makes the split reproducible (same split every time you run it)
train_imgs, valid_imgs, train_labels, valid_labels = train_test_split(
    images, labels, test_size=0.2, random_state=42, stratify=labels
)

def copy_images(img_list, label_list, split_name):
    """Copy images into class subfolders under train/ or valid/."""
    print(f"Organizing {split_name} images...")
    for img, label in zip(img_list, label_list):
        class_dir = os.path.join(output_dir, split_name, str(label))  # e.g. data/train/77/
        os.makedirs(class_dir, exist_ok=True)  # create folder if it doesn't exist

        src = os.path.join(data_dir, img)       # source path
        dst = os.path.join(class_dir, img)      # destination path
        shutil.copy(src, dst)                   # copy image to class folder

    print(f"Done! {len(img_list)} images organized.")

copy_images(train_imgs, train_labels, 'train')  # organize training images
copy_images(valid_imgs, valid_labels, 'valid')  # organize validation images

print("All done! Dataset is ready for training.")