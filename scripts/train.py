Replace everything in train.py with this:
python

# train.py
# Purpose: Train a flower/plant classifier using transfer learning.
# We use a pretrained ResNet50 model and replace its final layer
# to classify our 102 flower categories instead of ImageNet's 1000 classes.

import os
import torch
import torchvision
from torchvision import datasets, transforms, models
import torch.nn as nn
import torch.optim as optim

data_dir = 'data'    # root folder containing train/ and valid/ subfolders
model_dir = 'model'  # folder where the best model will be saved

# use GPU if available, otherwise fall back to CPU
# GPU training is ~10x faster — we'll use Google Colab's free GPU for this
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on: {device}")

# -------------------------
# Data Transforms
# Training: randomly crop and flip images to help the model generalize better
# Validation: just resize and center crop — no random augmentation needed
# Normalize: these specific values are the ImageNet mean/std that ResNet expects
# -------------------------
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),   # randomly crop to 224x224
    transforms.RandomHorizontalFlip(),   # randomly flip image left/right
    transforms.ToTensor(),               # convert image to PyTorch tensor
    transforms.Normalize([0.485, 0.456, 0.406],   # normalize red channel
                         [0.229, 0.224, 0.225])    # normalize green/blue channels
])

valid_transforms = transforms.Compose([
    transforms.Resize(256),              # resize shortest side to 256
    transforms.CenterCrop(224),          # crop center 224x224
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -------------------------
# Load Datasets
# ImageFolder automatically assigns class labels based on folder names
# -------------------------
print("Loading datasets...")
train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transforms)
valid_dataset = datasets.ImageFolder(os.path.join(data_dir, 'valid'), transform=valid_transforms)

# DataLoader batches images and shuffles training data each epoch
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=False)

print(f"Training images: {len(train_dataset)}")
print(f"Validation images: {len(valid_dataset)}")

# -------------------------
# Load Pretrained Model
# ResNet50 was pretrained on ImageNet (1.2M images, 1000 classes)
# We freeze all its layers and only train our custom final layer
# This is called Transfer Learning — reusing knowledge from a large dataset
# -------------------------
print("Loading pretrained model...")
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# freeze all layers so we don't overwrite the pretrained knowledge
for param in model.parameters():
    param.requires_grad = False

# replace the final fully connected layer with our own for 102 classes
# the original ResNet50 fc layer outputs 1000 classes (ImageNet)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),  # compress features to 512
    nn.ReLU(),                              # activation function — adds non-linearity
    nn.Dropout(0.2),                        # randomly drop 20% of neurons to prevent overfitting
    nn.Linear(512, 102)                     # final output: 102 flower classes
)

model = model.to(device)  # move model to GPU or CPU

# -------------------------
# Loss Function and Optimizer
# CrossEntropyLoss: standard loss for multi-class classification
# Adam optimizer: only update our custom fc layer (requires_grad=True)
# lr=0.001: learning rate — how big each update step is
# -------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# -------------------------
# Training Loop
# Each epoch = one full pass through the training data
# After each epoch we check accuracy on the validation set
# We save the model whenever validation accuracy improves
# -------------------------
epochs = 10
best_accuracy = 0

print("Starting training...")
for epoch in range(epochs):

    # --- Training Phase ---
    model.train()   # set model to training mode (enables dropout etc.)
    running_loss = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # move batch to GPU/CPU

        optimizer.zero_grad()           # clear gradients from previous step
        outputs = model(inputs)         # forward pass — get predictions
        loss = criterion(outputs, labels)  # calculate how wrong the predictions are
        loss.backward()                 # backward pass — calculate gradients
        optimizer.step()               # update model weights

        running_loss += loss.item()    # accumulate loss for this epoch

    # --- Validation Phase ---
    model.eval()    # set model to evaluation mode (disables dropout)
    correct = 0
    total = 0

    with torch.no_grad():  # don't calculate gradients during validation (saves memory)
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)  # get class with highest score
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(train_loader):.3f} | Val Accuracy: {accuracy:.2f}%")

    # save model if this is the best accuracy we've seen so far
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save({
            'model_state_dict': model.state_dict(),  # model weights
            'class_to_idx': train_dataset.class_to_idx,  # class name to index mapping
            'best_accuracy': best_accuracy
        }, os.path.join(model_dir, 'best_model.pth'))
        print(f"  ✅ New best model saved! ({accuracy:.2f}%)")

print(f"\nTraining complete! Best accuracy: {best_accuracy:.2f}%")