## predict.py
# Purpose: Takes a flower image and predicts which of the 102 classes it belongs to.
# Usage: python scripts/predict.py --image path/to/your/image.jpg

import torch
import argparse
from torchvision import models, transforms
from PIL import Image
import json

# -------------------------
# Argument Parser
# Lets us pass in the image path from the command line
# Example: python scripts/predict.py --image my_flower.jpg
# -------------------------
parser = argparse.ArgumentParser(description='Predict flower class from an image')
parser.add_argument('--image', type=str, required=True, help='Path to the image file')
parser.add_argument('--top_k', type=int, default=5, help='Return top K predictions')
args = parser.parse_args()

# -------------------------
# Load flower names mapping
# Maps class numbers to real flower names e.g. "54" -> "sunflower"
# -------------------------
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# -------------------------
# Load the trained model
# We rebuild the same architecture we used in training
# then load the saved weights into it
# -------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = models.resnet50(weights=None)  # create empty ResNet50
model.fc = torch.nn.Sequential(
    torch.nn.Linear(model.fc.in_features, 512),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.2),
    torch.nn.Linear(512, 102)  # 102 flower classes
)

# load the saved weights from our training
checkpoint = torch.load('model/best_model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()  # set to evaluation mode — disables dropout

class_to_idx = checkpoint['class_to_idx']  # maps class folder names to indices
idx_to_class = {v: k for k, v in class_to_idx.items()}  # reverse it: index -> class name

# -------------------------
# Image Preprocessing
# Must match exactly what we did during training
# otherwise the model gets confused
# -------------------------
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# load and preprocess the image
image = Image.open(args.image).convert('RGB')  # convert to RGB in case it's RGBA or grayscale
image_tensor = transform(image).unsqueeze(0).to(device)  # add batch dimension

# -------------------------
# Make Prediction
# -------------------------
with torch.no_grad():  # no need to calculate gradients for inference
    outputs = model(image_tensor)
    probabilities = torch.nn.functional.softmax(outputs, dim=1)  # convert to probabilities
    top_probs, top_indices = probabilities.topk(args.top_k)  # get top K predictions

# -------------------------
# Print Results
# -------------------------
print(f"\n🌸 Top {args.top_k} Predictions:")
print("-" * 30)
for prob, idx in zip(top_probs[0], top_indices[0]):
    class_num = idx_to_class[idx.item()]        # get class number from index
    flower_name = cat_to_name[str(class_num)]   # get real flower name from class number
    print(f"{flower_name.title()}: {prob.item()*100:.2f}%")