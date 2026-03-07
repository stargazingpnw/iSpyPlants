# app.py
# Purpose: A simple Flask web server that accepts an image and returns a plant prediction.
# This is the bridge between your AI model and the mobile app.
# Run it with: python app.py
# Test it by sending an image to: http://localhost:5000/predict

from flask import Flask, request, jsonify  # Flask is the web framework
from PIL import Image                       # for opening and processing images
import torch
from torchvision import models, transforms
import json
import io                                   # for reading image bytes from the request

# -------------------------
# Initialize Flask app
# -------------------------
app = Flask(__name__)

# -------------------------
# Load flower names mapping
# -------------------------
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# -------------------------
# Load the trained model
# Same as predict.py — rebuild architecture then load saved weights
# -------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = models.resnet50(weights=None)
model.fc = torch.nn.Sequential(
    torch.nn.Linear(model.fc.in_features, 512),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.2),
    torch.nn.Linear(512, 102)
)

checkpoint = torch.load('model/best_model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()  # set to evaluation mode

class_to_idx = checkpoint['class_to_idx']
idx_to_class = {v: k for k, v in class_to_idx.items()}  # reverse mapping

# -------------------------
# Image Preprocessing
# Must match exactly what we used during training
# -------------------------
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -------------------------
# Health Check Route
# A simple endpoint to confirm the server is running
# Visit http://localhost:5000/ in your browser to test
# -------------------------
@app.route('/')
def home():
    return jsonify({
        'status': 'iSpyPlants API is running',
        'usage': 'POST an image to /predict to get a plant prediction'
    })

# -------------------------
# Prediction Route
# Accepts a POST request with an image file
# Returns the top 5 predictions as JSON
# -------------------------
@app.route('/predict', methods=['POST'])
def predict():
    # check that an image was actually sent
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']          # get the image from the request
    image_bytes = file.read()              # read the raw bytes
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')  # open as PIL image

    # preprocess and run through model
    image_tensor = transform(image).unsqueeze(0).to(device)  # add batch dimension

    with torch.no_grad():  # no gradients needed for inference
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top_probs, top_indices = probabilities.topk(5)  # get top 5 predictions

    # build results list
    results = []
    for prob, idx in zip(top_probs[0], top_indices[0]):
        class_num = idx_to_class[idx.item()]
        flower_name = cat_to_name[str(class_num)]
        results.append({
            'plant': flower_name.title(),
            'confidence': round(prob.item() * 100, 2)  # round to 2 decimal places
        })

    return jsonify({
        'predictions': results
    })

# -------------------------
# Run the app
# debug=True auto-reloads when you make code changes
# -------------------------
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)