# 🌿 iSpyPlants

An AI-powered plant and flower identification app built with PyTorch and transfer learning.

## About
iSpyPlants uses a deep learning model to identify plant and flower species from photos. 
Point it at a flower and it will tell you what it is along with a confidence score.

This project was built as an extension of Udacity's AI Programming with Python nanodegree, 
with the goal of eventually launching as a mobile app.

## Results
- **V1 Model:** 95.12% accuracy across 102 flower species (Oxford 102 Flowers dataset)
- **V2 Model:** In progress — retraining on PlantNet dataset for broad plant identification

## How It Works
1. A pretrained ResNet50 model (trained on ImageNet) is used as a base
2. The final layer is replaced with a custom classifier for plant species
3. Only the final layer is trained (transfer learning) — this keeps training fast and accurate
4. The model is trained on Google Colab using a free T4 GPU

## Example Output
```
🌸 Top 5 Predictions:
------------------------------
Sunflower: 99.86%
Tree Poppy: 0.07%
Japanese Anemone: 0.03%
Colt'S Foot: 0.02%
Purple Coneflower: 0.01%
```

## Project Structure
```
iSpyPlants/
├── scripts/
│   ├── preview_data.py      # Verify dataset loaded correctly
│   ├── prepare_data.py      # Organize images into train/valid folders
│   ├── train.py             # Train the model using transfer learning
│   └── predict.py           # Predict plant species from an image
├── model/                   # Saved model weights (not tracked in git)
├── data/                    # Dataset images (not tracked in git)
├── cat_to_name.json         # Maps class numbers to flower names
└── test_*.jpg               # Sample images for testing
```

## How To Run

**1. Clone the repo:**
```bash
git clone https://github.com/stargazingpnw/iSpyPlants.git
cd iSpyPlants
```

**2. Install dependencies:**
```bash
pip install torch torchvision pillow numpy matplotlib scipy scikit-learn
```

**3. Run a prediction:**
```bash
python scripts/predict.py --image your_flower.jpg
```

## Tech Stack
- **PyTorch** — model training and inference
- **ResNet50** — pretrained base model (transfer learning)
- **Google Colab** — free GPU training
- **Python** — scripting and data preparation

## Roadmap
- [x] V1: Train flower classifier (95.12% accuracy)
- [ ] V2: Retrain on PlantNet for broad plant identification
- [x] Build Flask API to serve predictions
- [ ] Deploy to cloud
- [ ] Build Flutter mobile app
- [ ] Launch on App Store and Google Play

## Author
Built by [@stargazingpnw](https://github.com/stargazingpnw)