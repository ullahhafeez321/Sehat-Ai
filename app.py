from flask import Flask, render_template, request
from PIL import Image
import torch
from torchvision import transforms
import os
import numpy as np
from transformers import ViTForImageClassification

# --------------------------
# Flask Setup
# --------------------------
app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --------------------------
# Device
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --------------------------
# Labels (exactly from nih.NIHChestXRay.LABELS)
# --------------------------
NIH_LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia", "No Finding"
]

# --------------------------
# Model Definition (same as Kaggle notebook)
# --------------------------
class ViTForChestXray(torch.nn.Module):
    def __init__(self, num_labels=len(NIH_LABELS)):
        super().__init__()
        # Load pretrained ViT
        self.vit = ViTForImageClassification.from_pretrained(
            'google/vit-base-patch16-224',
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )

        # Replace classification head to match training
        hidden_size = self.vit.config.hidden_size
        self.vit.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_size, num_labels)
        )

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        return outputs.logits

# --------------------------
# Load Trained Model
# --------------------------
MODEL_PATH = "models/vit_chest_xray_epoch_3_auc_0.820.pt"
model = ViTForChestXray(num_labels=len(NIH_LABELS))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()
print("✅ Model loaded successfully and moved to device.")

# --------------------------
# Image Preprocessing (same as training transform)
# --------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# --------------------------
# Prediction Function
# --------------------------
def predict_xray(image_path):
    # Open and preprocess image
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        outputs = torch.sigmoid(logits).cpu().numpy()[0]

    # Binary threshold (same as training eval)
    binary_preds = (outputs > 0.5).astype(int)
    predicted_labels = [NIH_LABELS[i] for i, v in enumerate(binary_preds) if v == 1]

    # If no finding above threshold, default to "No Finding"
    if len(predicted_labels) == 0:
        predicted_labels = ["No Finding"]

    # Confidence scores ≥ 0.3 (as in notebook)
    conf_scores = {
        NIH_LABELS[i]: round(float(outputs[i]), 3)
        for i in range(len(NIH_LABELS))
        if outputs[i] >= 0.3
    }

    # Sort by confidence (descending)
    conf_scores = dict(sorted(conf_scores.items(), key=lambda x: x[1], reverse=True))

    return predicted_labels, conf_scores

# --------------------------
# Routes
# --------------------------
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/chest_xray", methods=["GET", "POST"])
def chest_xray():
    prediction_text = ""
    confidence_text = ""
    uploaded_image_path = None

    if request.method == "POST":
        file = request.files.get("xray_image")
        if file:
            uploaded_image_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(uploaded_image_path)

            # Run prediction
            preds, confs = predict_xray(uploaded_image_path)
            prediction_text = ", ".join(preds)
            confidence_text = confs

    return render_template(
        "chest_xray.html",
        prediction=prediction_text,
        confidence=confidence_text,
        image_path=uploaded_image_path
    )

# --------------------------
# Run Flask
# --------------------------
if __name__ == "__main__":
    app.run(debug=True)
