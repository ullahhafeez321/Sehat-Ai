from flask import Flask, render_template, request
from PIL import Image
import torch
from torchvision import transforms
import os

# --------------------
# Flask Setup
# --------------------
app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --------------------
# Device
# --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------
# Load Chest X-ray Model
# --------------------
from transformers import ViTForImageClassification

class ViTForChestXray(torch.nn.Module):
    def __init__(self, num_labels=14):  # default NIH labels
        super().__init__()
        self.vit = ViTForImageClassification.from_pretrained(
            'google/vit-base-patch16-224',
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )
        hidden_size = self.vit.config.hidden_size
        self.vit.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_size, num_labels)
        )

    def forward(self, pixel_values):
        return self.vit(pixel_values=pixel_values).logits

# Replace with actual number of NIH labels
NIH_LABELS = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", 
              "Mass", "Nodule", "Pneumonia", "Pneumothorax", 
              "Consolidation", "Edema", "Emphysema", "Fibrosis", 
              "Pleural_Thickening", "Hernia"]

model = ViTForChestXray(num_labels=len(NIH_LABELS))
model.load_state_dict(torch.load("models/vit_chest_xray_epoch_3_auc_0.820.pt", map_location=device))
model.to(device)
model.eval()

# --------------------
# Image Preprocessing
# --------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# --------------------
# Routes
# --------------------
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/chest_xray", methods=["GET", "POST"])
def chest_xray():
    prediction_text = ""
    confidence_text = ""
    uploaded_image_path = None

    if request.method == "POST":
        file = request.files["xray_image"]
        if file:
            uploaded_image_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(uploaded_image_path)

            # Load and preprocess image
            img = Image.open(uploaded_image_path).convert("RGB")
            input_tensor = transform(img).unsqueeze(0).to(device)

            # Model prediction
            with torch.no_grad():
                outputs = torch.sigmoid(model(input_tensor)).cpu().numpy()[0]

            # Labels above threshold
            labels = [NIH_LABELS[i] for i, val in enumerate(outputs) if val > 0.5]
            if not labels:
                labels = ["No Finding"]

            # Confidence for labels > 0.3
            confs = {NIH_LABELS[i]: float(outputs[i]) for i in range(len(NIH_LABELS)) if outputs[i] > 0.3}

            prediction_text = ", ".join(labels)
            confidence_text = confs

    return render_template("chest_xray.html",
                           prediction=prediction_text,
                           confidence=confidence_text,
                           image_path=uploaded_image_path)

# --------------------
# Run Flask
# --------------------
if __name__ == "__main__":
    app.run(debug=True)
