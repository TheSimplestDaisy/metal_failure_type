import os
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "metal_fracture_classifier.pt")

class_labels = [
    "brittle_fracture",
    "crack_fracture",
    "ductile_dimple_fracture",
    "fatigue_line_pattern",
    "intergranular_brittle_fracture",
    "river_pattern"
]

st.title("Metal Fracture Type Classifier")

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found: metal_fracture_classifier.pt")
    st.stop()

model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(class_labels))

try:
    checkpoint = torch.load(MODEL_PATH, map_location=torch.device("cpu"))

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]

    new_state_dict = {}
    for key, value in checkpoint.items():
        new_key = key.replace("module.", "").replace("model.", "")
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict)
    model.eval()

except Exception as e:
    st.error("Model loading error.")
    st.code(str(e))
    st.stop()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

uploaded_file = st.file_uploader(
    "Upload Fracture Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        probabilities = F.softmax(output, dim=1)[0]
        confidence, predicted_class = torch.max(probabilities, 0)

    predicted_label = class_labels[predicted_class.item()]
    confidence_percent = confidence.item() * 100

    st.success(f"Prediction: **{predicted_label}** ({confidence_percent:.2f}%)")
