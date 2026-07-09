import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import streamlit as st
import os

# Directory where this script is running
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "metal_fracture_classifier.pt")

# Class labels
class_labels = [
    'brittle_fracture',
    'crack_fracture',
    'ductile_dimple_fracture',
    'fatigue_line_pattern',
    'intergranular_brittle_fracture',
    'river_pattern'
]

# Check model file
if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Please make sure 'metal_fracture_classifier.pt' is in the same folder as this script.")
    st.stop()

# Load model
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, len(class_labels))

model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Streamlit UI
st.title("Metal Fracture Type Classifier")

uploaded_file = st.file_uploader(
    "Upload Fracture Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Preprocess image
    img_tensor = transform(image).unsqueeze(0)

    # Prediction
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = F.softmax(output, dim=1)[0]
        confidence, predicted_class = torch.max(probabilities, 0)

    predicted_label = class_labels[predicted_class.item()]
    confidence_percent = confidence.item() * 100

    st.success(
        f"Prediction: **{predicted_label}** ({confidence_percent:.2f}%)"
    )
```
