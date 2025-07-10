import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import gdown

# === Settings ===
MODEL_FILE = "metal_fracture_classifier_efficientnet.pt"
GDRIVE_ID = "1PzbRYmktxwCRoff9kr6_cj28wBqPjzHq"  # 🔁 Ganti dengan ID anda
GDRIVE_URL = f"https://drive.google.com/file/d/1PzbRYmktxwCRoff9kr6_cj28wBqPjzHq/view?usp=drive_link"

# === Manual label names (update if needed) ===
label_names = ["aluminum", "steel", "titanium"]  # Contoh. Ubah ikut class sebenar

# === Download model if not exists ===
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_FILE):
        with st.spinner("📥 Downloading model..."):
            gdown.download(GDRIVE_URL, MODEL_FILE, quiet=False)

    model = models.efficientnet_b0(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(model.classifier[1].in_features, len(label_names))
    )
    model.load_state_dict(torch.load(MODEL_FILE, map_location="cpu"))
    model.eval()
    return model

model = load_model()

# === Transform ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# === UI ===
st.title("🔩 Metal Fracture Type Classifier (EfficientNet)")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        confidence, predicted = torch.max(probabilities, 0)

    predicted_label = label_names[predicted.item()]
    st.success(f"✅ Predicted: **{predicted_label}**\n📊 Confidence: **{confidence.item()*100:.2f}%**")

    st.subheader("🔝 Top 3 Predictions")
    top3_prob, top3_idx = torch.topk(probabilities, 3)
    for i in range(3):
        st.write(f"{label_names[top3_idx[i]]}: {top3_prob[i].item() * 100:.2f}%")
