import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import gdown

# === Manual label names ===
label_names = ["aluminum", "steel", "titanium"]  # 🔁 EDIT ikut kelas sebenar anda

# === Google Drive download URL ===
MODEL_FILE = "metal_fracture_classifier_efficientnet.pt"
GDRIVE_URL = "https://drive.google.com/uc?id=1PzbRYmktxwCRoff9kr6_cj28wBqPjzHq"

@st.cache_resource
def load_model():
    # 1) Download model if belum ada
    if not os.path.exists(MODEL_FILE):
        st.info("📥 Downloading model...")
        gdown.download(GDRIVE_URL, MODEL_FILE, quiet=False)

    # 2) Siapkan arsitektur model
    model = models.efficientnet_b0(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(model.classifier[1].in_features, len(label_names))
    )

    # 3) Load berat model
    try:
        # cuba sebagai state_dict
        state = torch.load(MODEL_FILE, map_location="cpu")
        model.load_state_dict(state)
    except (RuntimeError, pickle.UnpicklingError):
        # kalau gagal, cuba load full-model
        model = torch.load(MODEL_FILE, map_location="cpu")
    model.eval()
    return model

model = load_model()

# === Preprocessing ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# === Streamlit UI ===
st.title("🔩 Metal Fracture Type Classifier")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # inference
    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        out = model(x)
        probs = torch.nn.functional.softmax(out[0], dim=0)
        conf, pred = torch.max(probs, 0)

    label = label_names[pred.item()]
    st.success(f"✅ Predicted: **{label}** ({conf.item()*100:.2f}%)")

    # Top-3
    st.subheader("🔝 Top 3 Predictions")
    top3_p, top3_i = torch.topk(probs, 3)
    for i in range(3):
        st.write(f"{label_names[top3_i[i]]}: {top3_p[i].item()*100:.2f}%")
