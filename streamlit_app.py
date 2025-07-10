# -*- coding: utf-8 -*-
"""
Created on Sat Jun 28 00:15:10 2025
@author: zzulk
"""

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# Load label names (from training folders)
label_names = os.listdir(r"C:\Users\zzulk\Downloads\Metal_Type_Fracture_Split - Copy\train")
label_names.sort()

# Image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
])

# Load trained model
@st.cache_resource
def load_model():
    model = models.efficientnet_b0(weights=None)
    # 🔧 Match the classifier architecture used during training
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(model.classifier[1].in_features, len(label_names))
    )
    model.load_state_dict(torch.load(
        r"C:\Users\zzulk\Downloads\Metal_Type_Fracture_Split - Copy\metal_fracture_classifier_efficientnet.pt", 
        map_location='cpu'))
    model.eval()
    return model

model = load_model()

# Streamlit UI
st.title("🔍 Metal Fracture Type Classifier")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess & predict
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        confidence, predicted = torch.max(probabilities, 0)

    predicted_label = label_names[predicted.item()]
    st.success(f"✅ Predicted Class: {predicted_label}  \n📊 Confidence: {confidence.item() * 100:.2f}%")

    # Show top 3 predictions
    st.subheader("🔝 Top 3 Predictions")
    top3_prob, top3_idx = torch.topk(probabilities, 3)
    for i in range(3):
        st.write(f"{label_names[top3_idx[i]]}: {top3_prob[i].item() * 100:.2f}%")
