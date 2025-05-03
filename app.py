# app.py

import os, sys
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

# 0. Allow import from Script/
HERE = os.path.dirname(__file__)
SCRIPT = os.path.join(HERE, "Script")
if SCRIPT not in sys.path:
    sys.path.append(SCRIPT)

from model_builder import TinyVGG

# 1. Constants
MODEL_FILE   = "05_going_modular_cell_model.pth"
MODEL_PATH   = os.path.join(HERE, "models", MODEL_FILE)
IMG_SIZE     = (64, 64)                         # match training
NORMALIZE_MEAN = [0.485, 0.456, 0.406]          # match training
NORMALIZE_STD  = [0.229, 0.224, 0.225]
CLASS_NAMES  = ["pizza", "steak", "sushi"]

# 2. Load & cache
@st.cache_resource
def load_model(device):
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found at `{MODEL_PATH}`")
        return None
    model = TinyVGG(input_shape=3, hidden_units=10, output_shape=len(CLASS_NAMES))
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model

# 3. Prediction helper
def predict_image(img: Image.Image, model, device):
    preprocess = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
    ])
    x = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs  = torch.softmax(logits, dim=1)[0]
        idx    = torch.argmax(probs).item()
    return CLASS_NAMES[idx], probs[idx].item()

# 4. Streamlit UI
def main():
    st.set_page_config(page_title="Food Vision TinyVGG", layout="wide")
    st.title("🍽️ Food Vision App")
    st.write("Upload an image of pizza, steak, or sushi.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(device)
    if model is None:
        return

    uploaded = st.file_uploader("", type=["jpg", "jpeg", "png"])
    if not uploaded:
        st.info("Please upload an image.")
        return

    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Your upload", use_column_width=True)

    with st.spinner("Predicting…"):
        label, conf = predict_image(img, model, device)

    st.success(f"Prediction: **{label}** ({conf:.2f})")

if __name__ == "__main__":
    main()
