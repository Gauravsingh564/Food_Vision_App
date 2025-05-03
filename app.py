# app.py

import os
import sys
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

# ─── 0. Allow imports from your Script folder ───────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT_PATH = os.path.join(HERE, "Script")
if SCRIPT_PATH not in sys.path:
    sys.path.append(SCRIPT_PATH)

# ─── 1. Import your TinyVGG definition ────────────────────────────────────
from model_builder import TinyVGG

# ─── 2. Constants ─────────────────────────────────────────────────────────
MODEL_FILE = "05_going_modular_cell_model.pth"
MODEL_PATH = os.path.join(HERE, "models", MODEL_FILE)

IMG_SIZE    = (224, 224)
CLASS_NAMES = ["pizza", "steak", "sushi"]   # ← adjust to your real labels

# ─── 3. Load & cache the model ─────────────────────────────────────────────
@st.cache_resource
def load_model(device):
    if not os.path.exists(MODEL_PATH):
        st.error(
            f"❌ Model file not found:\n  {MODEL_PATH}\n"
            "Please place your checkpoint in the `models/` folder."
        )
        return None

    # instantiate TinyVGG with the same hyperparameters you trained with
    model = TinyVGG(
        input_shape=3,
        hidden_units=10,
        output_shape=len(CLASS_NAMES)
    )
    # load your trained weights
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model

# ─── 4. Image → prediction helper ─────────────────────────────────────────
def predict_image(img: Image.Image, model: torch.nn.Module, device):
    preprocess = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
    ])
    x = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs  = torch.softmax(logits, dim=1)[0]
        idx    = torch.argmax(probs).item()
    return CLASS_NAMES[idx], probs[idx].item()

# ─── 5. Streamlit UI ───────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Food Vision TinyVGG", layout="wide")
    st.title("🍽️ Food Vision App (TinyVGG)")
    st.write("Upload an image and let's see what it is!")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(device)
    if model is None:
        st.stop()

    uploaded = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        st.info("Please upload an image to get started.")
        return

    img = Image.open(uploaded).convert("RGB")
    st.image(img, use_column_width=True, caption="Your upload")

    with st.spinner("Predicting…"):
        label, confidence = predict_image(img, model, device)

    st.success(f"**Prediction:** {label}")
    st.write(f"**Confidence:** {confidence:.2f}")

if __name__ == "__main__":
    main()
