# app.py

import os
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

# ─── 0. Make sure we can import your code ─────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT_PATH = os.path.join(HERE, "Script")
if SCRIPT_PATH not in os.sys.path:
    os.sys.path.append(SCRIPT_PATH)

from model_builder import TinyVGG  # your model architecture
# (we don't need engine, train, utils here)
# ─── 1. Constants ─────────────────────────────────────────────────────────
MODEL_FILE  = "05_going_modular_cell_model.pth"
MODEL_PATH  = os.path.join(HERE, "models", MODEL_FILE)
IMG_SIZE    = (224, 224)
CLASS_NAMES = ["pizza", "steak", "sushi"]

# ─── 2. Load & cache the model ─────────────────────────────────────────────
@st.cache_resource
def load_model(device):
    # instantiate same architecture you trained
    model = TinyVGG(input_shape=3,
                    hidden_units=10,
                    output_shape=len(CLASS_NAMES))
    # load weights
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model

# ─── 3. Image → prediction helper ─────────────────────────────────────────
def predict_image(img: Image.Image, model: torch.nn.Module, device):
    # same transforms you used for testing
    preprocess = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor()
    ])
    x = preprocess(img).unsqueeze(0).to(device)            # add batch dim
    with torch.no_grad():
        logits = model(x)                                  # raw outputs
        probs  = torch.softmax(logits, dim=1)[0]           # to probabilities
        idx    = torch.argmax(probs).item()                # predicted class index
    return CLASS_NAMES[idx], probs[idx].item()

# ─── 4. Streamlit UI ───────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Food Vision 🍕🥩🍣", layout="wide")
    st.title("🍽️ Food Vision App")
    st.write("Upload an image of pizza, steak, or sushi and I'll tell you which it is!")

    # 1️⃣ load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(device)

    # 2️⃣ uploader
    uploaded = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        st.info("Please upload a food image to get started.")
        return

    # 3️⃣ display
    img = Image.open(uploaded).convert("RGB")
    st.image(img, use_column_width=True, caption="Your upload")

    # 4️⃣ predict
    with st.spinner("Analyzing…"):
        label, confidence = predict_image(img, model, device)

    st.success(f"**Prediction:** {label}")
    st.write(f"**Confidence:** {confidence:.2f}")

if __name__ == "__main__":
    main()
