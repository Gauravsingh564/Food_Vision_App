import os
import sys
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

# ──────────────────────────────────────────────────────────────────────────────
# 0️⃣ Imports & Path Setup
# ──────────────────────────────────────────────────────────────────────────────
HERE   = os.path.dirname(__file__)
SCRIPT = os.path.join(HERE, "Script")
if SCRIPT not in sys.path:
    sys.path.append(SCRIPT)

# swap out TinyVGG for your EfficientNet builder
from Effnet_B0_Model_Builder import create_transfer_model

# ──────────────────────────────────────────────────────────────────────────────
# 1️⃣ Config
# ──────────────────────────────────────────────────────────────────────────────
MODEL_FILE     = "effnet_b0_best.pth"
MODEL_PATH     = os.path.join(HERE, "models", MODEL_FILE)
IMG_SIZE       = (224, 224)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD  = [0.229, 0.224, 0.225]
CLASS_NAMES    = ["pizza", "steak", "sushi"]

# ──────────────────────────────────────────────────────────────────────────────
# 2️⃣ Load model
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(device):
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found at {MODEL_PATH}")
        return None
    model = create_transfer_model(
        num_classes=len(CLASS_NAMES),
        pretrained=False,
        freeze_base=False,
        dropout=0.2
    )
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model

# ──────────────────────────────────────────────────────────────────────────────
# 3️⃣ Inference
# ──────────────────────────────────────────────────────────────────────────────
def preprocess_image(img: Image.Image):
    return transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
    ])(img).unsqueeze(0)

# ──────────────────────────────────────────────────────────────────────────────
# 4️⃣ UI
# ──────────────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Food Vision EfficientNet-B0", layout="wide")
    st.title("🍽️ Food Vision with EfficientNet-B0")
    st.write("Upload a pizza, steak, or sushi image to classify.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(device)
    if model is None:
        return

    uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if not uploaded:
        return

    img = Image.open(uploaded).convert("RGB")
    st.image(img, use_column_width=True, caption="Your upload")

    with st.spinner("Predicting…"):
        x = preprocess_image(img).to(device)
        with torch.no_grad():
            outputs = model(x)
            probs   = torch.softmax(outputs, dim=1)[0].cpu().numpy()
        idx  = int(probs.argmax())
        conf = float(probs[idx])
        label = CLASS_NAMES[idx]
    
    st.success(f"Prediction: **{label}** ({conf*100:.2f}%)")
    # Show bar chart of probabilities
    st.bar_chart({CLASS_NAMES[i]: probs[i] for i in range(len(CLASS_NAMES))})

if __name__ == "__main__":
    main()
