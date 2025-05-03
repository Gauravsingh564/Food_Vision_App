import os
import sys
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

# ─── 0. Ensure imports from Script folder ─────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT_PATH = os.path.join(HERE, "models")
if SCRIPT_PATH not in sys.path:
    sys.path.append(SCRIPT_PATH)

# ─── 1. Import the EfficientNet-B0 transfer learning builder ─────────────
from Effnet_B0_Model_Builder import create_transfer_model

# ─── 2. Constants ─────────────────────────────────────────────────────────
MODEL_FILE = "05_going_modular_cell_model.pth"
# Try common locations for the model file
possible_paths = [
    os.path.join(HERE, "models", MODEL_FILE),
    os.path.join(HERE, MODEL_FILE),
    os.path.join(HERE, "Script", MODEL_FILE)
]
# Find the first existing path
MODEL_PATH = next((p for p in possible_paths if os.path.exists(p)), None)

IMG_SIZE    = (224, 224)
CLASS_NAMES = ["pizza", "steak", "sushi"]  # adjust as needed

# ─── 3. Load & cache the model ─────────────────────────────────────────────
@st.cache_resource
def load_model(device):
    if MODEL_PATH is None:
        st.error(f"Model file '{MODEL_FILE}' not found. Please place it in the 'models/' folder or project root.")
        return None
    # instantiate transfer model
    model = create_transfer_model(
        num_classes=len(CLASS_NAMES),
        pretrained=False,
        freeze_base=True,
        dropout=0.2
    )
    # load weights
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model

# ─── 4. Image → prediction helper ─────────────────────────────────────────
def predict_image(img: Image.Image, model: torch.nn.Module, device):
    preprocess = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor()
    ])
    x = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs  = torch.softmax(logits, dim=1)[0]
        idx    = torch.argmax(probs).item()
    return CLASS_NAMES[idx], probs[idx].item()

# ─── 5. Streamlit UI ───────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Food Vision V1.0", layout="wide")
    st.title("🍽️ Food Vision App")
    st.write("Upload an image and I'll classify it!")

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

    with st.spinner("Analyzing…"):
        label, confidence = predict_image(img, model, device)

    st.success(f"**Prediction:** {label}")
    st.write(f"**Confidence:** {confidence:.2f}")

if __name__ == "__main__":
    main()
