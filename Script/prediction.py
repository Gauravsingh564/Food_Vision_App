import os
import sys
import argparse
import torch
from PIL import Image
from torchvision import transforms

# Ensure we can import TinyVGG from model_builder
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.append(HERE)

from model_builder import TinyVGG

# ─── Constants ─────────────────────────────────────────────────────────────
MODEL_FILE  = "05_going_modular_script_mode_tinyvgg_model.pth"
MODEL_PATH  = os.path.abspath(os.path.join(HERE, "..", "models", MODEL_FILE))
IMG_SIZE    = (64, 64)  # match training size
MEAN        = [0.485, 0.456, 0.406]  # training normalization mean
STD         = [0.229, 0.224, 0.225]  # training normalization std
CLASS_NAMES = ["pizza", "steak", "sushi"]

# ─── Load model utility ────────────────────────────────────────────────────
def load_model(device):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {MODEL_PATH}")
    # Instantiate TinyVGG with hidden_units matching your checkpoint (10)
    model = TinyVGG(
        input_shape=3,
        hidden_units=10,
        output_shape=len(CLASS_NAMES)
    )
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model

# ─── Single image prediction ───────────────────────────────────────────────
def predict_image(img_path: str, model, device):
    img = Image.open(img_path).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])
    x = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs  = torch.softmax(logits, dim=1)[0]
        idx    = torch.argmax(probs).item()
    return CLASS_NAMES[idx], probs[idx].item()

# ─── CLI entrypoint ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Run TinyVGG predictions on images")
    parser.add_argument("images", nargs="+", help="Path(s) to image file(s)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(device)

    for img_path in args.images:
        if not os.path.isfile(img_path):
            print(f"[SKIP] File not found: {img_path}")
            continue
        label, conf = predict_image(img_path, model, device)
        print(f"{os.path.basename(img_path)} → {label} ({conf:.2f})")

if __name__ == "__main__":
    main()
