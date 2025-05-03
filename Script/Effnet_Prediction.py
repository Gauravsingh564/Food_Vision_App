import argparse
import sys
import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from Effnet_B0_Model_Builder import create_transfer_model

THRESHOLD_PCT = 39.9  # 40%

def predict_image(model, device, image_path):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    try:
        img = Image.open(image_path).convert('RGB')
    except (FileNotFoundError, UnidentifiedImageError):
        print(f"Error: Cannot open image at '{image_path}'.")
        sys.exit(1)

    x = preprocess(img).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1)
        conf, pred = torch.max(probs, 1)
    # return index and raw probability (0–1)
    return pred.item(), conf.item()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Predict with EfficientNet-B0 classifier')
    parser.add_argument('--model-path',  required=True,
                        help='Path to .pth weights')
    parser.add_argument('--image-path',  required=True,
                        help='Path to input image')
    parser.add_argument('--num-classes', type=int, required=True)
    parser.add_argument('--class-names', nargs='+',
                        help='List of class names (optional)')
    parser.add_argument('--freeze-base', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_transfer_model(
        num_classes=args.num_classes,
        pretrained=False,
        freeze_base=args.freeze_base
    )
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)

    idx, confidence = predict_image(model, device, args.image_path)
    confidence_pct = confidence * 100

    # DEBUG: show raw confidence (fraction) and percent, plus threshold
    print(f"[DEBUG] confidence(fraction)={confidence:.4f}, "
          f"confidence_pct={confidence_pct:.2f}%, "
          f"threshold_pct={THRESHOLD_PCT:.2f}%")

    # strictly below 40%
    if confidence_pct < THRESHOLD_PCT:
        print("Low confidence. Please upload the right image.")
        sys.exit(1)

    # map to name if provided
    if args.class_names:
        if len(args.class_names) != args.num_classes:
            print("Error: class-names length mismatch.")
            sys.exit(1)
        label = args.class_names[idx]
    else:
        label = str(idx)

    print(f"Predicted: {label} (Confidence: {confidence_pct:.2f}%)")
