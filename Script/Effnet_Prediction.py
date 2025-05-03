import argparse
import torch
from PIL import Image
from torchvision import transforms
from Effnet_B0_Model_Builder import create_transfer_model


def predict_image(model, device, image_path, num_classes, class_names=None):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    tensor = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
        class_idx = pred.item()
        confidence = conf.item()

    if class_names:
        label = class_names[class_idx]
        return label, confidence
    return class_idx, confidence


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict with EfficientNet-B0 classifier')
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained model weights .pth')
    parser.add_argument('--image-path', type=str, required=True, help='Path to input image file')
    parser.add_argument('--num-classes', type=int, required=True, help='Number of target classes')
    parser.add_argument('--class-names', type=str, nargs='+', help='Optional: list of class names')
    parser.add_argument('--freeze-base', action='store_true', help='Match freeze setting used during training')
    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model and load weights
    model = create_transfer_model(num_classes=args.num_classes,
                                  pretrained=False,
                                  freeze_base=args.freeze_base)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)

    # Predict
    label, confidence = predict_image(model, device, args.image_path,
                                      args.num_classes, args.class_names)
    print(f"Predicted: {label} (Confidence: {confidence*100:.2f}%)")