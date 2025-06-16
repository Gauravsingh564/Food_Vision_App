import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from Effnet_B0_Model_Builder import create_transfer_model
def train(model, device, train_loader, val_loader, criterion, optimizer, epochs, save_path):
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
        val_acc = correct / len(val_loader.dataset)
        print(f"Epoch {epoch}/{epochs} - Loss: {epoch_loss:.4f} - Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
    print(f"Training complete. Best Val Acc: {best_acc:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train EfficientNet-B0 classifier')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to dataset root with train/val subfolders')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training and validation')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--num-classes', type=int, required=True, help='Number of target classes')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate in classifier head')
    parser.add_argument('--freeze-base', action='store_true', help='Freeze EfficientNet feature layers')
    parser.add_argument('--output-model-path', type=str, default='effnet_b0_best.pth', help='File to save best model weights')
    args = parser.parse_args()

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Datasets and loaders
    train_dataset = datasets.ImageFolder(os.path.join(args.data_dir, 'train'), transform=transform)
    val_dataset = datasets.ImageFolder(os.path.join(args.data_dir, 'val'), transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Model, criterion, optimizer
    model = create_transfer_model(num_classes=args.num_classes,
                                  pretrained=True,
                                  freeze_base=args.freeze_base,
                                  dropout=args.dropout)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    # Train
    train(model, device, train_loader, val_loader, criterion, optimizer,
          args.epochs, args.output_model_path)
