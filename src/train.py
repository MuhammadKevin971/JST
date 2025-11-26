import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, ResNet18_Weights
from dataset import get_dataloaders

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_model():

    # === SAFE PATH HANDLING (menghindari error FileNotFound) ===
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")

    print("Using DATA DIR:", DATA_DIR)

    train_loader, val_loader, classes = get_dataloaders(DATA_DIR)

    # === RESNET API BARU (tidak deprecated lagi) ===
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)

    # Ganti layer FC untuk jumlah kelas kamu
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    EPOCHS = 60
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss = {total_loss:.4f}")

        # === Validation ===
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                _, preds = outputs.max(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        acc = correct / total if total > 0 else 0
        print(f"Validation Accuracy = {acc:.4f}")

    # === Save model ===
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    os.makedirs(MODEL_DIR, exist_ok=True)

    SAVE_PATH = os.path.join(MODEL_DIR, "face_stress_model.pth")
    torch.save(model.state_dict(), SAVE_PATH)

    print(f"Model saved at: {SAVE_PATH}")


if __name__ == "__main__":
    train_model()
