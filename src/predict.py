import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_trained_model(num_classes):
    # Load ResNet18 dengan weights default
    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    # Update fully-connected layer agar sesuai jumlah kelas
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Load weight hasil training
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH = os.path.join(BASE_DIR, "models", "face_stress_model.pth")

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def predict(img_path):
    # Cek file gambar
    img = cv2.imread(img_path)
    if img is None:
        print("Gambar tidak ditemukan atau rusak:", img_path)
        return

    # BGR → RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert ke PIL
    img_pil = Image.fromarray(img_rgb)

    # Transform sama seperti training
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    img_tensor = tf(img_pil).unsqueeze(0).to(DEVICE)

    # Jumlah kelas → ambil dari folder data
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data", "train")
    classes = sorted(os.listdir(DATA_DIR))

    # Load model
    model = load_trained_model(len(classes))

    # Predict
    with torch.no_grad():
        output = model(img_tensor)
        _, pred = torch.max(output, 1)

    print("Kelas Prediksi:", classes[pred.item()])


if __name__ == "__main__":
    predict("data/val/low/PrivateTest_95094.jpg")
