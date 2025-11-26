import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==== Load Model ====
def load_trained_model(num_classes):
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH = os.path.join(BASE_DIR, "models", "face_stress_model.pth")

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


# ==== Setup ====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_TRAIN = os.path.join(BASE_DIR, "data", "train")
classes = sorted(os.listdir(DATA_TRAIN))

model = load_trained_model(len(classes))

# Haarcascade untuk deteksi wajah
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_detector = cv2.CascadeClassifier(CASCADE_PATH)

# Transform yang sama seperti training
tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ==== Kamera Live ====
cap = cv2.VideoCapture(0)

print("Tekan 'q' untuk keluar...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal membuka kamera.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        # Convert to PIL
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_rgb)

        # Transform
        tensor = tf(face_pil).unsqueeze(0).to(DEVICE)

        # Predict
        with torch.no_grad():
            output = model(tensor)
            _, pred = torch.max(output, 1)
            label = classes[pred.item()]

        # Tampilkan hasil
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, f"Stress: {label}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow("Stress Detection - Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
