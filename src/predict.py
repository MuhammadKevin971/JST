import cv2
import torch
from torchvision import transforms, models
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

classes = ["low", "medium", "high"]

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict(image_path):
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(torch.load("models/face_stress_model.pth", map_location=DEVICE))
    model.eval()

    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_tensor = tf(img_rgb).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        output = model(img_tensor)
        _, pred = output.max(1)

    print("Predicted Stress Level:", classes[pred.item()])

predict("test.jpg")
