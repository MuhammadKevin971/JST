import cv2
import os

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def extract_faces(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(input_dir):
        path = os.path.join(input_dir, file)
        img = cv2.imread(path)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        i = 0
        for (x, y, w, h) in faces:
            face = img[y:y+h, x:x+w]
            cv2.imwrite(f"{output_dir}/{file[:-4]}_face_{i}.jpg", face)
            i += 1
