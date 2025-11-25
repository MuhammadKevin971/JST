import cv2
import os

def extract_frames(video_path, output_dir, fps=10):
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(video_fps / fps)

    count = 0
    saved = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if count % interval == 0:
            cv2.imwrite(f"{output_dir}/frame_{saved:05d}.jpg", frame)
            saved += 1
        
        count += 1

    cap.release()
    print(f"Frames saved: {saved}")
