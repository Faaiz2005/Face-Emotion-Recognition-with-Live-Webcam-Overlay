"""
============================================================
Face Emotion Recognition - Images + Real-Time Webcam
Project: Lab 14 - Face Emotion Recognition
Roll No.: 23-AI-40
============================================================
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
try:
    import tensorflow as tf
    print(f"TensorFlow {tf.__version__} loaded.")
except ImportError:
    print("ERROR: TensorFlow not installed")
    exit(1)
MODEL_PATH = "fixed_model.keras"
IMG_SIZE = 48
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
EMOTION_COLORS = {
    'angry': (0, 0, 255),
    'disgust': (0, 140, 0),
    'fear': (180, 0, 180),
    'happy': (0, 220, 0),
    'neutral': (180, 180, 180),
    'sad': (255, 100, 0),
    'surprise': (0, 200, 200),
}

def load_model(path):
    if not os.path.exists(path):
        print("Model not found:", path)
        exit(1)
    print("Loading model...")
    model = tf.keras.models.load_model(path, compile=False)
    print("Model loaded:", model.input_shape)
    return model

def preprocess(face):
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=-1)
    face = np.expand_dims(face, axis=0)
    return face

def process_image(model, face_cascade, path):
    frame = cv2.imread(str(path))
    if frame is None:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    best = None
    best_score = -1
    for (x, y, w, h) in faces:
        pad = int(w * 0.2)
        x1, y1 = max(0, x-pad), max(0, y-pad)
        x2, y2 = min(frame.shape[1], x+w+pad), min(frame.shape[0], y+h+pad)
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            continue
        preds = model.predict(preprocess(face_crop), verbose=0)[0]
        idx = np.argmax(preds)
        emotion = EMOTION_LABELS[idx]
        conf = float(preds[idx])
        score = conf * (w * h)
        if score > best_score:
            best_score = score
            best = (x, y, w, h, emotion, conf)
    if best is None:
        return frame
    x, y, w, h, emotion, conf = best
    color = EMOTION_COLORS[emotion]
    cv2.rectangle(frame, (x, y), (x+w, y+h), color,3)
    label = f"{emotion.upper()}  {conf*100:.1f}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.9
    thickness = 4
    (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
    cv2.rectangle(frame, (x, y-th-30), (x+tw+25, y), (0,0,0), -1)
    cv2.putText(frame, label, (x+10, y-10),
                font, scale, (255,255,255), thickness)
    bar_h = 18
    cv2.rectangle(frame, (x, y+h+10), (x+w, y+h+10+bar_h), (40,40,40), -1)
    cv2.rectangle(frame, (x, y+h+10),
                  (x+int(w*conf), y+h+10+bar_h), color, -1)
    cv2.rectangle(frame, (x, y+h+10), (x+w, y+h+10+bar_h), (255,255,255), 2)
    return frame

def run_webcam(model, face_cascade):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam not found")
        return
    print("Webcam started (press Q to quit)")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        best = None
        best_score = -1
        for (x, y, w, h) in faces:
            pad = int(w * 0.2)
            x1, y1 = max(0, x-pad), max(0, y-pad)
            x2, y2 = min(frame.shape[1], x+w+pad), min(frame.shape[0], y+h+pad)
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue
            preds = model.predict(preprocess(face_crop), verbose=0)[0]
            idx = np.argmax(preds)
            emotion = EMOTION_LABELS[idx]
            conf = float(preds[idx])
            score = conf * (w * h)
            if score > best_score:
                best_score = score
                best = (x, y, w, h, emotion, conf)
        if best:
            x, y, w, h, emotion, conf = best
            color = EMOTION_COLORS[emotion]
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
            label = f"{emotion.upper()} {conf*100:.1f}%"
            cv2.rectangle(frame, (x, y-40), (x+280, y), (0,0,0), -1)
            cv2.putText(frame, label, (x+10, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (255,255,255), 3)
            cv2.rectangle(frame, (x, y+h+10), (x+w, y+h+25),
                          (40,40,40), -1)
            cv2.rectangle(frame, (x, y+h+10),
                          (x+int(w*conf), y+h+25),
                          color, -1)
        cv2.imshow("FER - Real Time", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--images", nargs="+")
    parser.add_argument("--folder")
    parser.add_argument("--webcam", action="store_true")
    args = parser.parse_args()
    model = load_model(args.model)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if args.webcam:
        run_webcam(model, face_cascade)
        return
    image_paths = []
    if args.images:
        image_paths.extend(args.images)
    if args.folder:
        folder = Path(args.folder)
        exts = [".jpg", ".png", ".jpeg"]
        image_paths.extend([str(p) for p in folder.iterdir() if p.suffix in exts])
    if not image_paths:
        print("No images provided")
        return
    for path in image_paths:
        img = process_image(model, face_cascade, path)
        if img is not None:
            cv2.imshow(str(path), img)
            cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()