# model_load.py — ASL Sign Language Recognition with improved UI overlay

import argparse
import os
import time
import cv2
import numpy as np
import tensorflow as tf

# ---------- CLI ----------
parser = argparse.ArgumentParser(description="ASL Sign Language Recognition (webcam)")
parser.add_argument("--cam", type=int, default=0, help="Webcam index (0 default; try 1 if needed)")
parser.add_argument("--size", type=int, default=64, help="Input size expected by the model")
parser.add_argument("--mirror", action="store_true", help="Mirror preview horizontally")
args = parser.parse_args()

# ---------- Model ----------
MODEL_PATH = "SLR_final.h5"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"'{MODEL_PATH}' not found. Place it next to this script.")

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES = [
    "A","B","C","D","E","F","G","H","I","J","K","L","M",
    "N","O","P","Q","R","S","T","U","V","W","X","Y","Z",
    "SPACE","DELETE","NOTHING"
]

# ---------- Camera ----------
cap = cv2.VideoCapture(args.cam)
if not cap.isOpened():
    raise RuntimeError(
        f"Cannot open camera index {args.cam}. "
        "Close other apps using the camera, or try --cam 1."
    )

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

INPUT = (args.size, args.size)

def preprocess(bgr_frame):
    """Center-crop → resize → RGB → [0,1]."""
    h, w = bgr_frame.shape[:2]
    side = min(h, w)
    y0, x0 = (h - side) // 2, (w - side) // 2
    crop = bgr_frame[y0:y0+side, x0:x0+side]
    img = cv2.resize(crop, INPUT, interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return img[None, ...]

print("Starting webcam. Press Q to quit.")
last_time = time.time()

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Could not read frame from camera.")
            time.sleep(0.05)
            continue

        if args.mirror:
            frame = cv2.flip(frame, 1)

        # prediction
        probs = model.predict(preprocess(frame), verbose=0)[0]
        idx = int(np.argmax(probs))
        label = CLASS_NAMES[idx]
        conf = float(probs[idx])

        disp = frame.copy()

        # ---- semi-transparent top bar ----
        overlay = disp.copy()
        cv2.rectangle(overlay, (0, 0), (disp.shape[1], 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, disp, 0.4, 0, disp)

        # ---- show top-3 predictions ----
        top3_idx = np.argsort(probs)[-3:][::-1]
        for i, id_ in enumerate(top3_idx):
            txt = f"{CLASS_NAMES[id_]}: {probs[id_]*100:.1f}%"
            color = (0, 255, 0) if i == 0 else (200, 200, 200)
            cv2.putText(disp, txt, (10, 35 + i*30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

        # ---- FPS counter ----
        curr_time = time.time()
        fps = 1.0 / (curr_time - last_time)
        last_time = curr_time
        cv2.putText(disp, f"FPS: {fps:.1f}",
                    (disp.shape[1]-140, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)

        # ---- instructions ----
        cv2.putText(disp, "Press Q to Quit",
                    (disp.shape[1]-280, disp.shape[0]-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        # ---- show window ----
        cv2.imshow("ASL Recognition", disp)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
