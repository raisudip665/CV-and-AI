from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from ultralytics import YOLO
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
obj_model = YOLO("yolo11l.pt")
emotion_model = YOLO("emotion.onnx")
emotion_classes = ["Angry", "Fearful", "Happy", "Neutral", "Sad"]

# Hand Landmarker
BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.4,  # Lower for better detection
    min_tracking_confidence=0.4
)
hands = HandLandmarker.create_from_options(options)

def get_hand_activity(landmarks, handedness):
    if not landmarks:
        return "No hand"
    
    # Correct Left/Right (flip for mirror view)
    side = "Left" if handedness[0].category_name == "Right" else "Right"  # Flip for mirror

    # Raise hand (wrist above middle finger MCP with threshold)
    wrist = landmarks[0]
    middle_mcp = landmarks[9]
    raised = wrist.y < middle_mcp.y - 0.05  # Lower threshold for better detection

    # Accurate finger count
    tips = [4, 8, 12, 16, 20]
    pips = [3, 6, 10, 14, 18]
    extended = 0

    # Thumb (x position based on side)
    if side == "Left":
        if landmarks[4].x > landmarks[3].x + 0.05:  # Adjusted threshold
            extended += 1
    else:
        if landmarks[4].x < landmarks[3].x - 0.05:
            extended += 1

    # Other fingers (y position with threshold)
    for tip, pip in zip(tips[1:], pips[1:]):
        if landmarks[tip].y < landmarks[pip].y - 0.03:  # Adjusted for accuracy
            extended += 1

    activity = f"{side} HAND"
    if raised:
        activity += " RAISED"
    activity += f" | Fingers: {extended}/5"
    return activity

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"objects": [], "emotions": [], "hands": []}
    frame = cv2.flip(frame, 1)

    # Object Detection
    obj_results = obj_model(frame, conf=0.5, verbose=False)[0]
    objects = []
    for box in obj_results.boxes:
        x1,y1,x2,y2 = map(int, box.xyxy[0])
        label = obj_model.names[int(box.cls[0])]
        conf = float(box.conf[0])
        objects.append({"label": label, "conf": conf, "box": [x1,y1,x2,y2]})

    # Emotion Detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray3 = cv2.merge([gray, gray, gray])
    emo_results = emotion_model(gray3, conf=0.4, verbose=False)[0]
    emotions = []
    for box in emo_results.boxes:
        x1,y1,x2,y2 = map(int, box.xyxy[0])
        emo = emotion_classes[int(box.cls[0])]
        conf = float(box.conf[0])
        emotions.append({"label": emo, "conf": conf, "box": [x1,y1,x2,y2]})

    # Hand Activity
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
    hand_results = hands.detect_for_video(mp_image, timestamp_ms)
    hands_data = []
    if hand_results.hand_landmarks:
        for landmarks, handedness in zip(hand_results.hand_landmarks, hand_results.handedness):
            x_coords = [lm.x * frame.shape[1] for lm in landmarks]
            y_coords = [lm.y * frame.shape[0] for lm in landmarks]
            x1, y1 = int(min(x_coords)), int(min(y_coords))
            x2, y2 = int(max(x_coords)), int(max(y_coords))
            activity = get_hand_activity(landmarks, handedness)
            hands_data.append({"activity": activity, "box": [x1,y1,x2,y2]})

    return {"objects": objects, "emotions": emotions, "hands": hands_data}

print("Server Ready! Run with uvicorn server:app --reload")