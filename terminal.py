import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp

# === LOAD MODELS ===
obj_model = YOLO("yolo11l.pt")               # Object detection
emotion_model = YOLO("emotion.onnx")         # Emotion detection
emotion_classes = ["Angry", "Fearful", "Happy", "Neutral", "Sad"]

# === MediaPipe Hand Landmarker (using your .task file) ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# === HAND ACTIVITY DETECTION ===
def get_hand_activity(landmarks, handedness):
    if not landmarks:
        return "No hand detected"
    
    side = "Left" if handedness.classification[0].label == "Left" else "Right"
    
    # Raise hand: wrist higher than middle finger MCP
    wrist = landmarks[0]
    middle_mcp = landmarks[9]
    raised = wrist.y < middle_mcp.y - 0.1

    # Count extended fingers
    tips = [4, 8, 12, 16, 20]
    pips = [2, 6, 10, 14, 18]
    extended = sum(1 for tip, pip in zip(tips, pips) if landmarks[tip].y < landmarks[pip].y)

    activity = f"{side} HAND"
    if raised:
        activity += " RAISED"
    activity += f" | Fingers: {extended}/5"
    return activity

# === MAIN LOOP ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open camera!")
    exit()

print("\n" + "="*60)
print("    AR TERMINAL APP - OBJECT + EMOTION + HAND ACTIVITY    ")
print("="*60)
print("• Green boxes: Objects")
print("• Yellow boxes: Emotions with emoji")
print("• Red banner: Hand activity")
print("Press 'q' to quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = frame.shape[:2]

    # === OBJECT DETECTION ===
    obj_res = obj_model(frame, conf=0.5, verbose=False)[0]
    for box in obj_res.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = obj_model.names[int(box.cls[0])]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(frame, label.upper(), (x1, y1 - 15), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0), 3)

    # === EMOTION DETECTION ===
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray3 = cv2.merge([gray, gray, gray])
    emo_res = emotion_model(gray3, conf=0.4, verbose=False)[0]
    emo_emojis = {"Happy": "😊", "Sad": "😢", "Angry": "😠", "Fearful": "😨", "Neutral": "😐"}
    for box in emo_res.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        emo = emotion_classes[int(box.cls[0])]
        emoji = emo_emojis.get(emo, "😐")
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 4)
        cv2.putText(frame, f"{emoji} {emo.upper()} {emoji}", (x1, y1 - 20), 
                    cv2.FONT_HERSHEY_DUPLEX, 1.3, (255, 255, 0), 4)

    # === HAND ACTIVITY ===
    hand_res = hands.process(rgb)
    hand_text = "No hand detected"
    if hand_res.multi_hand_landmarks:
        for landmarks, handedness in zip(hand_res.multi_hand_landmarks, hand_res.multi_handedness):
            mp_draw.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS,
                                   mp_draw.DrawingSpec(color=(0, 255, 255), thickness=3),
                                   mp_draw.DrawingSpec(color=(255, 0, 255), thickness=3))
            hand_text = get_hand_activity(landmarks.landmark, handedness)

    # Big red banner for hand activity
    banner = np.zeros((100, w, 3), dtype=np.uint8)
    banner[:] = (0, 0, 180)  # Dark red
    cv2.putText(banner, hand_text, (20, 65), cv2.FONT_HERSHEY_TRIPLEX, 1.8, (255, 255, 255), 3)
    frame = np.vstack((frame, banner))

    cv2.imshow("AR Terminal App - New UI", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\nAR app closed. Goodbye!")