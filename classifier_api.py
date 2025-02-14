import cv2
import mediapipe as mp
import joblib
import numpy as np
import math
import warnings

from rich.emoji import EmojiVariant

warnings.filterwarnings('ignore')

# Load saved artifacts (trained on 24 features: raw landmark coordinates).
model = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')
le = joblib.load('label_encoder.pkl')

# Setup MediaPipe Pose and drawing utilities.
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Settings
width, height = 640, 480
USE_IMAGE = False  # Set to True to test with an image file; False for live webcam feed.
test_image_path = "/home/raja/Documents/project/DATASET/TEST/flexibility/00000000.jpg"  # Update with your test image path.
CONF_THRESHOLD = 0.60  # Confidence threshold (80%)
EMOJI_CONF_THRESHOLD = 0.80  # Confidence threshold (80%)


# Define landmark indices mapping (should match training order).
landmark_indices = {
    'Left Shoulder': 11,
    'Right Shoulder': 12,
    'Left Elbow': 13,
    'Right Elbow': 14,
    'Left Wrist': 15,
    'Right Wrist': 16,
    'Left Hip': 23,
    'Right Hip': 24,
    'Left Knee': 25,
    'Right Knee': 26,
    'Left Ankle': 27,
    'Right Ankle': 28
}

# Define the order of landmarks (to extract 24 features in the correct order).
ordered_landmarks = [
    'Left Shoulder',
    'Right Shoulder',
    'Left Elbow',
    'Right Elbow',
    'Left Wrist',
    'Right Wrist',
    'Left Hip',
    'Right Hip',
    'Left Knee',
    'Right Knee',
    'Left Ankle',
    'Right Ankle'
]

# Initialize MediaPipe Pose.
pose_obj = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)


def process_frame(frame):
    frame_resized = cv2.resize(frame, (width, height))
    h, w, _ = frame_resized.shape
    rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    results = pose_obj.process(rgb_frame)

    prediction_text = "No pose detected 🙂. Do something!"

    if results.pose_landmarks:
        features = []
        for name in ordered_landmarks:
            idx = landmark_indices[name]
            lm = results.pose_landmarks.landmark[idx]
            x_px = int(lm.x * w)
            y_px = int(lm.y * h)
            features.extend([x_px, y_px])
            cv2.circle(frame_resized, (x_px, y_px), 5, (0, 255, 0), -1)

        features = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features)
        probs = model.predict_proba(features_scaled)
        max_prob = np.max(probs)
        predicted_class = np.argmax(probs)
        predicted_label = le.inverse_transform([predicted_class])[0]

        if max_prob < CONF_THRESHOLD:
            prediction_text = "No confident pose 🙂. Do some pose!"
        else:
            if predicted_label.lower() in ["endurance", "flexibility"]:
                if max_prob >= EMOJI_CONF_THRESHOLD:
                    prediction_text = "Endurance & Flexibility Pose Activated - YOU ROCK! 🤩"
                else:
                    prediction_text = "Endurance & Flexibility Pose Activated ☺️"
            elif predicted_label.lower() == "weight_loss":
                if max_prob >= EMOJI_CONF_THRESHOLD:
                    prediction_text = "Weight Loss Pose Activated - YOU ROCK! 🤩"
                else:
                    prediction_text = "Weight Loss Pose Activated ☺️"
            elif predicted_label.lower() == "stability":
                if max_prob >= EMOJI_CONF_THRESHOLD:
                    prediction_text = "Stability Pose Activated - YOU ROCK! 🤩"
                else:
                    prediction_text = "Stability Pose Activated ☺️"
            elif predicted_label.lower() == "strength":
                if max_prob >= EMOJI_CONF_THRESHOLD:
                    prediction_text = "Strength Pose Activated - YOU ROCK! 🤩"
                else:
                    prediction_text = "Strength Pose Activated ☺️"
            else:
                prediction_text = f"{predicted_label} Pose Activated ☺️"

        mp_drawing.draw_landmarks(frame_resized, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Instead of overlaying text on the frame, print it to the console.
    print(prediction_text)
    return frame_resized

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    output_frame = process_frame(frame)
    cv2.imshow("Live Pose Classification", output_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()

pose_obj.close()
cv2.destroyAllWindows()
