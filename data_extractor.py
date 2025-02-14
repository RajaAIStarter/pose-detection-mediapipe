import time
import cv2
import mediapipe as mp
import csv

# Initialize MediaPipe Pose and Drawing modules
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

time.sleep(10)
print('READDDYYYYYYY')

# Initialize webcam
cap = cv2.VideoCapture(0)

# Desired width and height for resizing
desired_width = 500
desired_height = 400

# Define a dictionary to map landmark indices to their names
landmark_names = {
    11: 'Left Shoulder',
    12: 'Right Shoulder',
    13: 'Left Elbow',
    14: 'Right Elbow',
    15: 'Left Wrist',
    16: 'Right Wrist',
    23: 'Left Hip',
    24: 'Right Hip',
    25: 'Left Knee',
    26: 'Right Knee',
    27: 'Left Ankle',
    28: 'Right Ankle'
}

# Define a label variable for the pose; update this variable based on your pose logic
pose_label = "endurance"  # Change this label as needed

start_time = time.time()  # Record the start time

total_duration = 6 # Run for 10 seconds

# Open the CSV file once before entering the loop
with open('pose_landmarks.csv', mode='a', newline='') as file:
    writer = csv.writer(file)

    with mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7) as pose:
        while cap.isOpened():
            if time.time() - start_time > total_duration:
                break

            ret, frame = cap.read()
            if not ret:
                break

            # Resize the frame to the desired resolution
            frame_resized = cv2.resize(frame, (desired_width, desired_height))

            # Convert the BGR image to RGB
            rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            # Process the frame and detect pose landmarks
            results = pose.process(rgb_frame)

            if results.pose_landmarks:
                # Extract landmark coordinates
                landmarks = results.pose_landmarks.landmark

                # Initialize a list to hold the row data for CSV
                row = []
                for idx, name in landmark_names.items():
                    # Get the landmark
                    landmark = landmarks[idx]
                    # Get normalized x and y coordinates
                    x = landmark.x
                    y = landmark.y
                    # Convert normalized coordinates to pixel values
                    h, w, _ = frame_resized.shape
                    x_px = int(x * w)
                    y_px = int(y * h)
                    # Print the coordinates
                    print(f'{name} - x: {x_px}, y: {y_px}')
                    # Append coordinates to the row list
                    row.extend([x_px, y_px])
                    # Optionally, draw a circle on the landmark
                    cv2.circle(frame_resized, (x_px, y_px), 5, (0, 255, 255), -1)

                # Append the label variable as the last element in the row
                row.append(pose_label)
                writer.writerow(row)

                # Draw all pose landmarks on the frame
                mp_drawing.draw_landmarks(
                    frame_resized, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Display the frame
            cv2.imshow('MediaPipe Pose Estimation', frame_resized)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

# Release resources
cap.release()
cv2.destroyAllWindows()
