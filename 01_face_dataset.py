import cv2
import os
import mediapipe as mp

# Mediapipe setup for face detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Prompt user for ID and name
face_id = input('\nEnter user ID (numeric): ')
face_name = input(f'\nEnter name for ID {face_id}: ')

# Create dataset directory if it doesn't exist
dataset_path = 'dataset'
if not os.path.exists(dataset_path):
    os.mkdir(dataset_path)

# Update labels.txt for the new user
labels_file = os.path.join(dataset_path, 'labels.txt')
with open(labels_file, 'a') as file:
    file.write(f"{face_id}\t{face_name}\n")

# Initialize webcam
cam = cv2.VideoCapture(0)
print("\n[INFO] Initializing face capture. Look at the camera and wait...")

count = 0
while True:
    ret, frame = cam.read()
    if not ret:
        print("[ERROR] Unable to access webcam.")
        break

    # Convert frame to RGB for Mediapipe processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(frame_rgb)

    # Detect faces and save images
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

            # Extract and save the face
            face = frame[y:y + h, x:x + w]
            face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(f"{dataset_path}/User.{face_id}.{count}.jpg", face_gray)
            count += 1

            # Draw a bounding box around the detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Show the video feed with bounding boxes
    cv2.imshow('Capturing Faces', frame)

    # Exit conditions: 'ESC' key or 70 images captured
    if cv2.waitKey(1) & 0xFF == 27 or count >= 70:
        break

# Clean up resources
print("\n[INFO] Dataset creation complete. Exiting...")
cam.release()
cv2.destroyAllWindows()
