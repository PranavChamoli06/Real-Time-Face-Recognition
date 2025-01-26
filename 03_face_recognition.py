import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model

mp_face_detection = mp.solutions.face_detection
model = load_model('trained_model.h5')

with open('dataset/labels.txt', 'r') as file:
    labels = dict(line.strip().split('\t') for line in file)

def start():
    cap = cv2.VideoCapture(0)
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(frame_rgb)

            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

                    face = frame[y:y + h, x:x + w]
                    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                    gray_resized = cv2.resize(gray, (32, 32)).astype('float32') / 255.0
                    gray_resized = gray_resized.reshape(-1, 32, 32, 1)

                    prediction = model.predict(gray_resized)
                    label_id = np.argmax(prediction)
                    label = labels.get(str(label_id), "Unknown")

                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow('Face Recognition', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

start()
