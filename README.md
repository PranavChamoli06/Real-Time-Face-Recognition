# Real-Time Face Recognition

A Python-based real-time face recognition system using CNN, OpenCV, and Mediapipe.

## Features
- Capture face images using `01_face_dataset.py`.
- Train a Convolutional Neural Network (CNN) using `02_face_training.py`.
- Perform real-time face recognition with `03_face_recognition.py`.

## Requirements
- Python 3.7+
- Libraries: OpenCV, Mediapipe, TensorFlow/Keras, NumPy, Pillow, Scikit-learn

## Usage
1. Run `01_face_dataset.py` to capture face images.
2. Run `02_face_training.py` to train the model.
3. Run `03_face_recognition.py` for real-time recognition.

## Folder Structure
Real-Time-Face-Recognition/
├── 01_face_dataset.py          # Script to capture face images and create the dataset
├── 02_face_training.py         # Script to train the CNN model on the dataset
├── 03_face_recognition.py      # Script to perform real-time face recognition
├── Dataset Maker.py            # (Optional) Script to augment the dataset with external data
├── dataset/                    # Folder containing face images and labels
│   ├── labels.txt              # File mapping user IDs to names
│   └── User.<ID>.<count>.jpg   # Captured face images for each user
├── trained_model.h5            # Trained CNN model saved after running the training script
└── README.md                   # Documentation file for the project

