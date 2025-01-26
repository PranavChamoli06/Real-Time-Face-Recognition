import os
import cv2
import numpy as np
from PIL import Image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras import callbacks

# Path for face image database
path = 'dataset'

# Function to downsample images
def downsample_image(img):
    img = Image.fromarray(img.astype('uint8'), 'L')
    img = img.resize((32, 32), Image.LANCZOS)
    return np.array(img)

# Function to get images and label data
def getImagesAndLabels(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset folder '{path}' does not exist. Please run the dataset creation script.")

    imagePaths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
    if len(imagePaths) == 0:
        raise ValueError(f"No images found in '{path}'. Please ensure the dataset is populated correctly.")

    faceSamples = []
    ids = []

    for imagePath in imagePaths:
        try:
            # Read image, convert to grayscale, and downsample
            img = cv2.imread(imagePath)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            downsampled = downsample_image(gray)

            # Extract ID from filename
            id = int(os.path.split(imagePath)[-1].split(".")[1])
            faceSamples.append(downsampled)
            ids.append(id)
        except Exception as e:
            print(f"Error processing file {imagePath}: {e}")
            continue

    return np.array(faceSamples), np.array(ids)

# Main script
print("\n[INFO] Loading dataset and preparing for training...")
try:
    faces, ids = getImagesAndLabels(path)
except Exception as e:
    print(f"[ERROR] {e}")
    exit()

# Normalize and preprocess data
faces = faces.astype('float32') / 255.0
faces = faces[..., np.newaxis]  # Add channel dimension
ids = to_categorical(ids)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(faces, ids, test_size=0.2, random_state=0)

# Build CNN model
n_faces = ids.shape[1]
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(n_faces, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define checkpoint to save the best model
checkpoint = callbacks.ModelCheckpoint(
    './trained_model.h5',
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)

# Train the model
print("\n[INFO] Training the model...")
model.fit(
    x_train, y_train,
    batch_size=32,
    epochs=10,
    validation_data=(x_test, y_test),
    shuffle=True,
    callbacks=[checkpoint]
)

# Training complete
print(f"\n[INFO] Model training complete. {n_faces} unique faces trained.")
print("[INFO] Trained model saved as 'trained_model.h5'.")
