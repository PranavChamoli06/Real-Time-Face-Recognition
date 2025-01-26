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

---

To successfully run your Real-Time Face Recognition project, follow these steps and keep these considerations in mind:

---

### Step 1: Install the Required Libraries:
Before running the scripts, ensure you have all necessary libraries installed. Use the following command:

```bash
pip install opencv-python mediapipe tensorflow keras numpy pillow scikit-learn
```

---

### Step 2: Prepare the Project Files:
1. Folder Structure:
   Ensure your project folder looks like this:
   ```
   Real-Time-Face-Recognition/
   ├── 01_face_dataset.py
   ├── 02_face_training.py
   ├── 03_face_recognition.py
   ├── Dataset Maker.py
   ├── dataset/
   │   ├── labels.txt
   │   └── User.<ID>.<count>.jpg
   ├── trained_model.h5
   └── README.md
   ```

2. Empty `dataset/` Folder:
   - `dataset/` will store captured face images (`User.<ID>.<count>.jpg`).
   - `labels.txt` will map user IDs to their names. Ensure this file exists, even if it’s empty initially.

---

### Step 3: Capture Face Images:
1. Run the **`01_face_dataset.py`** script:
   ```bash
   python 01_face_dataset.py
   ```

2. Follow the prompts:
   - Enter a unique numeric ID (e.g., `0`) for the user.
   - Enter the user’s name (e.g., `Alice`).

3. Look at the webcam for about 30-60 seconds to allow the script to capture 70 face images.

4. Repeat the process for multiple users:
   - Run the script again.
   - Provide a new ID and name for each additional user.

5. Verify:
   - The `dataset/` folder contains images like `User.0.0.jpg`, `User.1.0.jpg`, etc.
   - `labels.txt` contains entries like:
     ```
     0    Alice
     1    Bob
     ```

---

### Step 4: Train the Model:
1. Run the **`02_face_training.py`** script:
   ```bash
   python 02_face_training.py
   ```

2. What it does:
   - Reads images from the `dataset/` folder.
   - Preprocesses them (resizes to 32x32, normalizes pixel values).
   - Trains a Convolutional Neural Network (CNN) to recognize the faces.
   - Saves the trained model as `trained_model.h5`.

3. Verify:
   - Ensure `trained_model.h5` is created in the project folder.

---

### Step 5: Perform Real-Time Face Recognition:
1. Run the **`03_face_recognition.py`** script:
   ```bash
   python 03_face_recognition.py
   ```

2. What it does:
   - Loads the trained model (`trained_model.h5`) and `labels.txt`.
   - Uses the webcam to detect faces in real time.
   - Recognizes faces and displays the predicted name on the video feed.

3. Test:
   - Move in front of the webcam to see if the script recognizes you.
   - Switch between different users to test accuracy.

---
