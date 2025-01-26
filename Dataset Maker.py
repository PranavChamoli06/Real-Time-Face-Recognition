from sklearn.datasets import fetch_lfw_people
import numpy as np
from PIL import Image

def downsample_image(img):
    img = Image.fromarray(img.astype('uint8'), 'L')
    img = img.resize((32, 32), Image.ANTIALIAS)
    return np.array(img)

def get_face_data():
    people = fetch_lfw_people(color=False, min_faces_per_person=300)
    X_faces = people.images
    X_faces = np.array([downsample_image(img) for img in X_faces])
    Y_faces = people.target
    names = people.target_names
    return X_faces, Y_faces, names

X_faces, Y_faces, names = get_face_data()
np.save('x_data.npy', X_faces)
np.save('y_data.npy', Y_faces)
print("[INFO] Datasets saved.")
