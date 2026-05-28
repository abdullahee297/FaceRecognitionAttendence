import os
import pickle
import numpy as np

from django.conf import settings

known_faces = []
known_usernames = []

from deepface import DeepFace
import pickle

def generate_face_encoding(image_path, save_path):

    embedding = DeepFace.represent(
        img_path=image_path,
        model_name='Facenet',
        enforce_detection=False
    )

    with open(save_path, 'wb') as f:
        pickle.dump(embedding, f)

def load_known_faces():

    global known_faces
    global known_usernames

    known_faces.clear()
    known_usernames.clear()

    encoding_root = os.path.join(
        settings.MEDIA_ROOT,
        'encodings'
    )

    for username in os.listdir(encoding_root):

        user_folder = os.path.join(
            encoding_root,
            username
        )

        for file in os.listdir(user_folder):

            path = os.path.join(
                user_folder,
                file
            )

            with open(path, 'rb') as f:

                embedding = pickle.load(f)

                known_faces.append(
                    embedding[0]['embedding']
                )

                known_usernames.append(username)