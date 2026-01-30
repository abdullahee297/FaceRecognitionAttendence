import cv2
import csv
import time
import numpy as np 
import mediapipe as mp 
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BaseOptions = python.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = FaceLandmarkerOptions(
    base_options = BaseOptions(model_asset_path="face_landmarker.task"),
    running_mode = VisionRunningMode.IMAGE,
    num_faces = 1
)

detector = FaceLandmarker.create_from_options(options)


# cap = cv2.VideoCapture(0)

def imgconvert(imgaepath):
    image = cv2.imread(imgaepath)
    converted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



img = cv2.imread("img3_test.jpg")

if img is None:
    print("Image not found")
    exit()

# Convert to RGB
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

# Detect face
result = detector.detect(mp_image)

# Draw landmarks
if result.face_landmarks:
    for face in result.face_landmarks:
        h, w, _ = img.shape
        x_list = []
        y_list = []

        for lm in face:
            x = int(lm.x * w)
            x_list.append(x)
            y = int(lm.y * h)
            y_list.append(y)
            
            # cv2.circle(img, (x, y), 1, (0, 255, 255), cv2.FILLED)

        x_min, x_max = min(x_list), max(x_list)
        y_min, y_max = min(y_list), max(y_list)
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)


screen_width = 720
screen_height = 800

img = cv2.resize(img, (screen_width, screen_height))
cv2.imshow("Face Detection", img)
cv2.waitKey(0)