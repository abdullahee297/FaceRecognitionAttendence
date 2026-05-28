import os
import cv2
import json
import base64
import pickle
import numpy as np
from django.http import JsonResponse
from django.shortcuts import render
from django.conf import settings
from deepface import DeepFace
from account.models import Employee
from attendance.models import Attendance
from datetime import date

from .utils import load_known_faces
from .utils import known_faces
from .utils import known_usernames

load_known_faces()

def scan_page(request):
    return render(request, 'recognition/scan.html')


def scan_face(request):

    if request.method == 'POST':

        data = json.loads(request.body)

        image_data = data['image']

        format, imgstr = image_data.split(';base64,')

        image_bytes = base64.b64decode(imgstr)

        np_arr = np.frombuffer(image_bytes, np.uint8)

        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        temp_path = os.path.join(
            settings.MEDIA_ROOT,
            'temp.jpg'
        )

        cv2.imwrite(temp_path, frame)

        try:

            current_embedding = DeepFace.represent(
                img_path=temp_path,
                model_name='Facenet',
                enforce_detection=False
            )

            matched_employee = None

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

                        saved_embedding = pickle.load(f)

                    distance = np.linalg.norm(
                        np.array(
                            current_embedding[0]['embedding']
                        ) -

                        np.array(
                            saved_embedding[0]['embedding']
                        )
                    )

                    if distance < 4:

                        employee = Employee.objects.get(
                            user__username=username
                        )

                        matched_employee = employee

                        break

            if matched_employee:

                already_marked = Attendance.objects.filter(
                    employee=matched_employee,
                    check_in__date=date.today()
                ).exists()

                if not already_marked:

                    Attendance.objects.create(
                        employee=matched_employee
                    )

                return JsonResponse({
                    'message':
                    f'Attendance Marked For {matched_employee.user.username}'
                })

            return JsonResponse({
                'message': 'Face Not Recognized'
            })

        except Exception as e:

            return JsonResponse({
                'message': str(e)
            })