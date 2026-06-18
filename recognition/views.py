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

load_known_faces()


def scan_page(request):
    return render(request, 'recognition/scan.html')


def scan_face(request):

    if request.method != 'POST':
        return JsonResponse({
            'status': 'error',
            'message': 'Only POST method allowed'
        })

    try:
        # -----------------------------
        # 1. Decode base64 image
        # -----------------------------
        data = json.loads(request.body)
        image_data = data.get('image')

        if not image_data:
            return JsonResponse({
                'status': 'error',
                'message': 'No image received'
            })

        _, imgstr = image_data.split(';base64,')
        image_bytes = base64.b64decode(imgstr)

        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # -----------------------------
        # 2. Save temporary image
        # -----------------------------
        temp_path = os.path.join(settings.MEDIA_ROOT, 'temp.jpg')
        cv2.imwrite(temp_path, frame)

        # -----------------------------
        # 3. Get embedding
        # -----------------------------
        result = DeepFace.represent(
            img_path=temp_path,
            model_name='Facenet',
            enforce_detection=False
        )

        current_embedding = np.array(result[0]['embedding'])

        # -----------------------------
        # 4. Compare with stored encodings
        # -----------------------------
        encoding_root = os.path.join(settings.MEDIA_ROOT, 'encodings')

        matched_employee = None

        if not os.path.exists(encoding_root):
            return JsonResponse({
                'status': 'error',
                'message': 'Encodings folder not found'
            })

        for username in os.listdir(encoding_root):

            user_folder = os.path.join(encoding_root, username)

            if not os.path.isdir(user_folder):
                continue

            for file in os.listdir(user_folder):

                path = os.path.join(user_folder, file)

                with open(path, 'rb') as f:
                    saved_embedding = pickle.load(f)

                saved_embedding = np.array(saved_embedding[0]['embedding'])

                distance = np.linalg.norm(current_embedding - saved_embedding)

                # -----------------------------
                # 5. Threshold (IMPORTANT FIX)
                # -----------------------------
                if distance < 10:

                    try:
                        matched_employee = Employee.objects.get(
                            user__username=username
                        )
                    except Employee.DoesNotExist:
                        continue

                    break

            if matched_employee:
                break

        # -----------------------------
        # 6. Attendance Marking
        # -----------------------------
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
                'status': 'success',
                'message': f'Attendance Marked for {matched_employee.user.username}'
            })

        return JsonResponse({
            'status': 'error',
            'message': 'Face Not Recognized'
        })

    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        })
