import os
import base64
from django.conf import settings
from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from .models import Employ
from .forms import Register

def register(request):

    if request.method == 'POST':

        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        phone = request.POST.get('phone')
        department = request.POST.get('department')

        user = User.objects.create_user(
            username=username,
            email=email,
            password=password
        )

        employee = Employ.objects.create(
            user=user,
            phone=phone,
            department=department,
            profile_image=request.FILES.get('profile_image')
        )

        images = request.POST.getlist('images[]')

        dataset_path = os.path.join(
            settings.MEDIA_ROOT,
            'datasets',
            username
        )

        os.makedirs(dataset_path, exist_ok=True)

        for index, image_data in enumerate(images):

            format, imgstr = image_data.split(';base64,')
            ext = format.split('/')[-1]

            image_file = base64.b64decode(imgstr)

            file_path = os.path.join(
                dataset_path,
                f'{index}.jpg'
            )

            with open(file_path, 'wb') as f:
                f.write(image_file)

        return redirect('/admin')

    return render(request, 'account/register.html')