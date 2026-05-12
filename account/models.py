from django.db import models
from django.contrib.auth.models import User

# Create your models here.

class Employ(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    profile_image = models.ImageField(upload_to='profile/')
    phone = models.CharField(max_length=12)
    department = models.CharField(max_length=20)
    created_at = models.DateTimeField(auto_now_add = True)

    def __str__(self):
        return self.user.username