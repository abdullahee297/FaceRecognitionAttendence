from django.db import models
from account.models import Employee

# Create your models here.
class FaceEmbedding(models.Model):
    employ = models.OneToOneField(
        Employee, on_delete=models.CASCADE
    )

    encodin_path = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.employ.user.username