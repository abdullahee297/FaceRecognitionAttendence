from django.db import models
from account.models import Employ

# Create your models here.

class Attendance(models.Model):
    employee = models.ForeignKey(Employ, on_delete=models.CASCADE)
    check_in = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=20, default='Preent')

    def __str__(self):
        return f"{self.employee.user.username} - {self.check_in}"