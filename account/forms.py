from django import forms
from django.contrib.auth.models import User
from .models import Employ

class Register(forms.ModelForm):
    username = forms.CharField(max_length=20)
    email = forms.EmailField()
    password = forms.CharField(widget=forms.PasswordInput)

    class Meta:
        model = Employ
        fields = ['phone', 'department', 'profile_image']