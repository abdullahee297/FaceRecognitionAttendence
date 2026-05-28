from django.urls import path
from .views import scan_page, scan_face

urlpatterns = [
    path('scan/', scan_page, name='scan'),
    path('scan-face/', scan_face, name='scan_face'),
]