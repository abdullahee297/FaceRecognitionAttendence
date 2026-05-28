from django.shortcuts import render
from django.contrib.auth.decorators import login_required

from account.models import Employee
from .models import Attendance

@login_required
def dashboard(request):

    employee = Employee.objects.get(
        user=request.user
    )

    attendances = Attendance.objects.filter(
        employee=employee
    ).order_by('-check_in')

    return render(
        request,
        'account/dashboard.html',
        {
            'attendances': attendances
        }
    )