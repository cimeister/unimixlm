from celery.decorators import periodic_task
from celery import shared_task
from .models import *

import datetime

@shared_task
def promocodechecker():

    codes = Promocode.objects.filter(active=True)

    for code in codes:
        if (code.validity - datetime.date.today()).days <0:
            code.active = False
            code.save()