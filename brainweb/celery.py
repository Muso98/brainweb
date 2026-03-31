# brainweb/celery.py
import os
from celery import Celery

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "brainweb.settings")

app = Celery("brainweb")

# Django settings ichidagi CELERY_ prefiksli sozlamalarni o'qiydi
app.config_from_object("django.conf:settings", namespace="CELERY")

# Barcha installed app'larda tasks.py ni avtomatik topadi
app.autodiscover_tasks()
