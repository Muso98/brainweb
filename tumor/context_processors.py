# tumor/context_processors.py
from django.utils import timezone

def current_time(request):
    """Provide server current time as 'now' to all templates."""
    return {"now": timezone.now()}
