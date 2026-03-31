web: gunicorn brainweb.wsgi --bind 0.0.0.0:$PORT --workers 2 --timeout 120
worker: python -m celery -A brainweb worker -l info --concurrency 1
release: python manage.py migrate --noinput && python manage.py collectstatic --noinput
