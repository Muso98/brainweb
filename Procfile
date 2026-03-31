web: python manage.py migrate --noinput && python manage.py collectstatic --noinput && gunicorn brainweb.wsgi:application --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120 --log-level info
