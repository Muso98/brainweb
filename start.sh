#!/bin/sh
# Do NOT set -e here so we can continue even if tasks fail
echo "--- Starting BrainWeb Deployment ---"

# 1. Run migrations
echo "Running migrations..."
python manage.py migrate --noinput || echo "WARNING: migrations failed! Check your DATABASE_URL."

# 2. Collect static files
echo "Collecting static files..."
python manage.py collectstatic --noinput || echo "WARNING: collectstatic failed!"

# 3. Start Gunicorn
# Explicitly use 0.0.0.0 and PORT. 
echo "Starting Gunicorn on port ${PORT:-8000}..."
exec gunicorn brainweb.wsgi:application     --bind 0.0.0.0:${PORT:-8000}     --workers 1     --threads 4     --timeout 120     --log-level debug     --access-logfile -     --error-logfile -
