set -o errexit

pip install -r requirements.txt

python manage.py collectstatic --no-input
python manage.py migrate

python word_classifier/download_model.py
python human_emotions/download_model.py