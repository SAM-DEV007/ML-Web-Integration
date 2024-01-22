# ML-Web-Integration
Web integration of ML projects

# About
Website link: https://ml-integrations.onrender.com/

****Warning: The website is hosted on a free service platform with highly limited resources below the recommendations of machine learning models. The website loading may be slow and the webcam streaming may hang. To fully experience the website, it is recommended to clone the repository and use local webhosting.***

***It is advised to not use the word classifier section (using PREDICT button) from the above link as it will crash the website (usage of memory above free quota).****

The backend is written in Python based on Django. The website aims to feature most of the Machine Learning projects done by me and stack them in one place.
`build.sh` is for production purpose and can be ignored.

The repositories of the projects included:
- [Instagram Filters](https://github.com/SAM-DEV007/Instagram-Filters)
- [Word Classifier](https://github.com/SAM-DEV007/Word-Classifier)

# Installation
- Python 3.x version.
- `requirements.txt` for the dependencies.

### Clone the repository
```
git clone https://github.com/SAM-DEV007/ML-Web-Integration.git
cd ML-Web_Integration
```
### Create a virtual environment (optional)
```
python -m venv .venv
.venv\Scripts\activate.bat
```
### Requirements
```
pip install -r requirements.txt
```
- Download Word Classifier model: 
```
https://drive.google.com/file/d/1LAiyCV0p6v-lROdXbtrzKlF4APNmM3Qm/view?usp=sharing
```
- Save it in the location: 
```
ML-Web-Integration/word_classifier/Model/WordClassifier_Model.h5
```
### Initialize
```
python manage.py makemigrations
python manage.py migrate
python manage.py collectstatic
```
### Run the local webserver
`127.0.0.1` is for the local machine only. To apply for the available IP (use with other devices) - `0.0.0.0`.
```
daphne -b 127.0.0.1 -p 8000 ML_Integration.asgi:application
```
