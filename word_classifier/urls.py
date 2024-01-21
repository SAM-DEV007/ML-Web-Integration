from django.urls import path
from . import views


app_name = 'word_classifier'
urlpatterns = [
    path('', views.home, name='wc_home'),
    path('model', views.model, name='wc_model'),
]