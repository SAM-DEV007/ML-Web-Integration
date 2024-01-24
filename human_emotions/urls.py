from django.urls import path
from . import views


app_name = 'human_emotions'
urlpatterns = [
    path('', views.home, name='he_home'),
    path('model', views.model, name='he_model'),
]