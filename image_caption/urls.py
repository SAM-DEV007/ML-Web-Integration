from django.urls import path
from . import views


app_name = 'image_caption'
urlpatterns = [
    path('', views.home, name='ic_home'),
    path('model', views.model, name='ic_model'),
]