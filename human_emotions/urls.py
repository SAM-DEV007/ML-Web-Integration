from django.urls import path
from . import views


app_name = 'human_emotions'
urlpatterns = [
    path('', views.home, name='wc_home'),
]