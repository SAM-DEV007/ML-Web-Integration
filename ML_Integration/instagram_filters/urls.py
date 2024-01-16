from django.urls import path
from . import views


app_name = 'instagram_filters'
urlpatterns = [
    path('', views.home, name='ig_home'),
    path('flappy_bird/', views.flappy_bird, name='flappy_bird'),
    path('flappy_bird_feed', views.stream_flappy_bird, name='flappy_bird_feed'),
    path('maths_equation/', views.maths_equation, name='maths_equation'),
    path('hand_gesture/', views.hand_gesture, name='hand_gesture'),
]