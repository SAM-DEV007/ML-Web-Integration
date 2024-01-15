from django.urls import path
from . import views

urlpatterns = [
    path('flappy_bird/', views.flappy_bird, name='flappy_bird'),
    path('maths_equation/', views.maths_equation, name='maths_equation'),
    path('hand_gesture/', views.hand_gesture, name='hand_gesture'),
]