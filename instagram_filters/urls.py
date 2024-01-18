from django.urls import path
from . import views


app_name = 'instagram_filters'
urlpatterns = [
    path('', views.home, name='ig_home'),
    path('video', views.img_path, name='image_path'),
    path('flappy_bird', views.flappy_bird, name='flappy_bird'),
    path('flappy_bird_feed', views.stream_flappy_bird, name='flappy_bird_feed'),
    path('maths_equation', views.maths_equation, name='maths_equation'),
    path('maths_equation_feed', views.stream_maths_equation, name='maths_equation_feed'),
    path('hand_gesture', views.hand_gesture, name='hand_gesture'),
    path('hand_gesture_feed', views.stream_hand_gesture, name='hand_gesture_feed'),
]