from django.shortcuts import render, redirect
from django.http import HttpResponse, StreamingHttpResponse
from django.template import loader

from instagram_filters.flappy_bird import flappy_gen
from instagram_filters.hand_gesture import hand_gen


def home(request):
    web = redirect('main:main')
    if request.COOKIES.get('igstat'):
        web.set_cookie('igstat', 0)
    return web


def flappy_bird(request):
    template = loader.get_template('flappy_bird.html')
    return HttpResponse(template.render({}, request))


def stream_flappy_bird(request):
    return StreamingHttpResponse(flappy_gen(), content_type='multipart/x-mixed-replace; boundary=frame')


def maths_equation(request):
    template = loader.get_template('maths_equation.html')
    return HttpResponse(template.render({}, request))


def hand_gesture(request):
    template = loader.get_template('hand_gesture.html')
    return HttpResponse(template.render({}, request))


def stream_hand_gesture(request):
    return StreamingHttpResponse(hand_gen(), content_type='multipart/x-mixed-replace; boundary=frame')