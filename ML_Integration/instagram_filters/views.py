from django.shortcuts import render, redirect
from django.http import HttpResponse, StreamingHttpResponse
from django.template import loader
from django.views.decorators.csrf import csrf_protect

from instagram_filters.flappy_bird import flappy_gen
from instagram_filters.hand_gesture import hand_gen
from instagram_filters.maths_equation import maths_gen

from instagram_filters import storage


def home(request):
    web = redirect('main:main')
    if request.COOKIES.get('igstat'):
        web.set_cookie('igstat', 0)
    return web


@csrf_protect
def flappy_bird(request):
    template = loader.get_template('flappy_bird.html')
    return HttpResponse(template.render({}, request))


def path_flappy(request):
    if request.method == 'POST':
        path = request.POST.get('webimg')
        storage.FLAPPY_PATH = path

        return HttpResponse('Path set')
    return HttpResponse('Path not set')


def stream_flappy_bird(request):
    return StreamingHttpResponse(flappy_gen(), content_type='multipart/x-mixed-replace; boundary=frame')


def maths_equation(request):
    template = loader.get_template('maths_equation.html')
    return HttpResponse(template.render({}, request))


def stream_maths_equation(request):
    return StreamingHttpResponse(maths_gen(), content_type='multipart/x-mixed-replace; boundary=frame')


def hand_gesture(request):
    template = loader.get_template('hand_gesture.html')
    return HttpResponse(template.render({}, request))


def stream_hand_gesture(request):
    return StreamingHttpResponse(hand_gen(), content_type='multipart/x-mixed-replace; boundary=frame')