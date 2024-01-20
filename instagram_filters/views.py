from django.shortcuts import render, redirect
from django.http import HttpResponse, StreamingHttpResponse
from django.template import loader
from django.views.decorators.csrf import csrf_protect


def home(request):
    web = redirect('main:main')

    if request.COOKIES.get('igstat'):
        web.set_cookie('igstat', 0)
    return web


@csrf_protect
def flappy_bird(request):
    template = loader.get_template('flappy_bird.html')
    return HttpResponse(template.render({}, request))


@csrf_protect
def maths_equation(request):
    template = loader.get_template('maths_equation.html')
    return HttpResponse(template.render({}, request))


@csrf_protect
def hand_gesture(request):
    template = loader.get_template('hand_gesture.html')
    return HttpResponse(template.render({}, request))