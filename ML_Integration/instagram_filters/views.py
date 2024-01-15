from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.template import loader


def home(request):
    web = redirect('main:main')
    if request.COOKIES.get('igstat'):
        web.set_cookie('igstat', 0)
    return web


def flappy_bird(request):
    template = loader.get_template('flappy_bird.html')
    return HttpResponse(template.render({}, request))


def maths_equation(request):
    template = loader.get_template('maths_equation.html')
    return HttpResponse(template.render({}, request))


def hand_gesture(request):
    template = loader.get_template('hand_gesture.html')
    return HttpResponse(template.render({}, request))