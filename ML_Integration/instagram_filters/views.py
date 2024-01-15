from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.template import loader


def home(request):
    return redirect('main:main')


def flappy_bird(request):
    template = loader.get_template('flappy_bird.html')
    return HttpResponse(template.render({}, request))


def maths_equation(request):
    template = loader.get_template('maths_equation.html')
    return HttpResponse(template.render({}, request))


def hand_gesture(request):
    template = loader.get_template('hand_gesture.html')
    return HttpResponse(template.render({}, request))