from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.template import loader
from django.conf import settings


def home(request):
    web = redirect('main:main')
    web.set_cookie('rightshift', 2)

    return web


def model(request):
    if not os.path.exists(str(settings.BASE_DIR / 'human_emotions/Model/HumanEmotions_Model.h5')):
        web = redirect('main:main')
        web.set_cookie('model', 1)
        web.set_cookie('rightshift', 2)

        return web

    template = loader.get_template('he.html')
    return HttpResponse(template.render({}, request))