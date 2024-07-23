from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.template import loader
from django.conf import settings

import os


def home(request):
    web = redirect('main:main')
    web.set_cookie('rightshift', 3)

    return web


def model(request):
    if not os.path.exists(str(settings.BASE_DIR / '..model_path..')):
        web = redirect('main:main')
        web.set_cookie('model', 1)
        web.set_cookie('rightshift', 3)

        return web

    template = loader.get_template('image_caption.html')
    return HttpResponse(template.render({}, request))