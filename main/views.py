from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader

import json

from instagram_filters import storage


def home(request, context = {'igstat': 1, 'rightshift': 0}):
    template = loader.get_template('home.html')

    storage.IMG_PATH = None

    if request.COOKIES.get('igstat'):
        context = {'igstat': request.COOKIES['igstat'], 'rightshift': request.COOKIES['rightshift']}

    web = HttpResponse(template.render(context, request))
    web.set_cookie('igstat', 1)
    web.set_cookie('rightshift', 0)

    return web