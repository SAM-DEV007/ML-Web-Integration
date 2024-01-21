from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.template import loader
from django.views.decorators.csrf import csrf_protect
from django.conf import settings

import os


def home(request):
    web = redirect('main:main')
    web.set_cookie('rightshift', 1)

    return web


@csrf_protect
def model(request):
    if request.method == 'POST':
        pass
    
    if not os.path.exists(str(settings.BASE_DIR / 'word_classifier/Model/WordClassifier_Model.h5')):
        web = redirect('main:main')
        web.set_cookie('wcmodel', 1)

        return web

    template = loader.get_template('wc.html')
    return HttpResponse(template.render({}, request))
