from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.template import loader
from django.views.decorators.csrf import csrf_protect
from django.conf import settings

from word_classifier import word_classifier

import os


def home(request):
    web = redirect('main:main')
    web.set_cookie('rightshift', 1)

    return web


@csrf_protect
def model(request):
    if not os.path.exists(str(settings.BASE_DIR / 'word_classifier/Model/WordClassifier_Model.h5')):
        web = redirect('main:main')
        web.set_cookie('model', 1)
        web.set_cookie('rightshift', 1)

        return web

    if request.method == 'POST':
        return HttpResponse(word_classifier.predict(request.POST.get('sentence')))

    template = loader.get_template('wc.html')
    return HttpResponse(template.render({}, request))
