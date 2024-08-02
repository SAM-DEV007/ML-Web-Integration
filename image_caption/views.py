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
    models = 'weights', 'bias.pkl', 'tokenizer.pkl', 'mobilenet_v3_large_weights.h5', 'weights/checkpoint', 'weights/model.tf.data-00000-of-00001', 'weights/model.tf.index'

    for i in models:
        if not os.path.exists(str(settings.BASE_DIR / f'image_caption/Model/{i}')):
            web = redirect('main:main')
            web.set_cookie('model', 1)
            web.set_cookie('rightshift', 3)

            return web

    template = loader.get_template('image_caption.html')
    return HttpResponse(template.render({}, request))