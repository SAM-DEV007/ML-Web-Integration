from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.template import loader
from django.conf import settings

from . import caption

import os
import json


def home(request):
    web = redirect('main:main')
    web.set_cookie('rightshift', 3)

    return web


def model(request):
    models = 'weights', 'bias.pkl', 'tokenizer.pkl', 'weights/checkpoint', 'weights/model.tf.data-00000-of-00001', 'weights/model.tf.index'

    for i in models:
        if not os.path.exists(str(settings.BASE_DIR / f'image_caption/Model/{i}')):
            web = redirect('main:main')
            web.set_cookie('model', 1)
            web.set_cookie('rightshift', 3)

            return web
        
        if request.method == 'POST':
            npImg = request.FILES['image'].read()
            if npImg:
                result, image = caption.get_caption(npImg)
                data = {
                    'caption': result,
                    'image': image
                }
                return HttpResponse(json.dumps(data))

    template = loader.get_template('ic.html')
    return HttpResponse(template.render({}, request))