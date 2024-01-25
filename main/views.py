from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader


def home(request):
    template = loader.get_template('home.html')

    context = {
        'igstat': request.COOKIES.get('igstat', 1), 
        'rightshift': request.COOKIES.get('rightshift', 0),
        'model': request.COOKIES.get('model', 0),
    }

    web = HttpResponse(template.render(context, request))
    web.set_cookie('igstat', 1)
    web.set_cookie('rightshift', 0)
    web.set_cookie('model', 0)

    return web