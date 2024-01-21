from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.template import loader
from django.views.decorators.csrf import csrf_protect


def home(request):
    web = redirect('main:main')

    if request.COOKIES.get('rightshift'):
        web.set_cookie('rightshift', 1)
    return web


@csrf_protect
def model(request):
    if request.method == 'POST':
        pass

    template = loader.get_template('wc.html')
    return HttpResponse(template.render({}, request))