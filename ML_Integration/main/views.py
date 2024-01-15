from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader


def home(request, context={'igstat': 1, 'rightshift': 0}):
    template = loader.get_template('home.html')
    return HttpResponse(template.render(context, request))