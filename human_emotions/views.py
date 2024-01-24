from django.shortcuts import render, redirect


def home(request):
    web = redirect('main:main')
    web.set_cookie('rightshift', 2)

    return web