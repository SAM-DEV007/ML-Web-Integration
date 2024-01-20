"""
ASGI config for ML_Integration project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.0/howto/deployment/asgi/
"""

import os

from channels.auth import AuthMiddlewareStack
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.security.websocket import AllowedHostsOriginValidator
from django.core.asgi import get_asgi_application
from django.urls import path

from instagram_filters.flappy_bird import FlappyBird
from instagram_filters.maths_equation import MathsEquation
from instagram_filters.hand_gesture import HandGesture

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ML_Integration.settings')

application = get_asgi_application()
django_asgi_app = application

application = ProtocolTypeRouter({
    # Django's ASGI application to handle traditional HTTP requests
    "http": django_asgi_app,
    "https": django_asgi_app,

    # WebSocket chat handler
    "websocket": AllowedHostsOriginValidator(
        AuthMiddlewareStack(
            URLRouter([
                path("flappy_bird", FlappyBird.as_asgi()),
                path("maths_equation", MathsEquation.as_asgi()),
                path("hand_gesture", HandGesture.as_asgi()),
            ])
        )
    ),
})
