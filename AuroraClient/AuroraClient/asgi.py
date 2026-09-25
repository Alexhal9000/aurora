"""
ASGI config for AuroraClient project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.0/howto/deployment/asgi/
"""

import os
import sys
import threading
import time
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
from AuroraClient import routing


os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'AuroraClient.settings')

django_asgi_app = get_asgi_application()


def _ensure_django_schema():
    """Create/update Django's own tables. Safe no-op when the schema is current."""
    from django.core.management import call_command
    call_command('migrate', interactive=False, verbosity=0)


_ensure_django_schema()


def print_startup_message():
    """Print professional startup message with welcome banner and clickable URL."""
    time.sleep(0.5)  # Brief delay to let server logs print first
    
    # Fallback for terminals without color support
    print("\n" + "="*70)
    print("Hallgrímsson's Lab: Aurora Tools".center(70))
    print("="*70)
    print()
    print("[✓] Server is running and ready to use")
    print()
    print("[📍] Lab shell (login / install / handoff):")
    print("    https://hallgrimssonlab.ca/MainAurora")
    print()
    print("[📍] Packaged UI (same origin as this engine, after handoff):")
    print("    http://127.0.0.1:8020/local_aurora/")
    print()
    print("[💡] We suggest opening the link in a Chrome browser to use the tools.")
    print("[⏳] First use may take some time to load.")
    print()
    print("="*70 + "\n")


# Print startup message once when ASGI app is loaded
threading.Thread(target=print_startup_message, daemon=True).start()


application = ProtocolTypeRouter({
    "http": django_asgi_app,
    "websocket": AuthMiddlewareStack(
        URLRouter(
            routing.websocket_urlpatterns
        )
    ),
})