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
from channels.security.websocket import AllowedHostsOriginValidator
from AuroraClient import routing
from AuroraClient import compactAccessLog
from AuroraClient import engineConfig


os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'AuroraClient.settings')

# Replace daphne's one-line-per-request access log with a grouped, error-first one
# (AURORA_ACCESS_LOG=full restores the original). See compactAccessLog.py.
compactAccessLog.install()

django_asgi_app = get_asgi_application()

# Keep recent terminal lines for the read-only viewer (/terminal/). See engineTerminal.py.
from pipeline import engineTerminal  # noqa: E402
engineTerminal.install()
# Django logs every 4xx as a warning ("Conflict: /engine-jobs/"). 409 here is the shared
# computer's "in use by someone else" answer, expected while another window holds it.
import logging  # noqa: E402
logging.getLogger("django.request").addFilter(lambda record: getattr(record, "status_code", None) != 409)

# Imported after Django setup: engineAuth pulls in DRF.
from pipeline.engineAuth import EngineSessionWebsocketMiddleware  # noqa: E402
from pipeline import networkMode  # noqa: E402

# Paired computers report status to the lab and fetch their group every few minutes
# (no-op until "aurora-tools pair" has been run). Never on a request path.
networkMode.start_check_in_thread()


def _observe_progress_messages():
    """Let engineActivity see every progress_group message (job label and percentage for
    the lab's "In use by" line). Wraps the in-memory channel layer's group_send once;
    every tool already sends progress through it, so no tool code changes."""
    from channels.layers import get_channel_layer
    from pipeline import engineActivity

    layer = get_channel_layer()
    if layer is None or getattr(layer, "_aurora_activity_wrapped", False):
        return
    original_group_send = layer.group_send

    async def group_send(group, message):
        if group == "progress_group":
            engineActivity.observe_progress(message)
        return await original_group_send(group, message)

    layer.group_send = group_send
    layer._aurora_activity_wrapped = True


_observe_progress_messages()

# Job history: read it once now, so jobs cut off by the last shutdown show as interrupted.
from pipeline import engineJobs  # noqa: E402
engineJobs.load()


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
    print(f"    http://127.0.0.1:{engineConfig.engine_port()}/local_aurora/")
    print()
    if networkMode.network_ready():
        config = engineConfig.load_config()
        print("[🔒] Shared on the local network (HTTPS, self-signed certificate):")
        print(f"    https://{networkMode._lan_address() or '<this computer>'}:{config['network_port']}/local_aurora/")
        print("    Group members open it from hallgrimssonlab.ca/MainAurora/servers")
        print()
    print("[💡] We suggest opening the link in a Chrome browser to use the tools.")
    print("[⏳] First use may take some time to load.")
    print()
    print("="*70 + "\n")


# Print startup message once when ASGI app is loaded
threading.Thread(target=print_startup_message, daemon=True).start()


application = ProtocolTypeRouter({
    # Heartbeat bookkeeping for long requests (compact access log); transparent otherwise.
    "http": compactAccessLog.RequestHeartbeatMiddleware(django_asgi_app),
    # Browser WebSockets must come from an allowed Host origin (no cross-site socket
    # reads) and carry the engine session cookie set by /engine-auth/start/.
    "websocket": AllowedHostsOriginValidator(
        EngineSessionWebsocketMiddleware(
            AuthMiddlewareStack(
                URLRouter(
                    routing.websocket_urlpatterns
                )
            )
        )
    ),
})