"""Serve Aurora runtime media from AuroraClient/static_media at /static/...

Packaged SPA on :8020 must load Babylon textures / GLBs / icons same-origin.
These files are synced from BHTools/frontend/static via
deployment/shared/sync_aurora_static_media.sh (also run by deploy.sh).

Mirrors spaViews path-traversal checks (DEBUG is False in packaged Aurora).
"""

from __future__ import annotations

import mimetypes
from pathlib import Path

from django.conf import settings
from django.http import FileResponse, Http404
from rest_framework.views import APIView


def _static_media_root() -> Path:
    # BASE_DIR is AuroraClient/ (parent of the AuroraClient settings package).
    return Path(settings.BASE_DIR) / "static_media"


def _safe_file_under_static_media(relative: str) -> Path | None:
    root = _static_media_root().resolve()
    if not relative:
        return None
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    if candidate.is_file():
        return candidate
    return None


class StaticMediaView(APIView):
    """GET /static/<path> — Aurora-bundled runtime media (textures, GLBs, icons)."""

    authentication_classes = []
    permission_classes = []

    def get(self, request, asset_path, *args, **kwargs):
        rel = (asset_path or "").lstrip("/").replace("\\", "/")
        if not rel or ".." in rel.split("/"):
            raise Http404("Invalid static path")

        path = _safe_file_under_static_media(rel)
        if path is None:
            raise Http404("Static media not found")

        content_type, _ = mimetypes.guess_type(str(path))
        # Ensure GLB is treated as a binary model for Babylon XHR.
        if path.suffix.lower() == ".glb":
            content_type = "model/gltf-binary"
        response = FileResponse(
            open(path, "rb"),
            content_type=content_type or "application/octet-stream",
        )
        response["Cache-Control"] = "public, max-age=86400"
        return response
