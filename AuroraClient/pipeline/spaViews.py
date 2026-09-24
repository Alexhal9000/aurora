"""Serve the packaged Aurora React SPA from AuroraClient/frontend_dist.

The lab HTTPS site cannot <script>-load these assets (mixed content). Users navigate
to http://127.0.0.1:8020/local_aurora/ after handoff; UI and engine APIs then share this origin.

Mirrors DocsFigureView: explicit FileResponse with path-traversal checks (DEBUG is False
in packaged Aurora, so django.contrib.staticfiles would not serve these automatically).
"""

from __future__ import annotations

import mimetypes
import re
from pathlib import Path

from django.conf import settings
from django.http import FileResponse, Http404
from rest_framework.views import APIView


def _spa_root() -> Path:
    # BASE_DIR is AuroraClient/ (parent of the AuroraClient settings package).
    return Path(settings.BASE_DIR) / "frontend_dist"


def _safe_file_under_spa(relative: str) -> Path | None:
    root = _spa_root().resolve()
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


# Webpack contenthash in the filename (e.g. 3119b968d8eecb678897.png). Stable names like
# main.js / 165.js must NOT be cached as immutable or a new package keeps serving old UI.
_FINGERPRINT_RE = re.compile(r"\.[a-f0-9]{8,}\.", re.IGNORECASE)


def _cache_control_for_spa_file(path: Path) -> str:
    suffix = path.suffix.lower()
    fingerprinted = bool(_FINGERPRINT_RE.search(path.name))
    if suffix == ".html" or (suffix in {".js", ".css"} and not fingerprinted):
        return "no-cache"
    if fingerprinted or suffix in {".woff", ".woff2", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"}:
        return "public, max-age=31536000, immutable"
    return "public, max-age=3600"


class SpaIndexView(APIView):
    """GET /local_aurora/ — packaged SPA shell (index.html)."""

    authentication_classes = []
    permission_classes = []

    def get(self, request, *args, **kwargs):
        index = _spa_root() / "index.html"
        if not index.is_file():
            raise Http404(
                "Packaged Aurora UI missing (frontend_dist/index.html). "
                "On the release machine run: npm run build:aurora-local"
            )
        response = FileResponse(open(index, "rb"), content_type="text/html; charset=utf-8")
        # Avoid sticky shell during dual-run / rapid rebuilds.
        response["Cache-Control"] = "no-cache"
        return response


class SpaAssetView(APIView):
    """GET /local_aurora/<path> — hashed JS/CSS/assets, or index.html for client-side routes.

    Real files under frontend_dist win; unknown paths fall back to index.html so
    React Router can handle in-app navigation without colliding with API urls
    (those are registered before /local_aurora/ in urlpatterns).
    """

    authentication_classes = []
    permission_classes = []

    def get(self, request, asset_path, *args, **kwargs):
        # Normalize and reject empty / absolute tricks.
        rel = (asset_path or "").lstrip("/").replace("\\", "/")
        if not rel or ".." in rel.split("/"):
            raise Http404("Invalid asset path")

        path = _safe_file_under_spa(rel)
        if path is not None:
            content_type, _ = mimetypes.guess_type(str(path))
            response = FileResponse(
                open(path, "rb"),
                content_type=content_type or "application/octet-stream",
            )
            response["Cache-Control"] = _cache_control_for_spa_file(path)
            return response

        # SPA fallback (e.g. /local_aurora/some-client-route)
        index = _spa_root() / "index.html"
        if index.is_file():
            response = FileResponse(open(index, "rb"), content_type="text/html; charset=utf-8")
            response["Cache-Control"] = "no-cache"
            return response

        raise Http404("Packaged Aurora UI not found")
