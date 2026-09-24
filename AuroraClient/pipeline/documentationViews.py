"""Thin DRF APIViews for the Aurora living documentation system."""

import mimetypes
import os
import re

from typing import Optional

from django.http import FileResponse, Http404
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from .documentationManager import DocumentationManager, DOCS_DIR


def _manager():
    return DocumentationManager()


_FIGURE_NAME_RE = re.compile(r"^docs_[a-z0-9_]+\.(png|jpe?g|webp|gif)$", re.IGNORECASE)

# Preferred: BHTools permanent static publish (survives deleting figure_generation/).
# Installed apps: the copy synced into AuroraClient/static_media/docs-figures/.
# Fallback: unpublished renders in documentation/figure_generation/_work/staging/.
_REPO_ROOT = os.path.abspath(os.path.join(DOCS_DIR, "..", "..", "..", ".."))
_STATIC_FIGURES = os.path.join(
    _REPO_ROOT, "BHTools", "frontend", "static", "docs-figures"
)
_PACKAGED_FIGURES = os.path.abspath(
    os.path.join(DOCS_DIR, "..", "..", "static_media", "docs-figures")
)
_STAGING_FIGURES = os.path.join(DOCS_DIR, "figure_generation", "_work", "staging")


def _resolve_figure_path(name: str) -> Optional[str]:
    for root in (_STATIC_FIGURES, _PACKAGED_FIGURES, _STAGING_FIGURES):
        path = os.path.join(root, name)
        if os.path.isfile(path):
            return path
    return None


class DocsIndexView(APIView):
    """GET /docs/index/ — lightweight nav/search tree."""

    def get(self, request, *args, **kwargs):
        try:
            return Response(_manager().get_index(), status=status.HTTP_200_OK)
        except FileNotFoundError as exc:
            return Response({"error": str(exc)}, status=status.HTTP_404_NOT_FOUND)
        except Exception as exc:  # noqa: BLE001
            return Response({"error": str(exc)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class DocsEntryView(APIView):
    """GET /docs/entry/?id=<entry_id> — full entry prose (no live source body)."""

    def get(self, request, *args, **kwargs):
        entry_id = request.query_params.get("id")
        if not entry_id:
            return Response({"error": "Missing required query param: id"}, status=status.HTTP_400_BAD_REQUEST)
        try:
            return Response(_manager().get_entry(entry_id), status=status.HTTP_200_OK)
        except KeyError as exc:
            return Response({"error": str(exc)}, status=status.HTTP_404_NOT_FOUND)
        except Exception as exc:  # noqa: BLE001
            return Response({"error": str(exc)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class DocsCheckChangesView(APIView):
    """GET /docs/check-changes/ — stale-entry report for tooling / LLM agents."""

    def get(self, request, *args, **kwargs):
        try:
            stale = _manager().check_for_changes()
            return Response(
                {"stale_count": len(stale), "stale": stale},
                status=status.HTTP_200_OK,
            )
        except Exception as exc:  # noqa: BLE001
            return Response({"error": str(exc)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class DocsCorpusView(APIView):
    """GET /docs/corpus/ — full machine-readable corpus (methods + tutorial) from docs_index.json."""

    def get(self, request, *args, **kwargs):
        try:
            response = Response(_manager().get_corpus(), status=status.HTTP_200_OK)
            response["Cache-Control"] = "public, max-age=300"
            return response
        except FileNotFoundError as exc:
            return Response({"error": str(exc)}, status=status.HTTP_404_NOT_FOUND)
        except Exception as exc:  # noqa: BLE001
            return Response({"error": str(exc)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class DocsFigureView(APIView):
    """GET /docs/figure/<filename> — optional fallback for docs figure assets.

    Preferred public path is ``/static/docs-figures/<filename>`` on the website.
    This view serves the same bytes from the permanent static dir (or staging).
    """

    authentication_classes = []
    permission_classes = []

    def get(self, request, filename, *args, **kwargs):
        name = os.path.basename(filename or "")
        if not _FIGURE_NAME_RE.match(name):
            raise Http404("Invalid figure name")
        path = _resolve_figure_path(name)
        if not path:
            raise Http404("Figure not found")
        content_type, _ = mimetypes.guess_type(path)
        response = FileResponse(open(path, "rb"), content_type=content_type or "application/octet-stream")
        response["Cache-Control"] = "public, max-age=3600"
        return response
