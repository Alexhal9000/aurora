"""Local Aurora endpoints for foundation-model status and download."""

from __future__ import annotations

from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from .foundation_download import (
    DownloadError,
    cancel_download,
    start_download,
    status_payload,
)
from .foundation_models import enabled_models, get_spec
from pipeline.engineAuth import is_beta_session


def _beta(request) -> bool:
    """Beta-only models (DINOv3) are listed and downloadable for beta testers only."""
    return is_beta_session(getattr(request, "aurora_session", None))


class FoundationModelStatusView(APIView):
    def get(self, request):
        return Response(status_payload(include_beta=_beta(request)), status=status.HTTP_200_OK)


class FoundationModelDownloadView(APIView):
    def post(self, request):
        body = request.data or {}
        if not body.get("accepted_terms"):
            return Response(
                {"error": "You must accept the terms of use before downloading."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        raw_ids = body.get("model_ids") or []
        if isinstance(raw_ids, str):
            raw_ids = [raw_ids]
        model_ids = [str(item) for item in raw_ids if str(item)]
        if not model_ids:
            model_ids = [item["id"] for item in enabled_models(include_beta=_beta(request))]
        for model_id in model_ids:
            if get_spec(model_id) is None:
                return Response(
                    {"error": f"Unknown model {model_id!r}."},
                    status=status.HTTP_400_BAD_REQUEST,
                )
        try:
            start_download(
                model_ids,
                token=str(body.get("token") or ""),
                lab_origin=str(body.get("lab_origin") or ""),
                user_name=str(body.get("user_name") or ""),
                include_beta=_beta(request),
            )
        except DownloadError as exc:
            return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        return Response(status_payload(include_beta=_beta(request)), status=status.HTTP_202_ACCEPTED)


class FoundationModelCancelView(APIView):
    def post(self, request):
        cancel_download()
        return Response(status_payload(include_beta=_beta(request)), status=status.HTTP_200_OK)
