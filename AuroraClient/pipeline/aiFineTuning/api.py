"""Thin DRF endpoints for the fine-tuning workflow."""

from __future__ import annotations

from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from .discover import discover_project
from .library import (
    LibraryError,
    delete_registered_model,
    export_registered_zip,
    figure_path,
    import_registered_zip,
    list_registered_library,
    model_detail,
)
from .runner import cancel_run, get_run, register_run, start_run
from .gpu_monitor import gpu_snapshot
from .validate import validate_experiment
from ..aiModels.foundation_models import ModelMissingError


class FineTuneDiscoverView(APIView):
    def post(self, request):
        directory = request.data.get("directory")
        if not directory:
            return Response({"error": "directory is required"}, status=status.HTTP_400_BAD_REQUEST)
        try:
            payload = discover_project(directory)
        except FileNotFoundError as exc:
            return Response({"error": str(exc)}, status=status.HTTP_404_NOT_FOUND)
        except Exception as exc:
            return Response({"error": str(exc)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        return Response(payload)


class FineTuneValidateView(APIView):
    def post(self, request):
        report = validate_experiment(dict(request.data))
        http_status = status.HTTP_200_OK if report.ok else status.HTTP_400_BAD_REQUEST
        return Response(report.as_dict(), status=http_status)


class FineTuneGpuView(APIView):
    def get(self, request):
        return Response(gpu_snapshot())


class FineTuneStartView(APIView):
    def post(self, request):
        try:
            result = start_run(dict(request.data), dry_run=bool(request.data.get("dry_run")))
        except ModelMissingError as exc:
            return Response(
                {"code": "model_missing", "model": exc.model_id, "error": str(exc)},
                status=status.HTTP_409_CONFLICT,
            )
        except Exception as exc:
            return Response({"error": str(exc)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        if not result.get("ok"):
            return Response(result, status=status.HTTP_400_BAD_REQUEST)
        return Response(result, status=status.HTTP_202_ACCEPTED)


class FineTuneRunStatusView(APIView):
    def get(self, request, run_id):
        payload = get_run(run_id)
        if payload is None:
            return Response({"error": "Training run not found"}, status=status.HTTP_404_NOT_FOUND)
        return Response(payload)


class FineTuneCancelView(APIView):
    def post(self, request, run_id):
        try:
            return Response(cancel_run(run_id))
        except Exception as exc:
            return Response({"error": str(exc)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class FineTuneRegisterView(APIView):
    def post(self, request, run_id):
        try:
            result = register_run(run_id, display_name=request.data.get("display_name"))
        except FileNotFoundError as exc:
            return Response({"error": str(exc)}, status=status.HTTP_404_NOT_FOUND)
        except ValueError as exc:
            return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as exc:
            return Response({"error": str(exc)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        return Response(result)


class FineTuneModelListView(APIView):
    def get(self, request):
        return Response({"models": list_registered_library()})

    def post(self, request):
        upload = request.FILES.get("file") or request.FILES.get("zip")
        if upload is None:
            return Response({"error": "Upload a zip as file"}, status=status.HTTP_400_BAD_REQUEST)
        try:
            result = import_registered_zip(upload)
        except LibraryError as exc:
            return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as exc:
            return Response({"error": str(exc)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        return Response(result, status=status.HTTP_201_CREATED)


class FineTuneModelDetailView(APIView):
    def get(self, request, model_id):
        try:
            return Response(model_detail(model_id))
        except LibraryError as exc:
            return Response({"error": str(exc)}, status=status.HTTP_404_NOT_FOUND)

    def delete(self, request, model_id):
        try:
            return Response(delete_registered_model(model_id))
        except LibraryError as exc:
            return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)


class FineTuneModelExportView(APIView):
    def get(self, request, model_id):
        from django.http import FileResponse

        try:
            buffer, filename = export_registered_zip(model_id)
        except LibraryError as exc:
            return Response({"error": str(exc)}, status=status.HTTP_404_NOT_FOUND)
        response = FileResponse(buffer, as_attachment=True, filename=filename)
        response["Content-Type"] = "application/zip"
        return response


class FineTuneModelFigureView(APIView):
    def get(self, request, model_id, relative):
        from django.http import FileResponse

        try:
            path = figure_path(model_id, relative)
        except LibraryError as exc:
            return Response({"error": str(exc)}, status=status.HTTP_404_NOT_FOUND)
        return FileResponse(open(path, "rb"), filename=path.name)


def _preview_file_response(root, subject, kind, epoch):
    from django.http import FileResponse
    from .preview_store import resolve_preview_file

    path = resolve_preview_file(root, subject_key_value=subject, kind=kind, epoch=epoch)
    response = FileResponse(open(path, "rb"), filename=path.name)
    response["Content-Type"] = "application/gzip"
    return response


class FineTuneRunPreviewView(APIView):
    def get(self, request, run_id):
        from .preview_store import read_manifest, run_preview_dir
        from .runner import get_run

        if get_run(run_id) is None:
            return Response({"error": "Training run not found"}, status=status.HTTP_404_NOT_FOUND)
        return Response(read_manifest(run_preview_dir(run_id)))


class FineTuneRunPreviewFileView(APIView):
    def get(self, request, run_id):
        from .preview_store import run_preview_dir
        from .runner import get_run

        if get_run(run_id) is None:
            return Response({"error": "Training run not found"}, status=status.HTTP_404_NOT_FOUND)
        kind = request.query_params.get("kind") or "image"
        subject = request.query_params.get("subject") or ""
        epoch = request.query_params.get("epoch")
        try:
            epoch_n = int(epoch) if epoch not in (None, "") else None
            return _preview_file_response(run_preview_dir(run_id), subject, kind, epoch_n)
        except (FileNotFoundError, ValueError) as exc:
            return Response({"error": str(exc)}, status=status.HTTP_404_NOT_FOUND)


class FineTuneModelPreviewView(APIView):
    def get(self, request, model_id):
        from ..aiModels.paths import registered_model_dir
        from .preview_store import read_manifest

        try:
            detail = model_detail(model_id)
        except LibraryError as exc:
            return Response({"error": str(exc)}, status=status.HTTP_404_NOT_FOUND)
        return Response(detail.get("preview") or read_manifest(registered_model_dir(model_id)))


class FineTuneModelPreviewFileView(APIView):
    def get(self, request, model_id):
        from ..aiModels.paths import registered_model_dir

        try:
            model_detail(model_id)
        except LibraryError as exc:
            return Response({"error": str(exc)}, status=status.HTTP_404_NOT_FOUND)
        kind = request.query_params.get("kind") or "image"
        subject = request.query_params.get("subject") or ""
        epoch = request.query_params.get("epoch")
        try:
            epoch_n = int(epoch) if epoch not in (None, "") else None
            return _preview_file_response(registered_model_dir(model_id), subject, kind, epoch_n)
        except (FileNotFoundError, ValueError) as exc:
            return Response({"error": str(exc)}, status=status.HTTP_404_NOT_FOUND)
