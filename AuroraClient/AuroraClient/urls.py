"""
URL configuration for AuroraClient project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.contrib import admin
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from pipeline.views import *
from pipeline.elasticNccBackfill import BackfillElasticNccView  # TEMPORARY — delete with elasticNccBackfill.py
from pipeline.edgeTools import EdgeDetectionView, EdgeDetectionSlicePreviewView
from pipeline.meshBasedTools import ApplyMeshCleanupView, ApplyMeshCropView, ApplyMeshRotationView, ApplyMeshScaleView, ApplyMeshShellView, ApplyMeshSliceView, ApplyMeshSnapToMeshView, ApplyMeshWeldView, BatchWeldMeshView, MeshElasticRegistrationView
from pipeline.aiFineTuning.api import (
    FineTuneCancelView,
    FineTuneDiscoverView,
    FineTuneGpuView,
    FineTuneModelDetailView,
    FineTuneModelExportView,
    FineTuneModelFigureView,
    FineTuneModelListView,
    FineTuneModelPreviewFileView,
    FineTuneModelPreviewView,
    FineTuneRegisterView,
    FineTuneRunPreviewFileView,
    FineTuneRunPreviewView,
    FineTuneRunStatusView,
    FineTuneStartView,
    FineTuneValidateView,
)


from pipeline.spaViews import SpaIndexView, SpaAssetView
from pipeline.staticMediaViews import StaticMediaView
from pipeline.aiModels.foundation_views import (
    FoundationModelCancelView,
    FoundationModelDownloadView,
    FoundationModelStatusView,
)


urlpatterns = [
    path('admin/', admin.site.urls),
    path('upload/', FileUploadView.as_view(), name='upload_file'),
    path('list-directories/', ListDirectoriesView.as_view(), name='list_directories'),
    path('quick-access-paths/', QuickAccessPathsView.as_view(), name='quick_access_paths'),
    path('create-folder/', CreateFolderView.as_view(), name='create_folder'), 
    path('copy-folder/', FileCopyView.as_view(), name='create_folder'), 
    path('get-data-size-directory/', getDataSizeDirectoryView.as_view(), name='get_data_size_directory'),
    path('get-grid-view/', GetGridView.as_view(), name='get_grid_view'),
    path('projection-png/', ProjectionPngView.as_view(), name='projection_png'),
    path('ensure-quick-grid-projections/', EnsureQuickGridProjectionPngsView.as_view(), name='ensure_quick_grid_projections'),
    path('extract-scans-preflight/', ExtractScansPreflightView.as_view(), name='extract_scans_preflight'),
    path('extract-scans/', ExtractScansView.as_view(), name='extract_scans'),
    path('extract-json/', ExtractJsonView.as_view(), name='extract_json'),
    path('backfill-elastic-ncc/', BackfillElasticNccView.as_view(), name='backfill_elastic_ncc'),  # TEMPORARY
    path('display-nifti/', DisplayNiftiView.as_view(), name='display_nifti'),
    path('nifti-header-info/', NiftiHeaderInfoView.as_view(), name='nifti_header_info'),
    path('elastic-slice-ncc-heatmap/', ElasticSliceNccHeatmapView.as_view(), name='elastic_slice_ncc_heatmap'),
    path('marchingcubes/', MarchingCubesView.as_view(), name='marchingcubes'),
    path('edge-detection/', EdgeDetectionView.as_view(), name='edge_detection'),
    path('edge-detection-preview/', EdgeDetectionSlicePreviewView.as_view(), name='edge_detection_preview'),
    path('overwrite-threshold/', OverwriteThresholdView.as_view(), name='overwrite_threshold'),
    path('overwrite-voxel-size/', OverwriteVoxelSizeView.as_view(), name='overwrite_voxel_size'),
    path('get-label-names/', GetLabelNamesView.as_view(), name='get_label_names'),
    path('save-label-names/', SaveLabelNamesView.as_view(), name='save_label_names'),
    path('apply-rotation/', ApplyRotationView.as_view(), name='apply_rotation'),
    path('apply-crop/', ApplyCropView.as_view(), name='apply_crop'),
    path('apply-mesh-rotation/', ApplyMeshRotationView.as_view(), name='apply_mesh_rotation'),
    path('apply-mesh-scale/', ApplyMeshScaleView.as_view(), name='apply_mesh_scale'),
    path('apply-mesh-crop/', ApplyMeshCropView.as_view(), name='apply_mesh_crop'),
    path('apply-mesh-slice/', ApplyMeshSliceView.as_view(), name='apply_mesh_slice'),
    path('apply-mesh-cleanup/', ApplyMeshCleanupView.as_view(), name='apply_mesh_cleanup'),
    path('apply-mesh-weld/', ApplyMeshWeldView.as_view(), name='apply_mesh_weld'),
    path('batch-weld-mesh/', BatchWeldMeshView.as_view(), name='batch_weld_mesh'),
    path('apply-mesh-shell/', ApplyMeshShellView.as_view(), name='apply_mesh_shell'),
    path('apply-mesh-snap-to-mesh/', ApplyMeshSnapToMeshView.as_view(), name='apply_mesh_snap_to_mesh'),
    path('apply-crop-to-all-scans/', ApplyCropToAllScansView.as_view(), name='apply_crop_to_all_scans'),
    path('remove-background/', RemoveBackgroundView.as_view(), name='remove_background'),
    path('cleanup-mesh/', CleanupMeshView.as_view(), name='cleanup_mesh'),
    path('apply-cleanup-removal-mask/', ApplyCleanupRemovalMaskView.as_view(), name='apply_cleanup_removal_mask'),
    path('random-landmarking/', RandomLandmarkingView.as_view(), name='random_landmarking'),
    path('get-landmarks/', GetLandmarksView.as_view(), name='get_landmarks'),
    path('get-average-snapped-distances/', GetAverageSnappedDistancesView.as_view(), name='get_average_snapped_distances'),
    path('cohort-landmark-snap-distances/', CohortLandmarkSnapDistancesView.as_view(), name='cohort_landmark_snap_distances'),
    path('get-landmark-histogram/', GetLandmarkHistogramView.as_view(), name='get_landmark_histogram'),
    path('align-to-reference/', AlignToReferenceView.as_view(), name='align_to_reference'),
    path('delete-edit/', DeleteEditView.as_view(), name='delete_edit'),
    path('elastic-registration/', ElasticRegistrationView.as_view(), name='elastic_registration'),
    path('mesh-elastic-registration/', MeshElasticRegistrationView.as_view(), name='mesh_elastic_registration'),
    path('create-atlas/', CreateAtlasView.as_view(), name='create_atlas'),
    path('physically-accurate-mesh/', PhysicallyAccurateMeshView.as_view(), name='physically_accurate_mesh'),
    path('smart-landmarking/', SmartLandmarkingView.as_view(), name='smart_landmarking'),
    path('upload-landmarks/', UploadLandmarksView.as_view(), name='upload_landmarks'),
    path('transfer-landmarks/', TransferLandmarksView.as_view(), name='transfer_landmarks'),
    path('snap-to-mesh/', SnapToMeshView.as_view(), name='snap_to_mesh'),
    path('homogenize-background/', HomogenizeBackgroundView.as_view(), name='homogenize_background'),
    path('match-histogram/', MatchHistogramView.as_view(), name='match_histogram'),
    path('apply-threshold/', ApplyThresholdView.as_view(), name='apply_threshold'),
    path('batch-cleanup-mesh/', BatchCleanupMeshView.as_view(), name='batch_cleanup_mesh'),
    path('n4-bias-correction/', N4BiasCorrectionView.as_view(), name='n4_bias_correction'),
    path('n3-bias-correction/', N4BiasCorrectionView.as_view(), name='n3_bias_correction'),  # legacy alias
    path('denoise-all-scans/', DenoiseAllScansView.as_view(), name='denoise_all_scans'),
    path('restore-all-scans/', RestoreAllScansView.as_view(), name='restore_all_scans'),
    path('memory-monitor/', MemoryMonitorView.as_view(), name='memory_monitor'),
    path('save-guidepoints/', SaveGuidepointsView.as_view(), name='save_guidepoints'),
    path('load-guidepoints/', LoadGuidepointsView.as_view(), name='load_guidepoints'),
    path('load-landmark-guidepoints/', LoadLandmarkGuidepointsView.as_view(), name='load_landmark_guidepoints'),
    path('save-landmark-guidepoints/', SaveLandmarkGuidepointsView.as_view(), name='save_landmark_guidepoints'),
    path('registration-heatmap/', RegistrationHeatmapView.as_view(), name='registration_heatmap'),
    path('registration-subject-data/', RegistrationSubjectDataView.as_view(), name='registration_subject_data'),
    path('upload-mask/', UploadMaskView.as_view(), name='upload_mask'),
    path('download-mask/', DownloadMaskView.as_view(), name='download_mask'),
    path('save-mask/', SaveMaskView.as_view(), name='save_mask'),
    path('load-mask/', LoadMaskView.as_view(), name='load_mask'),
    path('delete-mask/', DeleteMaskView.as_view(), name='delete_mask'),
    path('save-background-values/', SaveBackgroundValuesView.as_view(), name='save_background_values'),
    path('save-threshold-values/', SaveThresholdValuesView.as_view(), name='save_threshold_values'),
    path('get-semi-landmarks/', GetSemiLandmarksView.as_view(), name='get_semi_landmarks'),
    path('rebalance-landmarks-along-path/', RebalanceLandmarksAlongPathView.as_view(), name='rebalance_landmarks_along_path'),
    path('report-bad-file/', ReportBadFileView.as_view(), name='report_bad_file'),
    path('flag-subjects/', FlagSubjectsView.as_view(), name='flag_subjects'),
    path('link-scans/', LinkScansView.as_view(), name='link_scans'),
    path('copy-mask-to-reference/', CopyMaskToReferenceView.as_view(), name='copy_mask_to_reference'),
    path('get-ai-models/', GetAIModelsView.as_view(), name='get_ai_models'),
    path('ai-models/foundation/status/', FoundationModelStatusView.as_view(), name='ai_models_foundation_status'),
    path('ai-models/foundation/download/', FoundationModelDownloadView.as_view(), name='ai_models_foundation_download'),
    path('ai-models/foundation/cancel/', FoundationModelCancelView.as_view(), name='ai_models_foundation_cancel'),
    path('run-ai-segmentation/', RunAISegmentationView.as_view(), name='run_ai_segmentation'),
    path('ai-finetune/discover/', FineTuneDiscoverView.as_view(), name='ai_finetune_discover'),
    path('ai-finetune/validate/', FineTuneValidateView.as_view(), name='ai_finetune_validate'),
    path('ai-finetune/gpu/', FineTuneGpuView.as_view(), name='ai_finetune_gpu'),
    path('ai-finetune/start/', FineTuneStartView.as_view(), name='ai_finetune_start'),
    path('ai-finetune/runs/<str:run_id>/preview-file/', FineTuneRunPreviewFileView.as_view(), name='ai_finetune_run_preview_file'),
    path('ai-finetune/runs/<str:run_id>/preview/', FineTuneRunPreviewView.as_view(), name='ai_finetune_run_preview'),
    path('ai-finetune/runs/<str:run_id>/', FineTuneRunStatusView.as_view(), name='ai_finetune_run_status'),
    path('ai-finetune/runs/<str:run_id>/cancel/', FineTuneCancelView.as_view(), name='ai_finetune_cancel'),
    path('ai-finetune/runs/<str:run_id>/register/', FineTuneRegisterView.as_view(), name='ai_finetune_register'),
    path('ai-finetune/models/', FineTuneModelListView.as_view(), name='ai_finetune_models'),
    path('ai-finetune/models/<str:model_id>/export/', FineTuneModelExportView.as_view(), name='ai_finetune_model_export'),
    path('ai-finetune/models/<str:model_id>/figures/<path:relative>', FineTuneModelFigureView.as_view(), name='ai_finetune_model_figure'),
    path('ai-finetune/models/<str:model_id>/preview-file/', FineTuneModelPreviewFileView.as_view(), name='ai_finetune_model_preview_file'),
    path('ai-finetune/models/<str:model_id>/preview/', FineTuneModelPreviewView.as_view(), name='ai_finetune_model_preview'),
    path('ai-finetune/models/<str:model_id>/', FineTuneModelDetailView.as_view(), name='ai_finetune_model_detail'),
    path('get-project-settings/', GetProjectSettingsView.as_view(), name='get_project_settings'),
    path('save-project-settings/', SaveProjectSettingsView.as_view(), name='save_project_settings'),
    path('save-scientific-report/', SaveScientificReportView.as_view(), name='save_scientific_report'),
    path('load-scientific-report/', LoadScientificReportView.as_view(), name='load_scientific_report'),
    path('load-report-figure-mesh-cache/', LoadReportFigureMeshCacheView.as_view(), name='load_report_figure_mesh_cache'),
    path('save-report-figure-mesh-cache/', SaveReportFigureMeshCacheView.as_view(), name='save_report_figure_mesh_cache'),
    path('delete-report-figure-mesh-cache/', DeleteReportFigureMeshCacheView.as_view(), name='delete_report_figure_mesh_cache'),
    path('delete-scientific-report/', DeleteScientificReportView.as_view(), name='delete_scientific_report'),
    path('get-report-environment/', GetReportEnvironmentView.as_view(), name='get_report_environment'),
    path('make-atlas-reference/', MakeAtlasReferenceView.as_view(), name='make_atlas_reference'),
    path('get-all-subjects-mesh/', GetAllSubjectsMeshView.as_view(), name='get_all_subjects_mesh'),
    path('get-version/', GetVersionView.as_view(), name='get_version'),
    path('preview-denoise/', PreviewDenoiseView.as_view(), name='preview_denoise'),
    path('propagate-mask/', PropagateMaskView.as_view(), name='propagate_mask'),
    path('get-rotation-matrix/', GetRotationMatrixView.as_view(), name='get_rotation_matrix'),
    path('export-landmarks/', ExportLandmarksView.as_view(), name='export_landmarks'),
    path('export-landmark-distances/', ExportLandmarkDistancesView.as_view(), name='export_landmark_distances'),
    path('export-landmarks-bundle/', ExportLandmarksBundleView.as_view(), name='export_landmarks_bundle'),
    path('export-landmarks-json/', ExportLandmarksJsonView.as_view(), name='export_landmarks_json'),
    path('list-mask-folders/', ListMaskFoldersView.as_view(), name='list_mask_folders'),
    path('apply-mask-to-image/', ApplyMaskToImageView.as_view(), name='apply_mask_to_image'),
    path('batch-apply-mask-to-scans/', BatchApplyMaskToScansView.as_view(), name='batch_apply_mask_to_scans'),
    path('mask-out-labels/', MaskOutLabelsView.as_view(), name='mask_out_labels'),
    path('batch-mask-out-scans/', BatchMaskOutScansView.as_view(), name='batch_mask_out_scans'),
    path('clear-intermediate-files/', ClearIntermediateFilesView.as_view(), name='clear_intermediate_files'),
    path('invert-alignment-landmarks/', InvertAlignmentLandmarksView.as_view(), name='invert_alignment_landmarks'),
    path('export-inverted-landmarks/', ExportInvertedLandmarksView.as_view(), name='export_inverted_landmarks'),
    path('docs/index/', DocsIndexView.as_view(), name='docs_index'),
    path('docs/entry/', DocsEntryView.as_view(), name='docs_entry'),
    path('docs/check-changes/', DocsCheckChangesView.as_view(), name='docs_check_changes'),
    path('docs/corpus/', DocsCorpusView.as_view(), name='docs_corpus'),
    path('docs/figure/<str:filename>', DocsFigureView.as_view(), name='docs_figure'),
    # Packaged React SPA — MUST stay after all API routes so upload/, get-version/, docs/, etc.
    # are never swallowed. /local_aurora/ (not /) keeps the API root unambiguous for same-origin clients.
    # Runtime media (textures/GLBs/icons) — same-origin for Babylon; synced into static_media/.
    path('static/<path:asset_path>', StaticMediaView.as_view(), name='static_media'),
    path('local_aurora/', SpaIndexView.as_view(), name='spa_index'),
    path('local_aurora/<path:asset_path>', SpaAssetView.as_view(), name='spa_asset'),
]
