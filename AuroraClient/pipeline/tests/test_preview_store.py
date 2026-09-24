from django.test import SimpleTestCase

import numpy as np

from pipeline.aiFineTuning.preview_store import (
    assemble_clip_overlay,
    preview_target_shape,
    resize_volume_nearest,
    resolve_preview_file,
    save_epoch_prediction,
    scatter_clip_slab,
    subject_key,
    update_manifest,
)


class PreviewStoreTests(SimpleTestCase):
    def test_subject_key(self):
        self.assertEqual(subject_key({"subject": "SubA", "label_id": 2}), "SubA__label2")

    def test_preview_target_shape_caps_long_axis(self):
        self.assertEqual(preview_target_shape((400, 200, 200), max_dim=100), (100, 50, 50))

    def test_resize_volume_nearest_keeps_binary(self):
        volume = np.zeros((8, 8, 8), dtype=np.uint8)
        volume[2:6, 2:6, 2:6] = 1
        out = resize_volume_nearest(volume, (4, 4, 4))
        self.assertEqual(out.shape, (4, 4, 4))
        self.assertGreater(int(out.sum()), 0)
        self.assertTrue(set(np.unique(out)).issubset({0, 1}))

    def test_fill_speckle_zeros_keeps_background(self):
        from pipeline.aiFineTuning.preview_store import fill_speckle_zeros

        volume = np.zeros((5, 5, 5), dtype=np.uint8)
        volume[1:4, 1:4, 1:4] = 120
        volume[2, 2, 2] = 0
        filled = fill_speckle_zeros(volume)
        self.assertGreater(int(filled[2, 2, 2]), 40)
        self.assertEqual(int(filled[0, 0, 0]), 0)

    def test_save_and_resolve_epoch_preview(self):
        import tempfile
        from pathlib import Path

        example = {"subject": "SubA", "label_id": 2}
        image = np.arange(8 * 8 * 8, dtype=np.uint8).reshape(8, 8, 8)
        pred = np.zeros((8, 8, 8), dtype=np.uint8)
        pred[2:5, 2:5, 2:5] = 1
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            info = save_epoch_prediction(root, example, image, pred, epoch=3, dice=0.8)
            manifest = update_manifest(root, sticky_key=info["key"], subject_info=info, epoch=3)
            self.assertEqual(manifest["sticky_key"], "SubA__label2")
            image_path = resolve_preview_file(root, subject_key_value=info["key"], kind="image")
            pred_path = resolve_preview_file(root, subject_key_value=info["key"], kind="pred", epoch=3)
            self.assertTrue(image_path.is_file())
            self.assertTrue(pred_path.is_file())
            self.assertEqual(pred_path.name, "pred_epoch_0003.nii.gz")
            self.assertIsNone(manifest.get("preview_kind"))
            manifest = update_manifest(
                root, sticky_key=info["key"], subject_info=info, epoch=3, preview_kind="clip_slabs",
            )
            self.assertEqual(manifest["preview_kind"], "clip_slabs")

    def test_scatter_clip_slab_paints_only_those_slices(self):
        volume_shape = (8, 8, 8)
        frame = np.ones((8, 8), dtype=bool)
        axial = scatter_clip_slab(volume_shape, 0, [3, 4], [frame, frame])
        self.assertEqual(axial.shape, volume_shape)
        self.assertTrue(np.all(axial[3] == 1))
        self.assertTrue(np.all(axial[4] == 1))
        self.assertEqual(int(axial[:3].sum() + axial[5:].sum()), 0)

        coronal = scatter_clip_slab(volume_shape, 1, [2], [frame])
        self.assertTrue(np.all(coronal[:, 2, :] == 1))
        self.assertEqual(int(coronal[:, :2, :].sum() + coronal[:, 3:, :].sum()), 0)

        sagittal = scatter_clip_slab(volume_shape, 2, [5], [frame])
        self.assertTrue(np.all(sagittal[:, :, 5] == 1))
        self.assertEqual(int(sagittal[:, :, :5].sum() + sagittal[:, :, 6:].sum()), 0)

    def test_assemble_clip_overlay_unions_views(self):
        frame = np.ones((8, 8), dtype=bool)
        overlay = assemble_clip_overlay((8, 8, 8), [
            {"view_index": 0, "slice_indices": [3], "frames": [frame]},
            {"view_index": 1, "slice_indices": [2], "frames": [frame]},
        ])
        self.assertTrue(np.all(overlay[3] == 1))
        self.assertTrue(np.all(overlay[:, 2, :] == 1))
        self.assertEqual(int(overlay[0, 0, 0]), 0)
