from pathlib import Path
from tempfile import mkdtemp

from django.test import TestCase

from pipeline.aiFineTuning.figures import (
    even_span_indices,
    overlay_magenta_cyan,
    save_before_after_bars,
    save_dice_curves,
    save_fold_val_summary,
    save_loss_curves,
    save_training_curves,
    save_validation_multiview,
)


class FigureExportTests(TestCase):
    def test_training_and_comparison_pngs(self):
        root = Path(mkdtemp())
        history = [
            {"epoch": 1, "train_loss": 0.8, "val_loss": 0.7, "train_dice": 0.35, "val_dice": 0.4},
            {"epoch": 2, "train_loss": 0.5, "val_loss": 0.45, "train_dice": 0.55, "val_dice": 0.6},
        ]
        loss_fig = save_loss_curves(root / "training_loss.png", history)
        dice_fig = save_dice_curves(root / "training_dice.png", history)
        self.assertTrue(loss_fig.is_file())
        self.assertTrue(dice_fig.is_file())
        self.assertGreater(loss_fig.stat().st_size, 0)
        self.assertGreater(dice_fig.stat().st_size, 0)

        curves = save_training_curves(root / "training_curves.png", history)
        self.assertTrue(curves.is_file())

        folds = save_fold_val_summary(root / "fold_val_dice.png", [
            {"fold": 0, "best_val_dice": 0.71, "selected": False},
            {"fold": 1, "best_val_dice": 0.80, "selected": True},
        ])
        self.assertTrue(folds.is_file())

        bars = save_before_after_bars(root / "validation_before_after.png", [
            {"subject": "SubA", "label_id": 1, "foundation_dice": 0.2, "finetuned_dice": 0.7},
            {"subject": "SubB", "label_id": 1, "foundation_dice": 0.3, "finetuned_dice": 0.75},
        ])
        self.assertTrue(bars.is_file())

    def test_even_span_includes_ends(self):
        idx = even_span_indices(4, 22, 10)
        self.assertEqual(idx[0], 4)
        self.assertEqual(idx[-1], 22)
        self.assertEqual(len(idx), 10)

    def test_magenta_cyan_overlap_blue(self):
        import numpy as np
        image = np.full((8, 8), 0.4, dtype=np.float32)
        mag = np.zeros((8, 8), dtype=bool)
        cyan = np.zeros((8, 8), dtype=bool)
        mag[2:6, 1:5] = True
        cyan[2:6, 3:7] = True
        rgb = overlay_magenta_cyan(image, mag, cyan, alpha=1.0)
        self.assertGreater(rgb[3, 2, 0], 0.8)
        self.assertGreater(rgb[3, 2, 2], 0.8)
        self.assertGreater(rgb[3, 6, 1], 0.8)
        self.assertGreater(rgb[3, 6, 2], 0.8)
        self.assertGreater(rgb[3, 4, 2], 0.8)
        self.assertLess(rgb[3, 4, 0], 0.3)

    def test_multiview_grid_is_high_res(self):
        import numpy as np
        from PIL import Image

        root = Path(mkdtemp())
        image = np.zeros((20, 16, 18), dtype=np.float32)
        image[5:15, 4:12, 3:14] = 80
        mag = np.zeros((20, 16, 18), dtype=bool)
        cyan = np.zeros((20, 16, 18), dtype=bool)
        mag[6:14, 5:11, 4:12] = True
        cyan[7:13, 4:12, 5:13] = True
        path = save_validation_multiview(
            root / "grid.png",
            image,
            mag,
            cyan,
            subject="SubA",
            label_id=1,
            gt_mask=mag,
            n_slices=10,
            dpi=100,
            min_panel_px=80,
        )
        self.assertTrue(path.is_file())
        width, height = Image.open(path).size
        self.assertGreater(width, 700)
        self.assertGreater(height, 200)
