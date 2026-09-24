from django.test import SimpleTestCase

from pipeline.aiFineTuning.checkpoint import (
    INFERENCE_ENGINE,
    export_inference_checkpoint,
    fold_rank_key,
    should_save_checkpoint,
)


class CheckpointSelectionTests(SimpleTestCase):
    def test_first_epoch_with_loss_is_saved(self):
        self.assertTrue(should_save_checkpoint(1.2, 0.4, None, None))

    def test_better_dice_at_worse_loss_is_rejected(self):
        self.assertFalse(should_save_checkpoint(1.1, 0.9, 0.8, 0.5))

    def test_lower_loss_is_saved_even_if_dice_drops(self):
        self.assertTrue(should_save_checkpoint(0.6, 0.4, 0.8, 0.7))

    def test_tied_loss_keeps_higher_dice(self):
        self.assertTrue(should_save_checkpoint(0.80005, 0.72, 0.8, 0.70))

    def test_tied_loss_rejects_worse_dice(self):
        self.assertFalse(should_save_checkpoint(0.80005, 0.65, 0.8, 0.70))

    def test_small_loss_drop_inside_plateau_does_not_beat_dice_gate(self):
        # 0.799 is within 0.2% of 0.8, so this is a plateau, not a new low.
        self.assertFalse(should_save_checkpoint(0.799, 0.50, 0.8, 0.70))

    def test_missing_val_loss_never_saves(self):
        self.assertFalse(should_save_checkpoint(None, 0.99, 0.8, 0.5))

    def test_fold_rank_prefers_lower_loss_over_higher_dice(self):
        high_dice = {"best_val_loss": 0.9, "best_val_dice": 0.85}
        low_loss = {"best_val_loss": 0.4, "best_val_dice": 0.60}
        self.assertLess(fold_rank_key(low_loss), fold_rank_key(high_dice))


class InferenceCheckpointExportTests(SimpleTestCase):
    def test_export_inference_checkpoint_normalizes_payload(self):
        import tempfile
        from pathlib import Path

        import torch

        state = {"layer.weight": torch.zeros(2, 2)}
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "train.pt"
            dest = Path(tmp) / "infer.pt"
            torch.save({"model": state, "epoch": 3}, src)
            export_inference_checkpoint(src, dest)
            payload = torch.load(dest, map_location="cpu")
        self.assertEqual(payload["inference_engine"], INFERENCE_ENGINE)
        self.assertEqual(payload["epoch"], 3)
        self.assertIn("layer.weight", payload["model"])
