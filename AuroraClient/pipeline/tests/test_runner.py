from pathlib import Path
from tempfile import mkdtemp
from unittest.mock import patch

from django.test import TestCase

from pipeline.aiFineTuning.evaluate import dice_score, iou_score
from pipeline.aiFineTuning.runner import register_run, start_run
from pipeline.aiFineTuning.storage import read_run_config, read_run_status, run_dir
from pipeline.aiModels.paths import ENV_MODELS_ROOT
from pipeline.aiModels.registry import get_model
from pipeline.tests.helpers import write_project


class RunnerDryRunTests(TestCase):
    def setUp(self):
        self.project = Path(mkdtemp())
        write_project(self.project)
        self.models = Path(mkdtemp())
        self.env = patch.dict("os.environ", {ENV_MODELS_ROOT: str(self.models)})
        self.env.start()

    def tearDown(self):
        self.env.stop()

    def test_dice_iou(self):
        import numpy as np
        pred = np.array([[1, 1, 0], [0, 0, 0]])
        target = np.array([[1, 0, 0], [0, 0, 0]])
        self.assertGreater(dice_score(pred, target), 0)
        self.assertGreater(iou_score(pred, target), 0)

    def test_dry_run_then_register(self):
        import time
        result = start_run({
            "directory": str(self.project),
            "subjects": ["SubA", "SubB", "SubC"],
            "label_ids": [1],
            "foundation_model": "MedSAM2",
            "evaluation_strategy": "train_val_test",
            "trainable_preset": "decoder_focused",
            "display_name": "MedSAM2 — Test — Mandible",
            "dry_run": True,
        })
        self.assertTrue(result["ok"], result)
        run_id = result["run_id"]
        status = None
        for _ in range(40):
            status = read_run_status(run_id)
            if status and status.get("status") in ("succeeded", "failed", "cancelled"):
                break
            time.sleep(0.25)
        self.assertIsNotNone(status)
        self.assertEqual(status.get("status"), "succeeded", status)
        registered = register_run(run_id)
        self.assertTrue(registered["ok"])
        loaded = get_model(registered["model_id"])
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["display_name"], "MedSAM2 — Test — Mandible")
        self.assertEqual(loaded["paradigm"], "prompt_guided")
        self.assertTrue((run_dir(run_id) / "figures" / "training_loss.png").is_file())
        self.assertTrue((run_dir(run_id) / "figures" / "training_dice.png").is_file())
        self.assertTrue((Path(registered["path"]) / "figures" / "training_loss.png").is_file())

    def test_kfold_dry_run_packs_all_folds(self):
        import time
        result = start_run({
            "directory": str(self.project),
            "subjects": ["SubA", "SubB", "SubC", "SubD"],
            "label_ids": [1],
            "foundation_model": "MedSAM2",
            "evaluation_strategy": "kfold",
            "k_folds": 4,
            "trainable_preset": "decoder_focused",
            "display_name": "MedSAM2 — Test — Kfold",
            "dry_run": True,
        })
        self.assertTrue(result["ok"], result)
        run_id = result["run_id"]
        config = read_run_config(run_id)
        self.assertEqual(len(config.get("folds") or []), 4)
        status = None
        for _ in range(40):
            status = read_run_status(run_id)
            if status and status.get("status") in ("succeeded", "failed", "cancelled"):
                break
            time.sleep(0.25)
        self.assertEqual(status.get("status"), "succeeded", status)
        self.assertEqual(status.get("selected_fold"), 0)
        self.assertTrue((run_dir(run_id) / "figures" / "fold_val_dice.png").is_file())
