from pathlib import Path
from tempfile import mkdtemp
from unittest.mock import patch

from django.test import TestCase

from pipeline.aiFineTuning.metadata import build_model_metadata, default_display_name
from pipeline.aiFineTuning.storage import atomic_write_json
from pipeline.aiModels.paths import ENV_MODELS_ROOT
from pipeline.aiModels.registry import get_model, list_model_descriptors, run_inference


class MetadataAndRegistryTests(TestCase):
    def setUp(self):
        self.root = Path(mkdtemp())
        self.env = patch.dict("os.environ", {ENV_MODELS_ROOT: str(self.root)})
        self.env.start()

    def tearDown(self):
        self.env.stop()

    def test_default_display_name(self):
        name = default_display_name("C57 Adult Skulls", ["Mandible", "Maxilla"])
        self.assertEqual(name, "MedSAM2 — C57 Adult Skulls — Mandible + Maxilla")

    def test_metadata_has_contract_and_ids(self):
        metadata = build_model_metadata(
            model_id="abc-123",
            display_name="MedSAM2 — Demo — Mandible",
            run_id="run-1",
            config={
                "directory": "/tmp/project",
                "project_name": "Demo",
                "foundation_model": "MedSAM2",
                "dimensionality": "3D",
                "paradigm": "prompt_guided",
                "labels": [{"id": 17, "name": "Mandible"}],
                "subjects": ["SubA"],
                "assignments": {"train": ["SubA"], "validation": [], "test": []},
                "trainable_preset": "decoder_focused",
                "hyperparameters": {"epochs": 2},
                "augmentation": {"enabled": False},
                "seed": 123,
            },
            evaluation={"validation": {"dice": 0.8}},
            selected_checkpoint={"path": "epoch.pt", "epoch": 1, "criterion": "best_val_dice"},
        )
        self.assertEqual(metadata["identity"]["model_id"], "abc-123")
        self.assertEqual(metadata["task_contract"]["dimensionality"], "3D")
        self.assertEqual(metadata["task_contract"]["paradigm"], "prompt_guided")
        self.assertEqual(metadata["task_contract"]["label_ids"], [17])
        self.assertEqual(metadata["training_data"]["source_project_name"], "Demo")
        self.assertEqual(metadata["checkpoint_selection"]["criterion"], "best_val_dice")
        self.assertEqual(metadata["inference"]["engine"], "MedSAM2Segmenter")

    def test_foundation_is_listed(self):
        models = list_model_descriptors()
        self.assertTrue(any(item["id"] == "MedSAM2" and item["is_foundation"] for item in models))

    def test_register_then_list(self):
        model_id = "11111111-2222-3333-4444-555555555555"
        model_dir = self.root / model_id
        model_dir.mkdir(parents=True)
        (model_dir / "MedSAM2_finetuned.pt").write_bytes(b"ckpt")
        metadata = build_model_metadata(
            model_id=model_id,
            display_name="MedSAM2 — Demo — Mandible",
            run_id="run-1",
            config={
                "directory": "/tmp/project",
                "project_name": "Demo",
                "labels": [{"id": 1, "name": "Mandible"}],
                "dimensionality": "3D",
                "paradigm": "prompt_guided",
            },
        )
        atomic_write_json(model_dir / "metadata.json", metadata)
        listed = list_model_descriptors()
        ids = [item["id"] for item in listed]
        self.assertIn(model_id, ids)
        loaded = get_model(model_id)
        self.assertEqual(loaded["display_name"], "MedSAM2 — Demo — Mandible")
        self.assertFalse(loaded["is_foundation"])

    def test_corrupt_metadata_is_ignored(self):
        model_id = "deadbeef-dead-beef-dead-beefdeadbeef"
        model_dir = self.root / model_id
        model_dir.mkdir(parents=True)
        (model_dir / "metadata.json").write_text("{not-json")
        self.assertIsNone(get_model(model_id))
        ids = [item["id"] for item in list_model_descriptors()]
        self.assertNotIn(model_id, ids)

    def test_missing_checkpoint_is_ignored(self):
        model_id = "00000000-0000-0000-0000-000000000001"
        model_dir = self.root / model_id
        model_dir.mkdir(parents=True)
        metadata = build_model_metadata(
            model_id=model_id,
            display_name="Broken",
            run_id="run-x",
            config={"labels": [{"id": 1, "name": "X"}]},
        )
        atomic_write_json(model_dir / "metadata.json", metadata)
        self.assertIsNone(get_model(model_id))

    def test_full_slice_inference_ignores_drawn_label_mask(self):
        import numpy as np

        model_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        model_dir = self.root / model_id
        model_dir.mkdir(parents=True)
        (model_dir / "MedSAM2_finetuned.pt").write_bytes(b"ckpt")
        metadata = build_model_metadata(
            model_id=model_id,
            display_name="MedSAM2 — Demo — Whole slice",
            run_id="run-2",
            config={
                "directory": "/tmp/project",
                "project_name": "Demo",
                "labels": [{"id": 1, "name": "Mandible"}],
                "dimensionality": "3D",
                "paradigm": "prompt_guided",
                "prompt": {
                    "prompt_type": "full_slice_ones",
                    "initialization": "full_slice_ones",
                },
            },
        )
        atomic_write_json(model_dir / "metadata.json", metadata)

        image = np.zeros((4, 6, 8), dtype=np.float32)
        mask = np.zeros((4, 6, 8), dtype=np.uint8)
        mask[1:3, 2:4, 2:5] = 1
        expected = np.zeros((4, 6, 8), dtype=np.uint8)
        expected[1:3, 2:4, 2:5] = 1

        with patch(
            "pipeline.aiModels.medsam2_adapter.MedSAM2Segmenter",
        ) as segmenter_cls:
            segmenter_cls.return_value.segment.return_value = expected
            result = run_inference(image, mask, target_label=1, model_id=model_id)

        segmenter_cls.assert_called_once()
        segmenter_cls.return_value.segment.assert_called_once()
        self.assertGreater(int(np.sum(result == 1)), 0)
