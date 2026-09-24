from pathlib import Path
from tempfile import mkdtemp
from unittest.mock import patch

from django.test import TestCase

from pipeline.aiFineTuning.library import (
    LibraryError,
    delete_registered_model,
    export_registered_zip,
    import_registered_zip,
    list_registered_library,
)
from pipeline.aiFineTuning.metadata import build_model_metadata
from pipeline.aiFineTuning.storage import atomic_write_json
from pipeline.aiModels.paths import ENV_MODELS_ROOT
from pipeline.aiModels.registry import get_model, list_model_descriptors


class RegisteredModelLibraryTests(TestCase):
    def setUp(self):
        self.root = Path(mkdtemp())
        self.env = patch.dict("os.environ", {ENV_MODELS_ROOT: str(self.root)})
        self.env.start()
        self.model_id = "11111111-2222-3333-4444-555555555555"
        model_dir = self.root / self.model_id
        model_dir.mkdir(parents=True)
        (model_dir / "MedSAM2_finetuned.pt").write_bytes(b"ckpt")
        (model_dir / "figures").mkdir()
        (model_dir / "figures" / "training_curves.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"x" * 20)
        metadata = build_model_metadata(
            model_id=self.model_id,
            display_name="MedSAM2 — Demo — Mandible",
            run_id="run-1",
            config={
                "directory": "/tmp/project",
                "project_name": "Demo",
                "labels": [{"id": 1, "name": "Mandible"}],
                "dimensionality": "3D",
                "paradigm": "prompt_guided",
                "prompt": {"initialization": "full_slice_ones", "prompt_type": "full_slice_ones"},
            },
            evaluation={"validation": {"dice": 0.81}},
        )
        atomic_write_json(model_dir / "metadata.json", metadata)

    def tearDown(self):
        self.env.stop()

    def test_list_excludes_foundation(self):
        rows = list_registered_library()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["id"], self.model_id)
        self.assertEqual(rows[0]["val_dice"], 0.81)
        self.assertTrue(any(item["is_foundation"] for item in list_model_descriptors()))

    def test_delete_removes_from_registry(self):
        delete_registered_model(self.model_id)
        self.assertIsNone(get_model(self.model_id))
        self.assertEqual(list_registered_library(), [])
        self.assertFalse((self.root / self.model_id).exists())

    def test_cannot_delete_foundation(self):
        with self.assertRaises(LibraryError):
            delete_registered_model("MedSAM2")

    def test_zip_roundtrip(self):
        buffer, filename = export_registered_zip(self.model_id)
        self.assertTrue(filename.endswith(".zip"))
        delete_registered_model(self.model_id)
        imported = import_registered_zip(buffer.getvalue())
        self.assertTrue(imported["ok"])
        loaded = get_model(imported["model_id"])
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["display_name"], "MedSAM2 — Demo — Mandible")
        self.assertEqual(loaded.get("prompt_initialization"), "full_slice_ones")
