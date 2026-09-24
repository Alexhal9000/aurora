from pathlib import Path
from tempfile import mkdtemp

from django.test import TestCase

from pipeline.aiFineTuning.validate import validate_experiment
from pipeline.tests.helpers import write_project


class ValidateTests(TestCase):
    def setUp(self):
        self.root = Path(mkdtemp())
        write_project(self.root)

    def _payload(self, **overrides):
        data = {
            "directory": str(self.root),
            "subjects": ["SubA", "SubB", "SubC"],
            "label_ids": [1],
            "foundation_model": "MedSAM2",
            "dimensionality": "3D",
            "paradigm": "prompt_guided",
            "evaluation_strategy": "train_val_test",
            "trainable_preset": "decoder_focused",
            "seed": 123,
        }
        data.update(overrides)
        return data

    def test_valid_split(self):
        report = validate_experiment(self._payload())
        self.assertTrue(report.ok, report.as_dict())
        self.assertEqual(
            sorted(report.assignments["train"] + report.assignments["validation"] + report.assignments["test"]),
            ["SubA", "SubB", "SubC"],
        )

    def test_rejects_direct_paradigm(self):
        report = validate_experiment(self._payload(paradigm="direct"))
        self.assertFalse(report.ok)
        self.assertTrue(any(item["code"] == "unsupported_paradigm" for item in report.errors))

    def test_rejects_2d_task(self):
        report = validate_experiment(self._payload(dimensionality="2D"))
        self.assertFalse(report.ok)
        self.assertTrue(any(item["code"] == "unsupported_dimensionality" for item in report.errors))

    def test_overlapping_user_split(self):
        report = validate_experiment(self._payload(
            evaluation_strategy="user",
            train_subjects=["SubA", "SubB"],
            validation_subjects=["SubB"],
            test_subjects=["SubC"],
        ))
        self.assertFalse(report.ok)
        self.assertTrue(any(item["code"] == "invalid_split" for item in report.errors))

    def test_insufficient_subjects(self):
        report = validate_experiment(self._payload(subjects=["SubA"]))
        self.assertFalse(report.ok)
        self.assertTrue(any(item["code"] == "insufficient_subjects" for item in report.errors))

    def test_kfold_assignments(self):
        report = validate_experiment(self._payload(
            subjects=["SubA", "SubB", "SubC", "SubD"],
            evaluation_strategy="kfold",
            k_folds=4,
        ))
        self.assertTrue(report.ok, report.as_dict())
        self.assertEqual(report.assignments["strategy"], "kfold")
        self.assertEqual(len(report.assignments["folds"]), 4)
