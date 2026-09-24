from pathlib import Path

from django.test import TestCase

from pipeline.aiFineTuning.discover import discover_project
from pipeline.tests.helpers import write_project


class DiscoverTests(TestCase):
    def setUp(self):
        self.root = Path(self._test_dir())
        write_project(self.root)

    def _test_dir(self):
        from tempfile import mkdtemp
        return mkdtemp()

    def test_label_names_and_ids(self):
        payload = discover_project(str(self.root))
        self.assertEqual(payload["label_names"]["1"], "Mandible")
        self.assertEqual(payload["label_names"]["2"], "Maxilla")
        ids = {item["id"] for item in payload["labels"]}
        self.assertIn(1, ids)
        self.assertIn(2, ids)

    def test_eligible_subjects_have_masks(self):
        payload = discover_project(str(self.root))
        names = {item["name"] for item in payload["subjects"] if item["eligible"]}
        self.assertEqual(names, {"SubA", "SubB", "SubC", "SubD"})
        mandible = next(item for item in payload["labels"] if item["id"] == 1)
        self.assertIn("SubA", mandible["subjects"])
        self.assertIn("SubD", mandible["subjects"])

    def test_missing_mask_is_flagged(self):
        subject = self.root / "extracted" / "SubB"
        mask = subject / "SubB.nii.mask.gz"
        mask.unlink()
        payload = discover_project(str(self.root))
        record = next(item for item in payload["subjects"] if item["name"] == "SubB")
        self.assertFalse(record["eligible"])
        self.assertIn("mask_missing", record["issues"])
        self.assertEqual(record["image_filename"], "SubB.nii.gz")

    def test_skips_elastic_even_when_it_has_a_mask(self):
        from pipeline.aiFineTuning.discover import find_latest_fullres_with_mask

        scan_dir = self.root / "extracted" / "SubA"
        _write_volume_pair(scan_dir, "SubA_edit_1_manual.nii.gz", label=1)
        _write_volume_pair(scan_dir, "SubA_edit_2_elastic.nii.gz", label=2)
        image, mask = find_latest_fullres_with_mask(str(scan_dir), "SubA")
        self.assertEqual(image, "SubA_edit_1_manual.nii.gz")
        self.assertEqual(mask, "SubA_edit_1_manual.nii.mask.gz")

    def test_raw_is_used_when_there_are_no_edits(self):
        from pipeline.aiFineTuning.discover import find_latest_fullres_with_mask

        scan_dir = self.root / "extracted" / "SubA"
        image, mask = find_latest_fullres_with_mask(str(scan_dir), "SubA")
        self.assertEqual(image, "SubA.nii.gz")
        self.assertEqual(mask, "SubA.nii.mask.gz")

    def test_raw_is_used_when_only_later_edits_are_elastic(self):
        from pipeline.aiFineTuning.discover import find_latest_fullres_with_mask

        scan_dir = self.root / "extracted" / "SubA"
        _write_volume_pair(scan_dir, "SubA_edit_3_elastic.nii.gz", label=2)
        _write_volume_pair(scan_dir, "SubA_edit_3_elastic_fwd.nii.gz", label=2)
        image, mask = find_latest_fullres_with_mask(str(scan_dir), "SubA")
        self.assertEqual(image, "SubA.nii.gz")
        self.assertEqual(mask, "SubA.nii.mask.gz")


def _write_volume_pair(scan_dir, image_name, label=1):
    import os

    import nibabel as nib
    import numpy as np

    shape = (8, 8, 8)
    affine = np.diag([0.05, 0.05, 0.05, 1.0])
    image = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    mask = np.zeros(shape, dtype=np.uint8)
    mask[1:3, 1:3, 1:3] = int(label)
    nib.save(nib.Nifti1Image(image, affine), str(scan_dir / image_name))
    mask_name = image_name[: -len(".nii.gz")] + ".nii.mask.gz"
    temp_mask = scan_dir / f"{image_name}.mask_tmp.nii.gz"
    nib.save(nib.Nifti1Image(mask, affine), str(temp_mask))
    os.replace(temp_mask, scan_dir / mask_name)
