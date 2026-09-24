from pathlib import Path
from tempfile import mkdtemp

from django.test import TestCase

from pipeline.aiFineTuning.dataset import collect_examples, isolate_label, load_example_arrays, preload_example_cache
from pipeline.aiFineTuning.discover import discover_project
from pipeline.tests.helpers import write_project


class DatasetTests(TestCase):
    def setUp(self):
        self.root = Path(mkdtemp())
        write_project(self.root)

    def test_label_isolation_and_alignment(self):
        discovered = discover_project(str(self.root))
        records = {item["name"]: item for item in discovered["subjects"]}
        examples = collect_examples(str(self.root), ["SubA"], [1], records)
        self.assertEqual(len(examples), 1)
        image, binary = load_example_arrays(examples[0])
        self.assertEqual(image.shape, binary.shape)
        self.assertEqual(set(binary.ravel().tolist()), {0, 1})
        self.assertGreater(int(binary.sum()), 0)

    def test_preload_cache_loads_once(self):
        from pipeline.aiFineTuning.dataset import clear_example_cache

        discovered = discover_project(str(self.root))
        records = {item["name"]: item for item in discovered["subjects"]}
        examples = collect_examples(str(self.root), ["SubA"], [1], records)
        clear_example_cache()
        self.assertEqual(preload_example_cache(examples), 1)
        self.assertEqual(preload_example_cache(examples), 0)
        image, binary = load_example_arrays(examples[0])
        self.assertEqual(image.dtype.kind, "u")
        self.assertGreater(int(binary.sum()), 0)

    def test_isolate_label_helper(self):
        import numpy as np
        mask = np.zeros((4, 4, 4), dtype=np.uint8)
        mask[1, 1, 1] = 7
        mask[2, 2, 2] = 3
        isolated = isolate_label(mask, 7)
        self.assertEqual(int(isolated.sum()), 1)
        self.assertEqual(isolated[1, 1, 1], 1)

    def test_records_edit_filenames(self):
        discovered = discover_project(str(self.root))
        record = next(item for item in discovered["subjects"] if item["name"] == "SubA")
        self.assertEqual(record["image_filename"], "SubA.nii.gz")
        self.assertEqual(record["mask_filename"], "SubA.nii.mask.gz")

    def test_middle_slab_is_central_half_of_structure(self):
        import numpy as np
        from pipeline.aiFineTuning.dataset import middle_slab_indices

        mask = np.zeros((20, 4, 4), dtype=np.uint8)
        mask[2:18, 1, 1] = 1
        slab = middle_slab_indices(mask, slab_fraction=0.5)
        self.assertTrue(np.all(slab >= 2) and np.all(slab <= 17))
        self.assertGreaterEqual(int(slab[0]), 6)
        self.assertLessEqual(int(slab[-1]), 13)

    def test_eval_clip_jobs_one_volume_three_views(self):
        from pipeline.aiFineTuning.train_worker import _build_eval_clip_jobs
        from pipeline.aiFineTuning.discover import discover_project
        from pipeline.aiFineTuning.dataset import collect_examples

        discovered = discover_project(str(self.root))
        records = {item["name"]: item for item in discovered["subjects"]}
        examples = collect_examples(str(self.root), ["SubA"], [1], records)
        jobs = _build_eval_clip_jobs(examples, num_frames=4, resolution=16)
        self.assertEqual(len(jobs), 3)
        self.assertEqual([job["view_index"] for job in jobs], [0, 1, 2])
        self.assertEqual(jobs[0]["clip"]["video"].shape[0], 4)

    def test_eval_prompt_is_deterministic_centre_of_slab(self):
        import numpy as np
        from pipeline.aiFineTuning.dataset import choose_init_slice, middle_slab_indices

        mask = np.zeros((20, 4, 4), dtype=np.uint8)
        mask[2:18, 1, 1] = 1
        slab = middle_slab_indices(mask)
        expected = int(slab[len(slab) // 2])
        self.assertEqual(choose_init_slice(mask), expected)
        self.assertEqual(choose_init_slice(mask, rng=None), expected)

    def test_training_prompt_samples_from_middle_slab(self):
        import numpy as np
        from pipeline.aiFineTuning.dataset import choose_init_slice, middle_slab_indices

        mask = np.zeros((20, 4, 4), dtype=np.uint8)
        mask[2:18, 1, 1] = 1
        slab = set(int(i) for i in middle_slab_indices(mask))
        rng = np.random.default_rng(0)
        picks = {choose_init_slice(mask, rng=rng) for _ in range(40)}
        self.assertTrue(picks.issubset(slab))
        self.assertGreater(len(picks), 1)

    def test_clip_starts_at_prompt_slice(self):
        import numpy as np
        from pipeline.aiFineTuning.dataset import (
            build_clip,
            choose_init_slice,
            clip_indices_from_prompt,
        )

        image = np.zeros((12, 8, 8), dtype=np.float32)
        mask = np.zeros((12, 8, 8), dtype=np.uint8)
        mask[3:10, 2:6, 2:6] = 1
        prompt = choose_init_slice(mask)
        indices = clip_indices_from_prompt(prompt, 12, 8)
        self.assertEqual(indices[0], prompt)
        clip = build_clip(image, mask, num_frames=8, resolution=8)
        self.assertIsNotNone(clip)
        self.assertEqual(clip["slice_indices"][0], clip["init_slice"])
        self.assertEqual(clip["init_slice"], prompt)

        rng = np.random.default_rng(0)
        train_clip = build_clip(image, mask, num_frames=8, resolution=8, rng=rng)
        self.assertEqual(train_clip["slice_indices"][0], train_clip["init_slice"])
        self.assertTrue(train_clip["video"].flags["C_CONTIGUOUS"])
        self.assertTrue(train_clip["masks"].flags["C_CONTIGUOUS"])

    def test_full_slice_prompt_uses_whole_frame_box_and_ones(self):
        import numpy as np
        from pipeline.aiFineTuning.dataset import build_clip
        from pipeline.aiModels.prompts import is_full_slice_prompt

        self.assertTrue(is_full_slice_prompt({"initialization": "full_slice_ones"}))
        image = np.zeros((12, 8, 8), dtype=np.float32)
        mask = np.zeros((12, 8, 8), dtype=np.uint8)
        mask[3:10, 2:6, 2:6] = 1
        clip = build_clip(
            image, mask, num_frames=8, resolution=8,
            prompt={"initialization": "full_slice_ones"},
        )
        self.assertIsNotNone(clip)
        self.assertTrue(clip["full_slice_prompt"])
        self.assertEqual(list(clip["bbox"]), [0.0, 0.0, 7.0, 7.0])
        self.assertEqual(float(clip["prompt_mask"].min()), 1.0)
        self.assertLess(float(clip["masks"][0].mean()), 1.0)

    def test_clip_near_volume_end_still_starts_at_prompt(self):
        from pipeline.aiFineTuning.dataset import clip_indices_from_prompt

        indices = clip_indices_from_prompt(10, 12, 8)
        self.assertEqual(indices[0], 10)
        self.assertEqual(len(indices), 8)
