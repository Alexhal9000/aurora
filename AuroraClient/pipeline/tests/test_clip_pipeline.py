from django.test import SimpleTestCase

from pipeline.aiFineTuning.clip_pipeline import (
    ClipPrefetcher,
    clip_queue_depth,
    clip_worker_count,
    prebuild_epoch_clips,
)
from pipeline.aiFineTuning.train_batching import clips_to_batch_tensors, clips_to_host_tensors


class ClipPipelineTests(SimpleTestCase):
    def test_clip_worker_count_defaults_when_zero(self):
        self.assertGreaterEqual(clip_worker_count({"num_workers": 0}), 2)

    def test_clip_worker_count_respects_positive_setting(self):
        self.assertEqual(clip_worker_count({"num_workers": 6}), 6)

    def test_clip_queue_depth_scales_with_batch(self):
        self.assertGreaterEqual(clip_queue_depth(4), 16)

    def test_prebuild_epoch_clips_empty(self):
        import numpy as np

        rng = np.random.default_rng(0)
        self.assertEqual(
            prebuild_epoch_clips([], num_frames=4, resolution=32, aug={}, rng=rng),
            [],
        )

    def test_clip_prefetcher_streams_two_epochs(self):
        import numpy as np

        def build_fn(example, view_index, num_frames, resolution, aug, prompt, seed):
            return {"id": example["id"], "view": view_index, "seed": int(seed)}

        rng = np.random.default_rng(0)
        examples = [{"id": "a"}, {"id": "b"}]
        stream = ClipPrefetcher(
            examples,
            num_frames=4,
            resolution=32,
            aug={},
            rng=rng,
            num_workers=2,
            queue_depth=8,
            max_epoch=2,
            build_fn=build_fn,
        )
        try:
            first = []
            while True:
                batch = stream.take_batch(1, 3)
                if not batch:
                    break
                first.extend(batch)
            second = []
            while True:
                batch = stream.take_batch(2, 3)
                if not batch:
                    break
                second.extend(batch)
            self.assertEqual(len(first), 6)
            self.assertEqual(len(second), 6)
            self.assertEqual({item["id"] for item in first}, {"a", "b"})
            self.assertEqual({item["view"] for item in first}, {0, 1, 2})
        finally:
            stream.close()


class TrainBatchingHostTests(SimpleTestCase):
    def test_host_tensors_pin_flag_shapes(self):
        import numpy as np
        import torch

        clip = {
            "video": np.ones((4, 3, 8, 8), dtype=np.float32),
            "masks": np.zeros((4, 8, 8), dtype=np.float32),
            "bbox": np.array([0.0, 0.0, 7.0, 7.0], dtype=np.float32),
            "prompt_mask": np.ones((8, 8), dtype=np.float32),
        }
        video, masks, bbox, dense = clips_to_host_tensors([clip], pin=False)
        self.assertEqual(tuple(video.shape), (1, 4, 3, 8, 8))
        device = torch.device("cpu")
        batched = clips_to_batch_tensors([clip, clip], device)
        self.assertEqual(tuple(batched[0].shape), (2, 4, 3, 8, 8))
