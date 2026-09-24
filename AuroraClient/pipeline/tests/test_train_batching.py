from django.test import SimpleTestCase

import numpy as np

from pipeline.aiFineTuning.train_batching import (
    clips_to_batch_tensors,
    last_frame_binary_metrics,
    loss_from_outputs,
)


class TrainBatchingTests(SimpleTestCase):
    def test_clips_to_batch_tensors_shapes(self):
        import torch

        clip_a = {
            "video": np.ones((4, 3, 8, 8), dtype=np.float32),
            "masks": np.zeros((4, 8, 8), dtype=np.float32),
            "bbox": np.array([0.0, 0.0, 7.0, 7.0], dtype=np.float32),
            "prompt_mask": np.ones((8, 8), dtype=np.float32),
        }
        clip_b = {
            "video": np.ones((4, 3, 8, 8), dtype=np.float32) * 2,
            "masks": np.zeros((4, 8, 8), dtype=np.float32),
            "bbox": np.array([1.0, 1.0, 6.0, 6.0], dtype=np.float32),
            "prompt_mask": np.ones((8, 8), dtype=np.float32),
        }
        video, masks, bbox, dense = clips_to_batch_tensors([clip_a, clip_b], torch.device("cpu"))
        self.assertEqual(tuple(video.shape), (2, 4, 3, 8, 8))
        self.assertEqual(tuple(masks.shape), (2, 4, 1, 8, 8))
        self.assertEqual(tuple(bbox.shape), (2, 4))
        self.assertEqual(tuple(dense.shape), (2, 1, 8, 8))

    def test_last_frame_metrics_are_per_clip(self):
        import torch

        pred = torch.zeros(2, 1, 4, 4)
        pred[0, 0, :2, :2] = 1
        pred[1, 0, :, :] = 1
        target = torch.zeros(2, 3, 1, 4, 4)
        target[0, -1, 0, :2, :2] = 1
        target[1, -1, 0, :2, :2] = 1
        rows = last_frame_binary_metrics([pred], target)
        self.assertEqual(len(rows), 2)
        self.assertGreater(rows[0]["dice"], 0.99)
        self.assertLess(rows[1]["dice"], rows[0]["dice"])

    def test_loss_from_outputs_is_a_scalar(self):
        import torch

        logit = torch.zeros(2, 1, 4, 4)
        masks = torch.zeros(2, 1, 1, 4, 4)
        loss = loss_from_outputs([logit], None, masks, {"loss_mask": 1.0, "loss_dice": 0.0, "loss_iou": 0.0})
        self.assertTrue(torch.is_tensor(loss))
        self.assertEqual(tuple(loss.shape), ())
