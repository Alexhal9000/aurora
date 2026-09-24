import torch
from django.test import TestCase
from torch import nn

from pipeline.aiModels.medsam2_adapter import apply_trainable_preset


class DummyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Linear(4, 4)
        self.neck = nn.Linear(4, 4)


class DummySAM(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = DummyEncoder()
        self.memory_attention = nn.Linear(4, 4)
        self.memory_encoder = nn.Linear(4, 4)
        self.sam_prompt_encoder = nn.Linear(4, 4)
        self.sam_mask_decoder = nn.Linear(4, 4)


class TrainablePresetTests(TestCase):
    def _flags(self, preset):
        model = DummySAM()
        return apply_trainable_preset(model, preset), model

    def test_decoder_focused_freezes_encoder(self):
        flags, model = self._flags("decoder_focused")
        self.assertTrue(flags["sam_mask_decoder.weight"])
        self.assertTrue(flags["sam_prompt_encoder.weight"])
        self.assertFalse(flags["image_encoder.trunk.weight"])
        self.assertFalse(flags["image_encoder.neck.weight"])
        self.assertFalse(flags["memory_attention.weight"])
        self.assertFalse(model.image_encoder.trunk.weight.requires_grad)

    def test_encoder_decoder_trains_neck_and_memory(self):
        flags, _ = self._flags("encoder_decoder")
        self.assertTrue(flags["image_encoder.neck.weight"])
        self.assertTrue(flags["memory_encoder.weight"])
        self.assertFalse(flags["image_encoder.trunk.weight"])

    def test_full_trains_everything(self):
        flags, _ = self._flags("full")
        self.assertTrue(all(flags.values()))
