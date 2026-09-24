from django.test import SimpleTestCase

from pipeline.aiFineTuning.validation_mode import (
    DICE_SOURCE_ALL_VOLUMES,
    DICE_SOURCE_CLIP,
    DICE_SOURCE_ONE_VOLUME,
    MODE_FAST,
    MODE_INTERMEDIATE,
    MODE_SLOW,
    dice_source_for_mode,
    resolve_validation_mode,
    volume_val_examples,
)


class ValidationModeTests(SimpleTestCase):
    def test_unknown_mode_falls_back_to_fast(self):
        self.assertEqual(resolve_validation_mode(None), MODE_FAST)
        self.assertEqual(resolve_validation_mode("nope"), MODE_FAST)

    def test_fast_skips_volume_dice(self):
        examples = [{"subject": "A"}, {"subject": "B"}]
        self.assertEqual(volume_val_examples(MODE_FAST, examples), [])
        self.assertEqual(dice_source_for_mode(MODE_FAST), DICE_SOURCE_CLIP)

    def test_intermediate_uses_sticky_subject_only(self):
        examples = [{"subject": "A"}, {"subject": "B"}]
        sticky = {"subject": "B"}
        chosen = volume_val_examples(MODE_INTERMEDIATE, examples, sticky=sticky)
        self.assertEqual(len(chosen), 1)
        self.assertEqual(chosen[0]["subject"], "B")
        self.assertEqual(dice_source_for_mode(MODE_INTERMEDIATE), DICE_SOURCE_ONE_VOLUME)

    def test_slow_uses_every_val_subject(self):
        examples = [{"subject": "A"}, {"subject": "B"}]
        chosen = volume_val_examples(MODE_SLOW, examples)
        self.assertEqual([row["subject"] for row in chosen], ["A", "B"])
        self.assertEqual(dice_source_for_mode(MODE_SLOW), DICE_SOURCE_ALL_VOLUMES)

    def test_contract_default_is_fast(self):
        from pipeline.aiModels.contracts import medsam2_train_capabilities

        spec = medsam2_train_capabilities()["hyperparameters"]["validation_mode"]
        self.assertEqual(spec["default"], MODE_FAST)
        self.assertEqual(spec["options"], [MODE_FAST, MODE_INTERMEDIATE, MODE_SLOW])
