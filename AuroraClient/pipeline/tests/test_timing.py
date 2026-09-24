from pathlib import Path
from tempfile import mkdtemp
from unittest.mock import patch

from django.test import SimpleTestCase

from pipeline.aiFineTuning.timing import TimingLog
from pipeline.aiModels.paths import ENV_MODELS_ROOT


class TimingLogTests(SimpleTestCase):
    def setUp(self):
        self.models = Path(mkdtemp())
        self.env = patch.dict("os.environ", {ENV_MODELS_ROOT: str(self.models)})
        self.env.start()

    def tearDown(self):
        self.env.stop()

    def test_span_writes_summary_and_verdict(self):
        log = TimingLog("run-timing", extra={"n_val": 4})
        with log.span("train", epoch=1):
            pass
        log.event("val_volume", seconds=2.0, epoch=1, n=4)
        log.event(
            "epoch",
            seconds=2.1,
            epoch=1,
            train_s=0.01,
            val_s=2.0,
            val_volume_s=2.0,
            steps=1,
            clips=3,
        )
        payload = log.flush()
        self.assertTrue((self.models / "runs" / "run-timing" / "logs" / "timing.jsonl").is_file())
        self.assertTrue((self.models / "runs" / "run-timing" / "logs" / "timing.txt").is_file())
        self.assertGreater(payload["val_total_s"], payload["train_total_s"])
        self.assertIn("Validation", payload["verdict"])
        self.assertEqual(len(payload["epochs"]), 1)
