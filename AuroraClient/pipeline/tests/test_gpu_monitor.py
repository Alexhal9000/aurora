from unittest.mock import patch

from django.test import SimpleTestCase

from pipeline.aiFineTuning.gpu_monitor import gpu_snapshot


class GpuMonitorTests(SimpleTestCase):
    def test_snapshot_has_required_keys(self):
        snapshot = gpu_snapshot(force=True)
        self.assertIn("available", snapshot)
        self.assertIn("kind", snapshot)
        self.assertIn("name", snapshot)
        self.assertIn("gpus", snapshot)

    def test_parses_nvidia_smi_csv(self):
        csv = "0, NVIDIA GeForce RTX 5070 Ti, 16303, 4455, 11848, 12, 38\n"
        completed = type("R", (), {"returncode": 0, "stdout": csv, "stderr": ""})()
        with patch("pipeline.aiFineTuning.gpu_monitor.shutil.which", return_value="/usr/bin/nvidia-smi"):
            with patch("pipeline.aiFineTuning.gpu_monitor.subprocess.run", return_value=completed):
                snapshot = gpu_snapshot(force=True)
        self.assertTrue(snapshot["available"])
        self.assertEqual(snapshot["name"], "NVIDIA GeForce RTX 5070 Ti")
        selected = snapshot["selected"]
        self.assertGreater(selected["memory_total_gib"], 15)
        self.assertEqual(selected["utilization_pct"], 12.0)
        self.assertEqual(selected["temperature_c"], 38.0)
