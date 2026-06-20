import json
import tempfile
import unittest
from pathlib import Path

from l20_stack.experiment import ExperimentConfig
from l20_stack.memory import estimate_training_memory


class ExperimentConfigTest(unittest.TestCase):
    def test_loads_config_and_estimates_memory(self):
        config = ExperimentConfig.from_file("configs/qlora_l20.json")
        estimate = estimate_training_memory(config.model, config.training)

        self.assertEqual(config.task, "qlora-smoke-plan")
        self.assertEqual(config.training.device_memory_gib, 48.0)
        self.assertTrue(estimate.fits_device)

    def test_rejects_unknown_keys(self):
        payload = {
            "task": "bad",
            "dataset": "fixture",
            "output_dir": "outputs/bad",
            "model": {
                "name": "bad",
                "params_b": 1.0,
                "hidden_size": 128,
                "layers": 2,
                "unexpected": True,
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unknown ModelSpec keys"):
                ExperimentConfig.from_file(str(path))


if __name__ == "__main__":
    unittest.main()
