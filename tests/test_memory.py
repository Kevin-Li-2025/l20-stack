import unittest

from l20_stack.memory import ModelSpec, TrainingSpec, estimate_training_memory


class MemoryEstimateTest(unittest.TestCase):
    def test_default_7b_qlora_plan_fits_l20_budget(self):
        estimate = estimate_training_memory(
            ModelSpec(
                name="test-7b",
                params_b=7.0,
                hidden_size=4096,
                layers=32,
                target_modules_per_layer=4,
            ),
            TrainingSpec(seq_len=4096, micro_batch_size=1, lora_rank=64),
        )

        self.assertTrue(estimate.fits_device)
        self.assertLess(estimate.total_gib, 46.0)
        self.assertGreater(estimate.lora_trainable_params_m, 60.0)

    def test_large_unquantized_plan_does_not_fit(self):
        estimate = estimate_training_memory(
            ModelSpec(name="test-70b", params_b=70.0, hidden_size=8192, layers=80),
            TrainingSpec(quant_bits=16, seq_len=8192, micro_batch_size=1),
        )

        self.assertFalse(estimate.fits_device)
        self.assertGreater(estimate.total_gib, 48.0)

    def test_rejects_invalid_quantization_bits(self):
        with self.assertRaises(ValueError):
            estimate_training_memory(
                ModelSpec(name="bad", params_b=7.0, hidden_size=4096, layers=32),
                TrainingSpec(quant_bits=0),
            )


if __name__ == "__main__":
    unittest.main()
