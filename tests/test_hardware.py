import unittest

from l20_stack.hardware import L20_SPEC, classify_roofline, roofline_balance_flops_per_byte


class HardwareTest(unittest.TestCase):
    def test_l20_spec_matches_research_baseline(self):
        self.assertEqual(L20_SPEC.compute_capability, "8.9")
        self.assertEqual(L20_SPEC.architecture, "Ada")
        self.assertEqual(L20_SPEC.vram_gb, 48.0)
        self.assertEqual(L20_SPEC.memory_bandwidth_gbps, 864.0)

    def test_roofline_balance_is_high_for_fp16(self):
        balance = roofline_balance_flops_per_byte(L20_SPEC, "fp16")
        self.assertGreater(balance, 60.0)
        self.assertLess(balance, 100.0)

    def test_dense_and_sparse_throughput_are_not_mixed(self):
        self.assertEqual(L20_SPEC.fp16_tflops, 59.8)
        self.assertEqual(L20_SPEC.fp16_sparse_tflops, 119.5)
        self.assertEqual(L20_SPEC.fp8_tflops, 119.5)
        self.assertEqual(L20_SPEC.fp8_sparse_tflops, 239.0)

    def test_low_intensity_operator_is_memory_bound(self):
        self.assertEqual(classify_roofline(1.0, "fp16"), "memory_bound")


if __name__ == "__main__":
    unittest.main()
