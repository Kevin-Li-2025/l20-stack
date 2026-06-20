import unittest

from l20_stack.operators import (
    OperatorShape,
    OperatorTarget,
    plan_operator,
    rope_kv_minimum_bytes,
)
from l20_stack.ops.triton_rope_kv import rope_kv_launch_config


class RopeKvPlanTest(unittest.TestCase):
    def test_rope_kv_launch_targets_l20_decode(self):
        config = rope_kv_launch_config(128)

        self.assertEqual(config.sm_target, "sm_89")
        self.assertEqual(config.block_size, 128)
        self.assertEqual(config.num_warps, 4)

    def test_rope_kv_fusion_reduces_minimum_traffic(self):
        shape = OperatorShape(rows=32 * 8, hidden_size=128, dtype_bytes=2)
        fused = rope_kv_minimum_bytes(shape, fused=True)
        unfused = rope_kv_minimum_bytes(shape, fused=False)
        plan = plan_operator(OperatorTarget(name="rope_kv_cache_write", shape=shape))

        self.assertLess(fused, unfused)
        self.assertEqual(plan.priority, 2)
        self.assertEqual(plan.roofline_class, "memory_bound")
        self.assertAlmostEqual(plan.launch["minimum_traffic_reduction_pct"], 33.33, places=2)

    def test_invalid_rotary_dim_is_rejected(self):
        with self.assertRaises(ValueError):
            rope_kv_launch_config(128, rotary_dim=127)


if __name__ == "__main__":
    unittest.main()
