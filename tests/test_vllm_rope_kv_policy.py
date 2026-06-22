import ast
import unittest
from pathlib import Path


def load_policy():
    source = Path("integrations/vllm/l20_rope_kv.py").read_text(encoding="utf-8")
    module = ast.parse(source)
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "l20_rope_kv_num_warps"
    )
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<policy>", "exec"), namespace)
    return namespace["l20_rope_kv_num_warps"]


class L20VllmRopeKvPolicyTest(unittest.TestCase):
    def test_head_dim_64_policy(self):
        policy = load_policy()
        self.assertEqual(policy(64, 64), 2)
        self.assertEqual(policy(96, 64), 1)

    def test_head_dim_128_policy(self):
        policy = load_policy()
        self.assertEqual(policy(1, 128), 4)
        self.assertEqual(policy(32, 128), 2)
        self.assertEqual(policy(128, 128), 1)

    def test_head_dim_256_policy(self):
        policy = load_policy()
        self.assertEqual(policy(1, 256), 2)
        self.assertEqual(policy(512, 256), 4)

    def test_rejects_unsupported_head_dim(self):
        with self.assertRaises(RuntimeError):
            load_policy()(1, 512)


if __name__ == "__main__":
    unittest.main()
