import importlib.util
import unittest
from pathlib import Path


def load_profile_script():
    path = Path("scripts/profile_vllm_l20_rope_kv.py")
    spec = importlib.util.spec_from_file_location("profile_vllm_l20_rope_kv", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_ncu_summary_script():
    path = Path("scripts/summarize_ncu_profile.py")
    spec = importlib.util.spec_from_file_location("summarize_ncu_profile", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_nsys_family_script():
    path = Path("scripts/summarize_nsys_kernel_families.py")
    spec = importlib.util.spec_from_file_location("summarize_nsys_kernel_families", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class L20KernelProfileTest(unittest.TestCase):
    def test_register_limited_occupancy(self):
        module = load_profile_script()
        result = module.theoretical_occupancy(num_warps=4, num_regs=128, shared_bytes=0)
        self.assertEqual(result["limiting_resource"], "register_blocks")
        self.assertEqual(result["resident_blocks_per_sm"], 4)
        self.assertAlmostEqual(result["theoretical_occupancy_pct"], 33.33)

    def test_measured_resource_shape_is_warp_limited(self):
        module = load_profile_script()
        result = module.theoretical_occupancy(num_warps=4, num_regs=24, shared_bytes=0)
        self.assertEqual(result["limiting_resource"], "warp_blocks")
        self.assertEqual(result["resident_warps_per_sm"], 48)

    def test_ncu_summary_extracts_roofline_metrics(self):
        module = load_ncu_summary_script()
        rows = module.read_ncu_csv(Path("tests/fixtures/ncu_raw_sample.csv"))
        summary = module.summarize_kernel("_l20_test_kernel", rows["_l20_test_kernel"])
        self.assertEqual(summary["kernel_name"], "_l20_test_kernel")
        self.assertAlmostEqual(summary["achieved_memory_bandwidth_gbps"], 864.0)
        self.assertAlmostEqual(summary["arithmetic_intensity_flops_per_byte"], 1.0)
        self.assertEqual(summary["roofline_class"], "memory_bound")
        self.assertEqual(summary["memory_bandwidth_utilization_pct"], 100.0)
        self.assertEqual(summary["active_warps_pct"], 65.0)
        self.assertEqual(summary["stall_long_scoreboard_pct"], 31.0)
        self.assertAlmostEqual(summary["sector_excess_ratio_l1_over_l2"], 2.0)

    def test_ncu_summary_extracts_wide_raw_csv(self):
        module = load_ncu_summary_script()
        rows = module.read_ncu_csv(Path("tests/fixtures/ncu_wide_sample.csv"))
        summary = module.summarize_kernel("_l20_wide_kernel", rows["_l20_wide_kernel"])
        self.assertAlmostEqual(summary["duration_ns"], 29760.0)
        self.assertAlmostEqual(summary["dram_bytes"], 15_166_208.0)
        self.assertAlmostEqual(summary["achieved_memory_bandwidth_gbps"], 509.617204)
        self.assertAlmostEqual(summary["memory_bandwidth_utilization_pct"], 59.129471)
        self.assertAlmostEqual(summary["l2_throughput_utilization_pct"], 33.646515)
        self.assertAlmostEqual(summary["l1_sector_hit_rate_pct"], 70.519970)
        self.assertAlmostEqual(summary["active_warps_pct"], 30.170858)
        self.assertAlmostEqual(summary["registers_per_thread"], 24.0)
        self.assertAlmostEqual(summary["stall_long_scoreboard_pct"], 24.25)
        self.assertAlmostEqual(summary["sector_excess_ratio_l1_over_l2"], 2.0)

    def test_profile_kernel_wrapper_exports_dashboard_artifacts(self):
        source = Path("scripts/profile_kernel.sh").read_text()
        self.assertIn("--section SpeedOfLight", source)
        self.assertIn("--section MemoryWorkloadAnalysis", source)
        self.assertIn("--section WarpStateStats", source)
        self.assertIn("scripts/summarize_ncu_profile.py", source)
        self.assertIn("summary_python=\"${PYTHON:-python3}\"", source)
        self.assertIn("--markdown-output", source)
        rope_source = Path("scripts/profile_vllm_l20_rope_kv_ncu.sh").read_text()
        self.assertIn("scripts/profile_kernel.sh", rope_source)
        self.assertIn("regex:_l20_.*rope_kv_kernel", rope_source)

    def test_nsys_family_classifier_tracks_serving_boundaries(self):
        module = load_nsys_family_script()
        self.assertEqual(
            module.classify_kernel("void flashinfer::sampling::TopPSamplingFromProbKernel"),
            "flashinfer_sampling",
        )
        self.assertEqual(module.classify_kernel("_topk_topp_kernel"), "sampler_other")
        self.assertEqual(module.classify_kernel("_temperature_kernel"), "sampler_other")
        self.assertEqual(module.classify_kernel("_min_p_kernel"), "sampler_other")
        self.assertEqual(module.classify_kernel("_penalties_kernel"), "sampler_other")
        self.assertEqual(
            module.classify_kernel(
                "void flashinfer::BatchPrefillWithPagedKVCacheKernel<T>(T)"
            ),
            "flashinfer_attention",
        )
        self.assertEqual(
            module.classify_kernel("void cutlass::Kernel2<cutlass_80_tensorop>(T)"),
            "cutlass_or_cublas_gemm",
        )
        self.assertEqual(
            module.classify_kernel("std::enable_if<!T7, void>::type internal::gemvx::kernel"),
            "cublas_gemv",
        )
        self.assertEqual(module.classify_kernel("_l20_qk_norm_rope_kv_kernel"), "custom_l20")
        self.assertEqual(module.classify_api("cudaEventSynchronize"), "sync")
        self.assertEqual(module.classify_api("cudaMemcpyAsync"), "memcpy")
        self.assertEqual(module.classify_api("cudaGraphLaunch_v10000"), "graph")


if __name__ == "__main__":
    unittest.main()
