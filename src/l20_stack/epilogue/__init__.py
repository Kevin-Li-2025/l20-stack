"""Epilogue-boundary analysis helpers for L20 serving experiments."""

from l20_stack.epilogue.compare import BoundaryImpact, build_boundary_impacts
from l20_stack.epilogue.flash_sampling import (
    FlashSamplingGateDecision,
    FlashSamplingLaunchPolicy,
    FlashSamplingRequest,
    flash_sampling_gate_reasons,
    flash_sampling_launch_policy,
    plan_flash_sampling_epilogue,
    should_use_flash_sampling_epilogue,
)
from l20_stack.epilogue.logits_boundary import (
    LogitsBoundaryBudget,
    load_logits_boundary_budget,
)
from l20_stack.epilogue.intervention import (
    CONTINUE_EPILOGUE_PROTOTYPE,
    DO_NOT_CLAIM_WIN,
    NEEDS_MORE_RUNS,
    render_logits_boundary_ab_markdown,
    summarize_logits_boundary_ab,
)
from l20_stack.epilogue.sampler_epilogue import (
    SamplerConfig,
    SamplerOptimizationPlan,
    plan_sampler_optimization,
    sampler_gate_reasons,
)

__all__ = [
    "BoundaryImpact",
    "CONTINUE_EPILOGUE_PROTOTYPE",
    "DO_NOT_CLAIM_WIN",
    "FlashSamplingGateDecision",
    "FlashSamplingLaunchPolicy",
    "FlashSamplingRequest",
    "flash_sampling_gate_reasons",
    "flash_sampling_launch_policy",
    "plan_flash_sampling_epilogue",
    "should_use_flash_sampling_epilogue",
    "LogitsBoundaryBudget",
    "NEEDS_MORE_RUNS",
    "SamplerConfig",
    "SamplerOptimizationPlan",
    "build_boundary_impacts",
    "load_logits_boundary_budget",
    "plan_sampler_optimization",
    "render_logits_boundary_ab_markdown",
    "sampler_gate_reasons",
    "summarize_logits_boundary_ab",
]
