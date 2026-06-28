#!/usr/bin/env python3
"""Install the L20 Q/K norm + RoPE + KV-cache fusion into vLLM.

This is an experimental Qwen3 serving hook.  It keeps the default vLLM path
unchanged unless ``VLLM_L20_QK_ROPE_KV=1`` is set.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import vllm


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        return text
    if old not in text:
        raise RuntimeError(f"cannot find patch point: {label}")
    return text.replace(old, new, 1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uninstall", action="store_true")
    return parser.parse_args()


def patch_attention(source: str) -> str:
    source = replace_once(
        source,
        """    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        # For some alternate attention backends like MLA the attention output
""",
        """    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache_dummy_dep: torch.Tensor | None = None,
        skip_kv_cache_update: bool = False,
        # For some alternate attention backends like MLA the attention output
""",
        "Attention.forward signature",
    )
    condition = (
        "                not self.attn_backend.forward_includes_kv_cache_update\n"
        "                and self.kv_sharing_target_layer_name is None\n"
        "                and key is not None\n"
    )
    guarded_condition = (
        "                not self.attn_backend.forward_includes_kv_cache_update\n"
        "                and self.kv_sharing_target_layer_name is None\n"
        "                and not skip_kv_cache_update\n"
        "                and key is not None\n"
    )
    if condition in source:
        source = source.replace(condition, guarded_condition)
    reset_line = "        kv_cache_dummy_dep = None\n"
    if reset_line in source:
        source = source.replace(reset_line, "", 1)
    return source


def patch_qwen3(source: str) -> str:
    source = replace_once(
        source,
        "from collections.abc import Iterable\n",
        "from collections.abc import Iterable\nimport os\nfrom pathlib import Path\n",
        "qwen3 env imports",
    )
    source = replace_once(
        source,
        "from vllm.transformers_utils.config import set_default_rope_theta\n",
        (
            "from vllm.transformers_utils.config import set_default_rope_theta\n"
            "from vllm.utils.torch_utils import (\n"
            "    LayerNameType,\n"
            "    _resolve_layer_name,\n"
            "    direct_register_custom_op,\n"
            ")\n"
        ),
        "qwen3 custom op imports",
    )
    if 'op_name="l20_qk_norm_rope_kv_cache_update"' not in source:
        source = replace_once(
            source,
            "logger = init_logger(__name__)\n",
            '''logger = init_logger(__name__)

try:
    from vllm.v1.attention.ops.l20_qk_norm_rope_kv import (
        l20_qk_norm_rope_and_cache,
    )
except Exception:  # pragma: no cover - optional experimental install path
    l20_qk_norm_rope_and_cache = None


_L20_QK_ROPE_KV_TRACE_COUNT = 0
_L20_QK_ROPE_KV_ENABLED = os.environ.get("VLLM_L20_QK_ROPE_KV", "0") == "1"
_L20_QK_ROPE_KV_STRICT = os.environ.get("VLLM_L20_QK_ROPE_KV_STRICT", "0") == "1"


def _l20_qk_rope_kv_enabled() -> bool:
    return _L20_QK_ROPE_KV_ENABLED


def _l20_qk_rope_kv_strict() -> bool:
    return _L20_QK_ROPE_KV_STRICT


def _l20_qk_rope_kv_trace(layer_name: str, status: str, detail: str) -> None:
    if torch.compiler.is_compiling():
        return
    path = os.environ.get("VLLM_L20_QK_ROPE_KV_TRACE")
    if not path:
        return
    global _L20_QK_ROPE_KV_TRACE_COUNT
    limit = int(os.environ.get("VLLM_L20_QK_ROPE_KV_TRACE_LIMIT", "256"))
    if _L20_QK_ROPE_KV_TRACE_COUNT >= limit:
        return
    _L20_QK_ROPE_KV_TRACE_COUNT += 1
    try:
        trace_path = Path(path)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        with trace_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{status}\\t{layer_name}\\t{detail}\\n")
    except Exception:
        pass


def l20_qk_norm_rope_kv_cache_update(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    layer_name: LayerNameType,
    num_q_heads: int,
    num_kv_heads: int,
    eps: float,
) -> torch.Tensor:
    """Mutate packed QKV in place and write the vLLM paged KV cache."""
    if l20_qk_norm_rope_and_cache is None:
        raise RuntimeError("l20_qk_norm_rope_kv is not installed")
    from vllm.model_executor.layers.attention.attention import get_attention_context

    resolved_layer_name = _resolve_layer_name(layer_name)
    _, _, kv_cache, layer_slot_mapping = get_attention_context(resolved_layer_name)
    if layer_slot_mapping is not None:
        l20_qk_norm_rope_and_cache(
            qkv,
            positions.flatten(),
            cos_sin_cache,
            q_weight,
            k_weight,
            kv_cache[:, 0],
            kv_cache[:, 1],
            layer_slot_mapping,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            eps=eps,
        )
    return torch.empty((), device=qkv.device, dtype=qkv.dtype)


def l20_qk_norm_rope_kv_cache_update_fake(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    layer_name: LayerNameType,
    num_q_heads: int,
    num_kv_heads: int,
    eps: float,
) -> torch.Tensor:
    return torch.empty((), device=qkv.device, dtype=qkv.dtype)


direct_register_custom_op(
    op_name="l20_qk_norm_rope_kv_cache_update",
    op_func=l20_qk_norm_rope_kv_cache_update,
    mutates_args=["qkv"],
    fake_impl=l20_qk_norm_rope_kv_cache_update_fake,
)
''',
            "qwen3 L20 custom op block",
        )
    source = source.replace(
        '''_L20_QK_ROPE_KV_TRACE_COUNT = 0


def _l20_qk_rope_kv_enabled() -> bool:
    return os.environ.get("VLLM_L20_QK_ROPE_KV", "0") == "1"


def _l20_qk_rope_kv_strict() -> bool:
    return os.environ.get("VLLM_L20_QK_ROPE_KV_STRICT", "0") == "1"
''',
        '''_L20_QK_ROPE_KV_TRACE_COUNT = 0
_L20_QK_ROPE_KV_ENABLED = os.environ.get("VLLM_L20_QK_ROPE_KV", "0") == "1"
_L20_QK_ROPE_KV_STRICT = os.environ.get("VLLM_L20_QK_ROPE_KV_STRICT", "0") == "1"


def _l20_qk_rope_kv_enabled() -> bool:
    return _L20_QK_ROPE_KV_ENABLED


def _l20_qk_rope_kv_strict() -> bool:
    return _L20_QK_ROPE_KV_STRICT
''',
        1,
    )
    source = replace_once(
        source,
        '''    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # Add qk-norm
''',
        '''    def _l20_qk_norm_rope_kv_forward(
        self,
        positions: torch.Tensor,
        qkv: torch.Tensor,
    ) -> torch.Tensor | None:
        layer_name = self.attn.layer_name
        reason = self._l20_qk_norm_rope_kv_guard(qkv)
        if reason is not None:
            _l20_qk_rope_kv_trace(layer_name, "fallback", reason)
            if _l20_qk_rope_kv_strict():
                raise RuntimeError(f"L20 QK/RoPE/KV hook rejected: {reason}")
            return None
        try:
            cos_sin_cache = self.rotary_emb._match_cos_sin_cache_dtype(qkv)
            dep = torch.ops.vllm.l20_qk_norm_rope_kv_cache_update(
                qkv,
                positions,
                self.q_norm.weight,
                self.k_norm.weight,
                cos_sin_cache,
                layer_name,
                self.num_heads,
                self.num_kv_heads,
                self.q_norm.variance_epsilon,
            )
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            attn_output = self.attn(
                q,
                k,
                v,
                kv_cache_dummy_dep=dep,
                skip_kv_cache_update=True,
            )
            output, _ = self.o_proj(attn_output)
            if not torch.compiler.is_compiling():
                _l20_qk_rope_kv_trace(layer_name, "hit", "runtime")
            return output
        except Exception as err:
            detail = f"{type(err).__name__}: {err}"
            _l20_qk_rope_kv_trace(layer_name, "fallback", detail)
            if _l20_qk_rope_kv_strict():
                raise
            return None

    def _l20_qk_norm_rope_kv_guard(self, qkv: torch.Tensor) -> str | None:
        if l20_qk_norm_rope_and_cache is None:
            return "kernel_not_installed"
        if not qkv.is_cuda:
            return "non_cuda_qkv"
        if torch.cuda.get_device_capability(qkv.device) != (8, 9):
            return "non_sm89_device"
        if qkv.dtype not in (torch.float16, torch.bfloat16):
            return f"unsupported_dtype={qkv.dtype}"
        if qkv.ndim != 2 or not qkv.is_contiguous():
            return "qkv_not_contiguous_2d"
        if self.head_dim != 128:
            return f"head_dim={self.head_dim}"
        if getattr(self.rotary_emb, "rotary_dim", self.head_dim) != self.head_dim:
            return "partial_rotary_dim"
        if not getattr(self.rotary_emb, "is_neox_style", True):
            return "non_neox_rope"
        if getattr(self.attn, "kv_cache_dtype", "auto") != "auto":
            return f"kv_cache_dtype={getattr(self.attn, 'kv_cache_dtype', None)}"
        expected = self.q_size + 2 * self.kv_size
        if qkv.shape[-1] != expected:
            return f"packed_width={qkv.shape[-1]} expected={expected}"
        return None

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        if _l20_qk_rope_kv_enabled():
            fused_output = self._l20_qk_norm_rope_kv_forward(positions, qkv)
            if fused_output is not None:
                return fused_output
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # Add qk-norm
''',
        "qwen3 attention hook",
    )
    return source


def main() -> int:
    args = parse_args()
    package = Path(next(iter(vllm.__path__)))
    integration = Path(__file__).resolve().parent
    targets = {
        "attention": package / "model_executor" / "layers" / "attention" / "attention.py",
        "qwen3": package / "model_executor" / "models" / "qwen3.py",
        "kernel": package / "v1" / "attention" / "ops" / "l20_qk_norm_rope_kv.py",
    }
    if args.uninstall:
        for name in ("attention", "qwen3"):
            backup = targets[name].with_suffix(targets[name].suffix + ".l20-qk-kv-backup")
            if backup.exists():
                shutil.copy2(backup, targets[name])
        targets["kernel"].unlink(missing_ok=True)
        return 0

    for name in ("attention", "qwen3"):
        backup = targets[name].with_suffix(targets[name].suffix + ".l20-qk-kv-backup")
        if not backup.exists():
            shutil.copy2(targets[name], backup)

    shutil.copy2(integration / "l20_qk_norm_rope_kv.py", targets["kernel"])

    attention = targets["attention"].read_text(encoding="utf-8")
    targets["attention"].write_text(patch_attention(attention), encoding="utf-8")

    qwen3 = targets["qwen3"].read_text(encoding="utf-8")
    targets["qwen3"].write_text(patch_qwen3(qwen3), encoding="utf-8")

    print(package)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
