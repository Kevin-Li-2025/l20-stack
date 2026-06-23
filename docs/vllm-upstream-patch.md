# vLLM Upstream Patch

The repository now carries an apply-ready vLLM patch based on tag `v0.23.0`
and commit `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`:

```text
integrations/vllm/vllm-v0.23.0-l20-paged-decode.patch
```

The corresponding fork commit is
`6efb66d4eedf6b410abc8e74db027ee8dca2d8ff`.

## Patch Scope

- add the SM89 paged-decode CUDA source to vLLM's `_C` extension;
- register `_C::l20_paged_decode_split_out`;
- expose the op through `vllm._custom_ops` with FakeTensor support;
- extend native FlashInfer decode metadata with block tables and sequence
  lengths;
- dispatch only on SM89, FP16, head dimension 128, page size 16, measured
  12Q/2KV or 16Q/8KV shapes, eager execution, and the existing conservative
  batch/context gate;
- preserve FlashInfer for every unsupported shape and CUDA Graph capture;
- add four randomized correctness cases and one FakeTensor test.

## L20 Validation

The `_C` namespace fragment compiled and registered on the L20. The upstream
test file passes `5/5` cases, and Qwen2.5-Coder-1.5B completes a real
eight-token request through the source-tree FlashInfer backend.

The full editable vLLM wheel now builds with the CUDA 13.0 components installed
inside the isolated vLLM environment. The system CUDA 12 compiler must not be
used. NVIDIA wheel components for NVCC, CRT, NVVM, runtime, and CCCL must all
remain on the CUDA 13.0 release line. The reproducible environment setup is in
`scripts/build_vllm_cuda13_l20.sh`.

The CUDA 13 Compute Sanitizer package reports zero memcheck errors across all
four numerical GPU cases. The fully built wheel also completes the
Qwen2.5-Coder-1.5B eight-token FlashInfer eager service request.

The remote host has intermittent GitHub connectivity. CUTLASS and
vLLM FlashAttention are therefore supplied through `VLLM_CUTLASS_SRC_DIR` and
`VLLM_FLASH_ATTN_SRC_DIR` instead of being fetched during CMake configuration.

Apply the patch from a clean vLLM `v0.23.0` checkout:

```bash
git apply /path/to/vllm-v0.23.0-l20-paged-decode.patch
```
