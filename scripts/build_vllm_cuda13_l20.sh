#!/usr/bin/env bash
set -euo pipefail

source_tree=${VLLM_SOURCE_TREE:-"$HOME/vllm-l20-upstream"}
python_prefix=${VIRTUAL_ENV:?activate the target vLLM virtualenv first}
cuda_root=${CUDA_ROOT:-"$python_prefix/lib/python3.12/site-packages/nvidia/cu13"}
cutlass_root=${VLLM_CUTLASS_SRC_DIR:-"$HOME/deps/cutlass-v4.4.2"}
flash_attn_root=${VLLM_FLASH_ATTN_SRC_DIR:-"$source_tree/.deps/vllm-flash-attn-src"}

export CUDA_HOME="$cuda_root"
export CUDAToolkit_ROOT="$cuda_root"
export CUDACXX="$cuda_root/bin/nvcc"
export PATH="$cuda_root/bin:$PATH"
export LD_LIBRARY_PATH="$cuda_root/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export VLLM_CUTLASS_SRC_DIR="$cutlass_root"
export VLLM_FLASH_ATTN_SRC_DIR="$flash_attn_root"
export MAX_JOBS=${MAX_JOBS:-4}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.9}

"$CUDACXX" --version
test -f "$cutlass_root/CMakeLists.txt"
test -f "$flash_attn_root/CMakeLists.txt"

cd "$source_tree"
python -m pip install -e . --no-build-isolation --no-deps
