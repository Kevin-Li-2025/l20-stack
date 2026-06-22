#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cmath>

namespace {

__inline__ __device__ float warp_sum(float value) {
  for (int offset = 16; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

__global__ void paged_decode_kernel(
    const half* query,
    const half* key_cache,
    const half* value_cache,
    const int* block_table,
    const int* seq_lens,
    half* output,
    int num_q_heads,
    int num_kv_heads,
    int page_size,
    int max_pages) {
  const int batch = blockIdx.y;
  const int q_head = blockIdx.x;
  const int kv_head = q_head / (num_q_heads / num_kv_heads);
  const int thread = threadIdx.x;
  const int lane = thread & 31;
  const int warp = thread >> 5;
  __shared__ float scores[16];
  __shared__ float probabilities[16];
  __shared__ float alpha_shared;
  __shared__ float running_max_shared;
  __shared__ float running_sum_shared;

  const int q_base = (batch * num_q_heads + q_head) * 128;
  const int pair0 = lane * 2;
  const int pair1 = pair0 + 64;
  const half2 q01 = *reinterpret_cast<const half2*>(query + q_base + pair0);
  const half2 q23 = *reinterpret_cast<const half2*>(query + q_base + pair1);
  const float2 q01f = __half22float2(q01);
  const float2 q23f = __half22float2(q23);
  float2 accumulator = make_float2(0.0f, 0.0f);
  const int seq_len = seq_lens[batch];
  if (thread == 0) {
    running_max_shared = -INFINITY;
    running_sum_shared = 0.0f;
  }
  __syncthreads();

  for (int tile_start = 0; tile_start < seq_len; tile_start += 16) {
#pragma unroll
    for (int warp_token = 0; warp_token < 2; ++warp_token) {
      const int token_index = warp + warp_token * 8;
      const int token = tile_start + token_index;
      float dot = 0.0f;
      if (token < seq_len) {
        const int logical_page = token / page_size;
        const int page_offset = token - logical_page * page_size;
        const int physical_page = block_table[batch * max_pages + logical_page];
        const int cache_base =
            ((physical_page * page_size + page_offset) * num_kv_heads + kv_head) *
            128;
        const half2 k01 =
            *reinterpret_cast<const half2*>(key_cache + cache_base + pair0);
        const half2 k23 =
            *reinterpret_cast<const half2*>(key_cache + cache_base + pair1);
        const float2 k01f = __half22float2(k01);
        const float2 k23f = __half22float2(k23);
        dot = q01f.x * k01f.x + q01f.y * k01f.y +
              q23f.x * k23f.x + q23f.y * k23f.y;
      }
      dot = warp_sum(dot);
      if (lane == 0) {
        scores[token_index] = token < seq_len
            ? dot * 0.08838834764831845f
            : -INFINITY;
      }
    }
    __syncthreads();
    if (thread == 0) {
      float tile_max = scores[0];
#pragma unroll
      for (int index = 1; index < 16; ++index) {
        tile_max = fmaxf(tile_max, scores[index]);
      }
      const float next_max = fmaxf(running_max_shared, tile_max);
      alpha_shared = expf(running_max_shared - next_max);
      float tile_sum = 0.0f;
#pragma unroll
      for (int index = 0; index < 16; ++index) {
        probabilities[index] = expf(scores[index] - next_max);
        tile_sum += probabilities[index];
      }
      running_sum_shared = running_sum_shared * alpha_shared + tile_sum;
      running_max_shared = next_max;
    }
    __syncthreads();
    if (thread < 64) {
      accumulator.x *= alpha_shared;
      accumulator.y *= alpha_shared;
#pragma unroll
      for (int index = 0; index < 16; ++index) {
        const int value_token = tile_start + index;
        if (value_token < seq_len) {
          const int logical_page = value_token / page_size;
          const int page_offset = value_token - logical_page * page_size;
          const int physical_page =
              block_table[batch * max_pages + logical_page];
          const int cache_offset =
              ((physical_page * page_size + page_offset) * num_kv_heads +
               kv_head) *
                  128 +
              thread * 2;
          const half2 value =
              *reinterpret_cast<const half2*>(value_cache + cache_offset);
          const float2 value_float = __half22float2(value);
          accumulator.x += probabilities[index] * value_float.x;
          accumulator.y += probabilities[index] * value_float.y;
        }
      }
    }
    __syncthreads();
  }
  if (thread < 64) {
    const half2 result = __floats2half2_rn(
        accumulator.x / running_sum_shared,
        accumulator.y / running_sum_shared);
    *reinterpret_cast<half2*>(
        output + (batch * num_q_heads + q_head) * 128 + thread * 2) = result;
  }
}

}  // namespace

torch::Tensor l20_paged_decode_cuda(
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor block_table,
    torch::Tensor seq_lens) {
  TORCH_CHECK(query.is_cuda(), "query must be CUDA");
  TORCH_CHECK(query.scalar_type() == torch::kFloat16, "FP16 only");
  TORCH_CHECK(query.dim() == 3 && query.size(2) == 128, "Q must be [B,H,128]");
  TORCH_CHECK(key_cache.dim() == 4 && key_cache.size(3) == 128, "NHD cache only");
  TORCH_CHECK(key_cache.sizes() == value_cache.sizes(), "K/V cache mismatch");
  TORCH_CHECK(block_table.scalar_type() == torch::kInt32, "int32 block table");
  TORCH_CHECK(seq_lens.scalar_type() == torch::kInt32, "int32 sequence lengths");
  const at::cuda::CUDAGuard guard(query.device());
  auto output = torch::empty_like(query);
  const dim3 grid(query.size(1), query.size(0));
  paged_decode_kernel<<<grid, 256, 0, at::cuda::getDefaultCUDAStream()>>>(
      reinterpret_cast<const half*>(query.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(key_cache.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(value_cache.data_ptr<at::Half>()),
      block_table.data_ptr<int>(),
      seq_lens.data_ptr<int>(),
      reinterpret_cast<half*>(output.data_ptr<at::Half>()),
      query.size(1),
      key_cache.size(2),
      key_cache.size(1),
      block_table.size(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
