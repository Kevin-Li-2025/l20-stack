#include <torch/extension.h>
#include <torch/library.h>

torch::Tensor l20_paged_decode_cuda(
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor block_table,
    torch::Tensor seq_lens);

torch::Tensor l20_paged_decode_split_cuda(
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor block_table,
    torch::Tensor seq_lens,
    int64_t max_seq_len,
    int64_t split_size);

void l20_paged_decode_split_out_cuda(
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor block_table,
    torch::Tensor seq_lens,
    torch::Tensor partial_output,
    torch::Tensor partial_max,
    torch::Tensor partial_sum,
    torch::Tensor output,
    int64_t max_seq_len,
    int64_t split_size);

void l20_paged_decode_split_indices_out_cuda(
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor page_indptr,
    torch::Tensor page_indices,
    torch::Tensor seq_lens,
    torch::Tensor partial_output,
    torch::Tensor partial_max,
    torch::Tensor partial_sum,
    torch::Tensor output,
    int64_t max_seq_len,
    int64_t split_size);

void l20_paged_decode_fp8_e4m3_split_out_cuda(
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor block_table,
    torch::Tensor seq_lens,
    torch::Tensor partial_output,
    torch::Tensor partial_max,
    torch::Tensor partial_sum,
    torch::Tensor output,
    double k_scale,
    double v_scale,
    int64_t max_seq_len,
    int64_t split_size);

torch::Tensor l20_paged_decode_split_out_dispatch(
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor block_table,
    torch::Tensor seq_lens,
    torch::Tensor partial_output,
    torch::Tensor partial_max,
    torch::Tensor partial_sum,
    torch::Tensor output,
    int64_t max_seq_len,
    int64_t split_size) {
  l20_paged_decode_split_out_cuda(
      query,
      key_cache,
      value_cache,
      block_table,
      seq_lens,
      partial_output,
      partial_max,
      partial_sum,
      output,
      max_seq_len,
      split_size);
  return output;
}

torch::Tensor l20_paged_decode_fp8_e4m3_split_out_dispatch(
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor block_table,
    torch::Tensor seq_lens,
    torch::Tensor partial_output,
    torch::Tensor partial_max,
    torch::Tensor partial_sum,
    torch::Tensor output,
    double k_scale,
    double v_scale,
    int64_t max_seq_len,
    int64_t split_size) {
  l20_paged_decode_fp8_e4m3_split_out_cuda(
      query,
      key_cache,
      value_cache,
      block_table,
      seq_lens,
      partial_output,
      partial_max,
      partial_sum,
      output,
      k_scale,
      v_scale,
      max_seq_len,
      split_size);
  return output;
}

TORCH_LIBRARY(l20_stack, module) {
  module.def(
      "paged_decode_split_out("
      "Tensor query, Tensor key_cache, Tensor value_cache, "
      "Tensor block_table, Tensor seq_lens, "
      "Tensor(a!) partial_output, Tensor(b!) partial_max, "
      "Tensor(c!) partial_sum, Tensor(d!) output, "
      "int max_seq_len, int split_size) -> Tensor(d!)");
  module.def(
      "paged_decode_fp8_e4m3_split_out("
      "Tensor query, Tensor key_cache, Tensor value_cache, "
      "Tensor block_table, Tensor seq_lens, "
      "Tensor(a!) partial_output, Tensor(b!) partial_max, "
      "Tensor(c!) partial_sum, Tensor(d!) output, "
      "float k_scale, float v_scale, "
      "int max_seq_len, int split_size) -> Tensor(d!)");
}

TORCH_LIBRARY_IMPL(l20_stack, CUDA, module) {
  module.impl("paged_decode_split_out", &l20_paged_decode_split_out_dispatch);
  module.impl(
      "paged_decode_fp8_e4m3_split_out",
      &l20_paged_decode_fp8_e4m3_split_out_dispatch);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("paged_decode", &l20_paged_decode_cuda);
  module.def("paged_decode_split", &l20_paged_decode_split_cuda);
  module.def("paged_decode_split_out", &l20_paged_decode_split_out_cuda);
  module.def(
      "paged_decode_split_indices_out",
      &l20_paged_decode_split_indices_out_cuda);
  module.def(
      "paged_decode_fp8_e4m3_split_out",
      &l20_paged_decode_fp8_e4m3_split_out_cuda);
}
