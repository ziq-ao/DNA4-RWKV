#include <torch/extension.h>

// 保持原有的入口函数名不变
void wkv_fp32_v2_cuda(
    int B,
    int T,
    int C,
    int H,
    int mode,
    torch::Tensor state,
    torch::Tensor r,
    torch::Tensor w,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor y);

namespace {

// =================================================================
// 锁死 IO 类型为 BF16，摒弃宏判断
// =================================================================
constexpr auto IO_DTYPE = torch::kBFloat16;
constexpr const char* IO_DTYPE_NAME = "bf16";

// state 依然保持 FP32 校验，用于内部高精度累加
void check_float_cuda_contig(const torch::Tensor& x, const char* name) {
  TORCH_CHECK(x.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(x.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(x.scalar_type() == torch::kFloat32, name, " must be fp32");
}

// 其余所有的输入输出必须是 BF16
void check_io_cuda_contig(const torch::Tensor& x, const char* name) {
  TORCH_CHECK(x.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(x.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(x.scalar_type() == IO_DTYPE, name, " must be ", IO_DTYPE_NAME);
}

void check_inputs(
    int64_t B,
    int64_t T,
    int64_t C,
    int64_t H,
    torch::Tensor state,
    torch::Tensor r,
    torch::Tensor w,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor y) {
  TORCH_CHECK(C == H * 64, "only head size 64 is supported");
  check_float_cuda_contig(state, "state"); // FP32
  check_io_cuda_contig(r, "r");            // BF16
  check_io_cuda_contig(w, "w");
  check_io_cuda_contig(k, "k");
  check_io_cuda_contig(v, "v");
  check_io_cuda_contig(a, "a");
  check_io_cuda_contig(b, "b");
  check_io_cuda_contig(y, "y");
  TORCH_CHECK(state.dim() == 4 && state.size(0) == B && state.size(1) == H && state.size(2) == 64 && state.size(3) == 64,
              "state must have shape [B,H,64,64]");
  TORCH_CHECK(r.sizes() == w.sizes() && r.sizes() == k.sizes() && r.sizes() == v.sizes() &&
              r.sizes() == a.sizes() && r.sizes() == b.sizes() && r.sizes() == y.sizes(),
              "r,w,k,v,a,b,y shape mismatch");
  TORCH_CHECK(r.dim() == 3 && r.size(0) == B && r.size(1) == T && r.size(2) == C,
              "r must have shape [B,T,C]");
}

}  // namespace

void forward(
    int64_t B,
    int64_t T,
    int64_t C,
    int64_t H,
    torch::Tensor state,
    torch::Tensor r,
    torch::Tensor w,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor y) {
  check_inputs(B, T, C, H, state, r, w, k, v, a, b, y);
  wkv_fp32_v2_cuda(
      static_cast<int>(B),
      static_cast<int>(T),
      static_cast<int>(C),
      static_cast<int>(H),
      0,
      state, r, w, k, v, a, b, y);
}

void forward_seq(
    int64_t B,
    int64_t T,
    int64_t C,
    int64_t H,
    torch::Tensor state,
    torch::Tensor r,
    torch::Tensor w,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor y) {
  check_inputs(B, T, C, H, state, r, w, k, v, a, b, y);
  wkv_fp32_v2_cuda(static_cast<int>(B), static_cast<int>(T), static_cast<int>(C), static_cast<int>(H), 1, state, r, w, k, v, a, b, y);
}

void forward_small(
    int64_t B,
    int64_t T,
    int64_t C,
    int64_t H,
    torch::Tensor state,
    torch::Tensor r,
    torch::Tensor w,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor y) {
  check_inputs(B, T, C, H, state, r, w, k, v, a, b, y);
  wkv_fp32_v2_cuda(static_cast<int>(B), static_cast<int>(T), static_cast<int>(C), static_cast<int>(H), 2, state, r, w, k, v, a, b, y);
}

void forward_block(
    int64_t B,
    int64_t T,
    int64_t C,
    int64_t H,
    torch::Tensor state,
    torch::Tensor r,
    torch::Tensor w,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor y) {
  check_inputs(B, T, C, H, state, r, w, k, v, a, b, y);
  wkv_fp32_v2_cuda(static_cast<int>(B), static_cast<int>(T), static_cast<int>(C), static_cast<int>(H), 3, state, r, w, k, v, a, b, y);
}

// =================================================================
// 还原模块注册名，完全兼容你 Python 端的已有调用
// =================================================================
TORCH_LIBRARY(rwkv7_wkv_fp32_v2, m) {
  m.def("forward(int B, int T, int C, int H, Tensor(a!) state, Tensor r, Tensor w, Tensor k, Tensor v, Tensor a, Tensor b, Tensor(a!) y) -> ()");
  m.def("forward_seq(int B, int T, int C, int H, Tensor(a!) state, Tensor r, Tensor w, Tensor k, Tensor v, Tensor a, Tensor b, Tensor(a!) y) -> ()");
  m.def("forward_small(int B, int T, int C, int H, Tensor(a!) state, Tensor r, Tensor w, Tensor k, Tensor v, Tensor a, Tensor b, Tensor(a!) y) -> ()");
  m.def("forward_block(int B, int T, int C, int H, Tensor(a!) state, Tensor r, Tensor w, Tensor k, Tensor v, Tensor a, Tensor b, Tensor(a!) y) -> ()");
}

TORCH_LIBRARY_IMPL(rwkv7_wkv_fp32_v2, CUDA, m) {
  m.impl("forward", &forward);
  m.impl("forward_seq", &forward_seq);
  m.impl("forward_small", &forward_small);
  m.impl("forward_block", &forward_block);
}