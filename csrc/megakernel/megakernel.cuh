#pragma once

#include "kittens.cuh"
#include "pyutils/torchutils.cuh"

#include "../mxfp8.cuh"
#include "../utils.cuh"

#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/zeros.h>
#include <cmath>
#include <cstring>

using namespace kittens;

enum class RoutedPrecision {
    BF16,
    MXFP8,
};

template <int NUM_DEVICES, RoutedPrecision ROUTED_PRECISION = RoutedPrecision::MXFP8,
          bool ACCUMULATE_WGRAD = false, bool BF16_MAIN_GRAD = false,
          bool HAS_SHARED = true>
struct dispatch_mlp_swiglu_combiner {

static constexpr bool USE_MXFP8 = ROUTED_PRECISION == RoutedPrecision::MXFP8;
static_assert(ACCUMULATE_WGRAD || !BF16_MAIN_GRAD,
              "BF16_MAIN_GRAD is only meaningful for fused wgrad accumulation");

// No TMA descriptors or device storage for the absent shared branch.
template <typename Layout>
using shared_gl = std::conditional_t<HAS_SHARED, Layout, gl<bf16, 1, 1, -1, -1>>;

// Undefined tensors represent absent shared storage on the host only.
static at::Tensor shared_empty(at::IntArrayRef shape, const at::TensorOptions &options) {
    if constexpr (HAS_SHARED) return at::empty(shape, options);
    return {};
}
static at::Tensor shared_empty_like(const at::Tensor &tensor) {
    if constexpr (HAS_SHARED) return at::empty_like(tensor);
    return {};
}
template <typename Layout>
static shared_gl<Layout> shared_layout(const at::Tensor &tensor) {
    if constexpr (HAS_SHARED) return kittens::py::tensor_to_gl<Layout>(tensor);
    else return {tensor.defined() ? reinterpret_cast<bf16 *>(tensor.data_ptr()) : nullptr,
                 nullptr, nullptr, tensor.defined() ? static_cast<size_t>(tensor.size(-2)) : 0,
                 tensor.defined() ? static_cast<size_t>(tensor.size(-1)) : 0};
}

static void validate_situ(std::optional<float> limit, float beta1, float beta2) {
    if (beta1 == 0.0f && beta2 == 0.0f) return;
    TORCH_CHECK(!limit.has_value() &&
                std::isfinite(beta1) && std::isfinite(beta2) && beta1 > 0.0f && beta2 > 0.0f,
                "MoK: SiTU needs positive finite betas and no SwiGLU clamp");
}

#include "types.cuh"

#include "utils.cuh"
#include "dispatch_combine.cuh"
#include "swiglu.cuh"
#include "grouped_gemm.cuh"

#include "forward.cuh"
#include "backward.cuh"
#include "recompute_forward_context.cuh"

}; // struct dispatch_mlp_swiglu_combiner
