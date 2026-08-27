#pragma once

#include "kittens.cuh"
#include "pyutils/torchutils.cuh"

#include "../mxfp8.cuh"
#include "../utils.cuh"

#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/zeros.h>
#include <cstring>

using namespace kittens;

enum class RoutedPrecision {
    BF16,
    MXFP8,
};

template <int NUM_DEVICES, RoutedPrecision ROUTED_PRECISION = RoutedPrecision::MXFP8,
          bool ACCUMULATE_WGRAD = false, bool BF16_MAIN_GRAD = false>
struct dispatch_mlp_swiglu_combiner {

static constexpr bool USE_MXFP8 = ROUTED_PRECISION == RoutedPrecision::MXFP8;
static_assert(ACCUMULATE_WGRAD || !BF16_MAIN_GRAD,
              "BF16_MAIN_GRAD is only meaningful for fused wgrad accumulation");

#include "types.cuh"

#include "utils.cuh"
#include "dispatch_combine.cuh"
#include "swiglu.cuh"
#include "grouped_gemm.cuh"

#include "forward.cuh"
#include "backward.cuh"
#include "recompute_forward_context.cuh"

}; // struct dispatch_mlp_swiglu_combiner
