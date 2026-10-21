#pragma once
#include <functional>
#include <vector>
#include <ATen/ATen.h>

// Benchmark-only host owners. No timestamps, counters or profiler code in device kernels.
namespace mok_distribution_ablation {
struct Context {
    std::vector<at::Tensor> keepalive;
    std::vector<at::Tensor> ready;
    std::vector<at::Tensor> observables;
    std::function<void(int,int)> launch;
};
inline bool capture_enabled = false;
inline Context fwd, bwd;
inline Context &get(bool backward) { return backward ? bwd : fwd; }
inline void reset(bool backward) {
    auto &ctx = get(backward);
    TORCH_CHECK(bool(ctx.launch), "Capture a FULL functional launch first");
    for (auto &tensor : ctx.ready) tensor.zero_();
}
inline void launch(bool backward, int mode, int comm_sms) {
    TORCH_CHECK(mode >= 0 && mode <= 2, "Invalid ablation mode");
    TORCH_CHECK(comm_sms > 0 && comm_sms % 2 == 0, "comm_sms must be positive and even");
    get(backward).launch(mode, comm_sms);
}
inline std::vector<at::Tensor> observables(bool backward) { return get(backward).observables; }
inline void clear() { fwd = Context{}; bwd = Context{}; }
}
