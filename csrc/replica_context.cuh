#pragma once
#include <ATen/ATen.h>
#include <array>
#include <functional>
#include <vector>

// Benchmark-only owner. Source symmetric allocations stay alive in the harness.
namespace mok_distribution_replica {
struct Parameters {
    std::array<unsigned char *, 8> sources{};
    unsigned char *destination = nullptr;
    int *ready = nullptr;
    int64_t copy_bytes = 0;
    int64_t stride_bytes = 0;
    int num_experts = 0;
    int capacity = 0;
    int home_rows = 0;
    int prefix_rows = 0;
};
inline Parameters parameters;
inline at::Tensor destination_owner, ready_owner;
inline std::function<void(int,int)> launch;
inline std::function<std::vector<int>()> resources;
inline void configure(const std::vector<int64_t> &sources, const at::Tensor &destination,
                      const at::Tensor &ready, int64_t bytes, int home_rows) {
    TORCH_CHECK(sources.size() <= 8 && destination.is_cuda() && destination.is_contiguous() &&
                destination.scalar_type() == at::kByte && destination.dim() == 2 && destination.size(0) >= static_cast<int64_t>(sources.size()) && destination.size(0) >= 1 && destination.size(0) <= 8,
                "Expected 0 to 8 active source pointers fitting a nonempty destination capacity");
    TORCH_CHECK(ready.is_cuda() && ready.device() == destination.device() && ready.is_contiguous() &&
                ready.scalar_type() == at::kInt && ready.numel() == 1, "Expected one CUDA int32 ready counter");
    TORCH_CHECK(bytes > 0 && bytes <= destination.size(1) && bytes % 16 == 0 && destination.size(1) % 16 == 0,
                "Invalid aligned payload/stride bytes");
    TORCH_CHECK(home_rows >= 0 && home_rows % 256 == 0, "Invalid home row count");
    for (int e=0; e<static_cast<int>(sources.size()); ++e) {
        TORCH_CHECK(sources[e] > 0 && sources[e] % 16 == 0, "Invalid source address");
        parameters.sources[e] = reinterpret_cast<unsigned char *>(sources[e]);
    }
    destination_owner = destination; ready_owner = ready;
    parameters.destination = destination.data_ptr<unsigned char>();
    parameters.ready = ready.data_ptr<int>();
    parameters.copy_bytes = bytes; parameters.stride_bytes = destination.size(1);
    parameters.num_experts = static_cast<int>(sources.size());
    parameters.capacity = static_cast<int>(destination.size(0));
    parameters.home_rows = home_rows;
}
inline void reset() { TORCH_CHECK(ready_owner.defined(), "Configure transport first"); ready_owner.zero_(); }
inline void clear() { launch = {}; resources = {}; destination_owner = at::Tensor{}; ready_owner = at::Tensor{}; parameters = {}; }
}
