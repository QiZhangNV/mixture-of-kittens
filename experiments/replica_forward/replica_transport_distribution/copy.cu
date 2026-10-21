#include <torch/extension.h>
#include "kittens.cuh"
#include "pyutils/torchutils.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <array>
#include <vector>

namespace replica_distribution_copy {
constexpr int kMaxExperts = 8;
constexpr int kTileBytes = 32768;
constexpr int kStages = 4;
struct Config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_THREADS = 256;
    static constexpr int DYNAMIC_SHARED_MEMORY = kTileBytes * kStages + 1024;
};
struct Globals {
    std::array<unsigned char *, kMaxExperts> sources;
    unsigned char *destination;
    int64_t copy_bytes;
    int64_t stride_bytes;
    int num_sms;
    int num_experts;
    __host__ dim3 grid() const { return dim3(num_sms); }
};
__device__ void kernel(const Globals &g) {
    extern __shared__ unsigned char raw[];
    auto *stage_data = reinterpret_cast<unsigned char *>((reinterpret_cast<uint64_t>(raw)+127)&~uint64_t(127));
    __shared__ kittens::semaphore arrived[kStages];
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int s=0;s<kStages;++s) kittens::init_semaphore(arrived[s],0,1);
    }
    __syncthreads();
    if (threadIdx.x != 0) return;
    const int64_t chunks = (g.copy_bytes+kTileBytes-1)/kTileBytes;
    const int64_t tasks = chunks*g.num_experts;
    int phases[kStages] = {0,0,0,0};
    auto issue = [&](int64_t task,int stage) {
        const int expert = task%g.num_experts;
        const int64_t offset = (task/g.num_experts)*kTileBytes;
        const uint32_t bytes = static_cast<uint32_t>(((g.copy_bytes-offset)<kTileBytes ? (g.copy_bytes-offset) : kTileBytes));
        kittens::tma::expect_bytes(arrived[stage],bytes);
        kittens::tma::load_async(static_cast<void *>(stage_data+stage*kTileBytes),
            static_cast<void *>(g.sources[expert]+offset),bytes,arrived[stage]);
    };
    #pragma unroll
    for (int stage=0;stage<kStages;++stage) {
        int64_t task=blockIdx.x+int64_t(stage)*gridDim.x;
        if (task<tasks) issue(task,stage);
    }
    int64_t sequence=0;
    for (int64_t task=blockIdx.x;task<tasks;task+=gridDim.x,++sequence) {
        const int stage=sequence%kStages;
        kittens::wait(arrived[stage],phases[stage]);
        phases[stage]^=1;
        const int expert=task%g.num_experts;
        const int64_t offset=(task/g.num_experts)*kTileBytes;
        const uint32_t bytes=static_cast<uint32_t>(((g.copy_bytes-offset)<kTileBytes ? (g.copy_bytes-offset) : kTileBytes));
        kittens::tma::store_async(static_cast<void *>(g.destination+expert*g.stride_bytes+offset),
            static_cast<void *>(stage_data+stage*kTileBytes),bytes);
        // The stage may be overwritten only after all stores stop reading it.
        kittens::tma::store_async_read_wait();
        const int64_t next=task+int64_t(kStages)*gridDim.x;
        if (next<tasks) issue(next,stage);
    }
    // Serial's next kernel must observe every byte, not merely reusable SMEM.
    kittens::tma::store_async_wait();
}
class Context {
    at::Tensor destination_;
    Globals globals_{};
    int sm_count_;
public:
    Context(const std::vector<int64_t> &sources,at::Tensor destination,int64_t copy_bytes)
        :destination_(destination) {
        TORCH_CHECK(destination.is_cuda() && destination.scalar_type()==at::kByte && destination.is_contiguous(),"destination must be contiguous CUDA uint8");
        TORCH_CHECK(destination.dim()==2 && sources.size()<=kMaxExperts && destination.size(0)>=static_cast<int64_t>(sources.size()) && destination.size(0)>=1 && destination.size(0)<=kMaxExperts,"expected 0 to 8 active sources fitting destination capacity");
        TORCH_CHECK(copy_bytes>0 && copy_bytes<=destination.size(1) && copy_bytes%16==0 && destination.size(1)%16==0,"copy size/stride must be positive aligned byte counts");
        c10::cuda::CUDAGuard guard(destination.device());
        for (int j=0;j<static_cast<int>(sources.size());++j) {
            TORCH_CHECK(sources[j]!=0 && sources[j]%16==0,"unaligned source pointer");
            globals_.sources[j]=reinterpret_cast<unsigned char *>(sources[j]);
        }
        globals_.destination=destination.data_ptr<unsigned char>();
        globals_.copy_bytes=copy_bytes;globals_.stride_bytes=destination.size(1);globals_.num_sms=1;globals_.num_experts=static_cast<int>(sources.size());
        sm_count_=at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
        CUDACHECK(cudaFuncSetAttribute(kittens::py::global_kernel<Config,Globals,kernel>,cudaFuncAttributeMaxDynamicSharedMemorySize,Config::DYNAMIC_SHARED_MEMORY));
    }
    void run(int sms) {
        c10::cuda::CUDAGuard guard(destination_.device());
        TORCH_CHECK(sms>0 && sms<=sm_count_,"invalid copy SM budget");
        globals_.num_sms=sms;
        kittens::py::launch_kernel<Config,Globals,kernel>(globals_);
    }
    pybind11::dict resources() {
        c10::cuda::CUDAGuard guard(destination_.device());
        int active=0;cudaFuncAttributes attr{};
        CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,kittens::py::global_kernel<Config,Globals,kernel>,Config::NUM_THREADS,Config::DYNAMIC_SHARED_MEMORY));
        CUDACHECK(cudaFuncGetAttributes(&attr,kittens::py::global_kernel<Config,Globals,kernel>));
        pybind11::dict d;d["active_blocks_per_sm"]=active;d["registers"]=attr.numRegs;d["dynamic_shared_bytes"]=Config::DYNAMIC_SHARED_MEMORY;d["static_shared_bytes"]=attr.sharedSizeBytes;d["tile_bytes"]=kTileBytes;d["stages"]=kStages;d["sm_count"]=sm_count_;return d;
    }
};
}
PYBIND11_MODULE(_replica_transport_distribution,m) {
    pybind11::class_<replica_distribution_copy::Context>(m,"CopyContext")
      .def(pybind11::init<const std::vector<int64_t>&,at::Tensor,int64_t>())
      .def("run",&replica_distribution_copy::Context::run)
      .def("resources",&replica_distribution_copy::Context::resources);
}
