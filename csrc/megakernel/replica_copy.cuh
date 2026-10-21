// Forward-only, fixed-plan transport with up to eight replicas. Reuses this communication CTA's
// dispatch/combine staging memory after the preceding operation is complete.
// All CTAs retain their original communication/compute roles and CLC protocol.
static constexpr int REPLICA_TILE_BYTES = 32768;
static constexpr int REPLICA_STAGES = 4;
static_assert(REPLICA_TILE_BYTES * REPLICA_STAGES <= config::DYNAMIC_SHARED_MEMORY - 1024);

static __device__ __forceinline__ void replica_copy_kernel(
    const globals_fwd &g, int comm_cta_idx, uint64_t smem_base_addr,
    semaphore (&arrived)[REPLICA_STAGES]) {
    if (threadIdx.x == 0) {
        auto *data = reinterpret_cast<unsigned char *>(smem_base_addr);
        const auto &p = g.replica;
        const int64_t chunks = (p.copy_bytes + REPLICA_TILE_BYTES - 1) / REPLICA_TILE_BYTES;
        const int64_t tasks = chunks * p.num_experts;
        int phases[REPLICA_STAGES] = {0, 0, 0, 0};
        auto issue = [&](int64_t task, int stage) {
            int e = task % p.num_experts;
            int64_t offset = (task / p.num_experts) * REPLICA_TILE_BYTES;
            uint32_t bytes = static_cast<uint32_t>(min(int64_t(REPLICA_TILE_BYTES), p.copy_bytes-offset));
            tma::expect_bytes(arrived[stage], bytes);
            tma::load_async(static_cast<void *>(data + stage * REPLICA_TILE_BYTES),
                            static_cast<void *>(p.sources[e] + offset), bytes, arrived[stage]);
        };
        #pragma unroll
        for (int stage=0; stage<REPLICA_STAGES; ++stage) {
            int64_t task = comm_cta_idx + int64_t(stage) * g.num_comm_sms;
            if (task < tasks) issue(task, stage);
        }
        int sequence = 0;
        for (int64_t task=comm_cta_idx; task<tasks; task+=g.num_comm_sms, ++sequence) {
            int stage = sequence % REPLICA_STAGES;
            wait(arrived[stage], phases[stage]); phases[stage] ^= 1;
            int e = task % p.num_experts;
            int64_t offset = (task / p.num_experts) * REPLICA_TILE_BYTES;
            uint32_t bytes = static_cast<uint32_t>(min(int64_t(REPLICA_TILE_BYTES), p.copy_bytes-offset));
            tma::store_async(static_cast<void *>(p.destination + e*p.stride_bytes + offset),
                             static_cast<void *>(data + stage*REPLICA_TILE_BYTES), bytes);
            tma::store_async_read_wait();
            int64_t next = task + int64_t(REPLICA_STAGES) * g.num_comm_sms;
            if (next < tasks) issue(next, stage);
        }
        // Full completion (not just read-wait) before release publication.
        tma::store_async_wait();
        asm volatile("red.release.gpu.global.add.s32 [%0], 1;" :: "l"(p.ready) : "memory");
    }
    __syncthreads(); // all threads must finish before dispatch reuses staging
}
static __device__ __forceinline__ void replica_wait(const globals_fwd &g) {
    if (threadIdx.x == 0) {
        int value;
        do {
            asm volatile("ld.relaxed.gpu.global.s32 %0, [%1];" : "=r"(value) : "l"(g.replica.ready) : "memory");
            if (value < g.num_comm_sms) __nanosleep(16);
        } while (value < g.num_comm_sms);
        asm volatile("fence.acquire.gpu;" ::: "memory");
    }
    __syncthreads();
}
