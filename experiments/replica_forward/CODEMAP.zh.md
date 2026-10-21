# 源码和接口

## 根目录 MOK 改动

- `csrc/replica_context.cuh`：保存 replica 目标 tensor、source pointers、ready counter 和 launch 回调。
- `csrc/megakernel/replica_copy.cuh`：32 KiB tile、4 stage 的 TMA pull，统一轮转任务分配；复制完成后发布 ready。
- `csrc/megakernel/forward.cuh`：在通信 CTA 中交错 home token、replica weight、后续 token；replica 计算等待相应复制完成。保留原 Shared/Home 计算。
- `csrc/megakernel/types.cuh`、`megakernel.cuh`：增加 replica 参数及头文件依赖。
- `csrc/ablation_context.cuh`、`bindings.cu`：捕获已有 functional launch 的参数并持有 tensor，配置复制和重放。
- `dispatch_combine.cuh`、`grouped_gemm.cuh`、`swiglu.cuh`、`backward.cuh`、`utils.cuh`：此前通信/计算消融的模式参数与占位控制逻辑，Full 模式保留真实工作。这些是测量工具，不是新实现的 Backward replica transport。
- `entrypoints.cuh`：只实例化历史验证使用的 EP4/EP64，以控制编译量；并非所有原版 EP 都在本分享版本中启用。

CUDA/Python 计算源文件按历史验证版本原样保留，没有把失败或无稳定收益的调度实验混入。`mok.functional` 原有参数签名未在本轮整理中改变。

## benchmark 和对照

- `benchmarks/bench_distribution_replica.py`：同一 logical/physical plan 下验证 OriginalReady/Serial/Fused，扫描通信 SM、prefix 和 minibatch；CUDA events 在 kernel 外计时，不使用 TKProfiler/IKET。
- `replica_transport_distribution/copy.cu`：独立 copy，供 Serial 和 Copy-only 使用；与 Fused 使用相同源、目标、payload 和 TMA 粒度。
- `benchmarks/distribution_plan.py`：由截图的 ingress/egress 边缘分布生成合成 placement。不是实际 planner 的每 microbatch 路由回放。
- `patches/baseline_*.patch`：以固定 MOK commit 生成原版对照和 capture hooks，`.build/` 为生成物，不纳入 Git。
- `patches/thunderkittens.patch`：此前已有的 MXFP8 MN-major MMA/descriptor 兼容改动；所有待比较扩展使用同一份 headers。
- `source_provenance.json`：实际计算源码、生成对照和115个 dependency headers 的逐文件 SHA256。

## 融合接口

```python
from mok import _C, functional

# 先执行一次真实 forward，分配 outputs/context，并捕获 kernel 参数。
_C.ablation_capture(True)
y, ctx = functional.forward(...)
_C.ablation_capture(False)

# sources: 本GPU可直接访问的peer symmetric home地址列表。
_C.replica_configure(sources, replica_uint8, ready_int32, row_bytes, home_rows)

# 以下 reset 在 benchmark 计时外；每次重放必须重置。
_C.ablation_reset(False)
_C.replica_reset()
_C.replica_launch(comm_sms, first_home_rows)
```

完整使用方式以 benchmark 为准。此入口需要先配置有效的 tensor/pointer 生命周期；capture 是进程级全局状态，本分支只验证串行单 context，未提供多层或多 microbatch 并发的生产管理接口。

捕获 `functional.forward` 得到的 `y` 是完整 layer 输出；单独重放 MegaKernel 只更新 core buffers。需要完整 layer 输出时还要调用原外部 forward epilogue，不能把它遗漏在端到端实现中。为了比较 MegaKernel，本报告统一排除该 epilogue。

## 权重和生命周期边界

源权重预先放在 symmetric home bank；接收端 TMA pull，源端真实供数。普通权重更新后需要更新源 bank，相关发布/量化不在本次计时范围内。源 bank 在所有相关 peer 完成 pull 之前不能覆写。

BF16 搬 FC1/FC2 数据；MXFP8 搬 rowwise FC1/FC2 与预先 swizzle 的 scale。MXFP8 columnwise 空间保留但不传，因此不能把本实验 replica context 直接用于 Backward。布局在 benchmark 的 `payload_layout` 中定义，不是 MCore 强制的布局约定。

固定 home、固定最大 spare 容量、每 rank 最多8个 active replicas，单 macrobatch，单 NVLink domain。每次调用刷新所有 active replicas；inactive slots 不执行有效 replica 计算。跨 domain、生产训练 weight/main_grad 生命周期和新的外部 planner 接入需要单独实现。
