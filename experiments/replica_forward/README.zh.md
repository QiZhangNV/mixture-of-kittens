# MOK Forward replica weight dispatch

本分支分享已经完成 EP64 验证的 Forward 融合原型、独立 benchmark 和报告。默认采用粗粒度 TMA pull：先 dispatch 一批 home token，再拉取 replica 权重，随后继续 dispatch token；shared/home expert 计算与复制并行。不启用后续没有稳定收益的 W1 early/fine-priority 实验。

[性能报告](report/BENCHMARK.zh.md) · [指标 CSV](report/metrics.csv) · [源码与接口说明](CODEMAP.zh.md) · [验证范围](report/SHARE_VALIDATION.zh.md)

## 获取和构建

```bash
git clone --branch qizhang/replica-forward-share-20261020 --recurse-submodules \
  git@github.com:QiZhangNV/mixture-of-kittens.git
cd mixture-of-kittens/experiments/replica_forward

# CPU 检查；不会初始化 CUDA。
python3 scripts/verify.py

# 在 GPU 计算节点上构建。源码不含预编译 .so。
ARCH=SM103 PYTHON=python3 bash scripts/build.sh all
```

环境：Linux、git、make、Python>=3.12、CUDA 13.x NVCC、CUDA 版 PyTorch>=2.10、NumPy。MXFP8 另需与 PyTorch/CUDA 匹配的 Transformer Engine；历史环境版本见原始 JSONL。GB300 使用 `ARCH=SM103`；SM100 是构建选项，本报告没有在 SM100 上复测。

本分支基于此前测试的 integration commit `d85f7ce4ec1254d4473a595962e9e31467c17db2`，未混入之后其他功能分支的修改。请保留 Git 历史，不使用 depth=1 的浅克隆，原版对照需要该 commit。

`build.sh` 构建根目录融合扩展、独立 transport，以及原版对照扩展。原版从固定 Git commit 加 `patches/baseline_*.patch` 生成到 `.build/`，不提交重复仓库。MXFP8/BF16 对照都保留 capture/ablation hooks，Full 模式运行真实计算。BF16 相对 MXFP8 对照只多了 BF16 host capture，未修改 BF16 device 计算。

ThunderKittens 固定在 `1c3920d993404dd49a6d4c7267ea11d583bd5c68`，构建脚本在 `.build/ThunderKittens` 生成带已有 MXFP8 MN-major 兼容补丁的 include 副本；不修改 checkout 内的 submodule。115 个 include 文件逐文件匹配历史验证版本。原版对照、融合版、transport 共用这份 headers。

`build.sh bf16`/`build.sh mxfp8` 可分别构建需要的对照；`all` 构建全部。构建日志和 trial 记录写入 `build_logs/`。源码和数据哈希在 `source_provenance.json`/`MANIFEST.sha256.json`；修改源码后需要重新记录验证。

## 先做 4 GPU 正确性检查

```bash
bash scripts/run.sh --model dsv4_pro --precision bf16 --profile ECHO \
  --smoke --output results/smoke_bf16/timings.jsonl
bash scripts/run.sh --model dsv4_pro --precision mxfp8 --profile ECHO \
  --smoke --output results/smoke_mxfp8/timings.jsonl
python3 scripts/report.py results/smoke_bf16/timings.jsonl \
  results/smoke_mxfp8/timings.jsonl --output results/smoke_report
```

`--smoke` 使用 H512/I256/T2048，只验证接口和协议，不能用于完整模型性能比较。检查包含 poisoned replica weights、独立复制字节与 guard、相同 logical/physical routing 的输出一致性、所有 SM/prefix 组合、weight epoch 更新和 CUDA Graph 重放。

当前分享源码为了复现既有编译配置，仅实例化 EP4/EP64 的 entrypoint；其他 EP 未启用。它是单 NVLink domain 的实验分支。

## EP64 正式测试

64 GPU 必须位于同一个 NVLink domain。假设16节点、每节点4 GPU，在每个节点各启动一次，设置相同的 MASTER_ADDR/PORT 和对应的 NODE_RANK：

```bash
export NNODES=16 NPROC_PER_NODE=4
export NODE_RANK=0                 # 其他节点为1..15；默认也可读取SLURM_NODEID
export MASTER_ADDR=<node0-address> MASTER_PORT=29500
bash scripts/run.sh --model dsv4_pro --precision mxfp8 --profile ECHO \
  --tokens 4096 --minibatches 4096 --mok-sms 24,32,40,48,56,64 \
  --warmup 50 --iterations 100 --rounds 3 \
  --output results/dsv4_mxfp8_echo/timings.jsonl

# 测试完成后仅在一个节点生成报告。
python3 scripts/report.py results/dsv4_mxfp8_echo/timings.jsonl \
  --output results/dsv4_mxfp8_echo/report
```

每次运行使用新的 output 文件，不能向同一 JSONL 追加第二组实验。`--profile` 可选 ECHO/UltraEP/MoonEP，表示截图边缘分布的合成近似，不执行这些 planner，也不能据此给 planner 排名。

| model 参数 | H | I | EP64逻辑experts | top-k | 报告精度 |
|---|---:|---:|---:|---:|---|
| dsv4_pro | 7168 | 3072 | 384 | 6 | BF16 / MXFP8 |
| glm52 | 6144 | 2048 | 256 | 8 | BF16 |
| qwen35_397b | 4096 | 1024 | 512 | 10 | BF16 |

Shared expert 的 intermediate 同为 I，计算精度为 BF16；DSv4 启用 clamp10，另外两个 shape 不带 clamp。非 EP64 运行保持原每 rank home 数，因此全局 expert 数随 world 缩放。

## 报告复算和画图

CPU 即可从附带的历史原始数据生成报告；无需 torch/CUDA：

```bash
python3 scripts/report.py --output results/recomputed_history
# 可选，安装 matplotlib 后生成 PNG/SVG。
python3 scripts/plot_report.py --metrics results/recomputed_history/metrics.csv \
  --output results/recomputed_history/forward_summary.png
```

报告区分 OriginalReady（原版 MOK，副本权重预先就绪）、Serial（独立 copy 后运行原版 MOK）和 Fused（通信 CTA 内 copy），三者使用相同 physical token plan。OriginalReady 不是关闭负载均衡的结果。

计时保留 token dispatch/combine、shared expert、激活/量化处理和训练 context 保存；排除 router/planner/schedule、权重量化/swizzle/源权重发布、reset、最终外部 epilogue。主表为 routed FLOPs / 整个 MOK kernel 时间，CSV 另列含 shared 的总 MLP TFLOPS。

```text
有效隐藏比例 = (最佳 Serial - 最佳 Fused) / (最佳 Serial - 最佳 OriginalReady)
暴露比例 = (最佳 Fused - 最佳 OriginalReady) / (最佳 Serial - 最佳 OriginalReady)
```

这表示回收的串行增量，包含重新调节 SM 的效果，不等于时间线物理重叠比例。CSV 还给出固定在最佳 Fused SM 数下的隐藏比例。

本轮未修改 MCore，不需要 MCore patch。融合接口目前是 benchmark capture/configure/launch 原型，尚不是 MCore layer/autograd 接口；Backward weight dispatch、replica gradient reduce、wgrad accum 生命周期不在本分支实现范围。详细限制见源码说明和报告。
