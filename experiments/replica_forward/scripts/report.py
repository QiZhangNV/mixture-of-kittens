#!/usr/bin/env python3
"""Recompute MOK Forward throughput and recovered serial overhead from JSONL."""
import argparse
import collections
import csv
import gzip
import json
import statistics
from pathlib import Path


def read_case(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    environments = [r for r in records if r.get("kind") == "environment"]
    assert len(environments) == 1, f"Use a fresh output filename per run: {path}"
    env = environments[0]
    assert records[-1].get("kind") == "complete" and records[-1].get("status") == "PASS", path
    checks = [r for r in records if r.get("kind") == "validation"]
    assert checks and all(r["status"] == "PASS" for r in checks), path
    groups = collections.defaultdict(list)
    for r in records:
        if r.get("kind") != "timing" or r["mode"] not in ("OriginalReady", "Ready", "Serial", "Fused"):
            continue
        groups[(r["minibatch"], r["mode"], r["mok_sms"], r.get("copy_sms"), r.get("prefix_fraction"))].append(r)
    aggregated = []
    for (mb, mode, sms, copy, prefix), rows in groups.items():
        assert len(rows) == env["arguments"]["rounds"], (path, mode, len(rows))
        times = [r["median_rank_max_ms"] for r in rows]
        aggregated.append(dict(minibatch=mb, mode=mode, comm_sms=sms, copy_sms=copy,
                               prefix_fraction=prefix, ms=statistics.median(times),
                               round_min_ms=min(times), round_max_ms=max(times)))
    shape = env["shape"]
    payload = next(r for r in records if r.get("kind") == "input")
    plan = next(r for r in records if r.get("kind") == "plan")
    result = []
    for mb in sorted({r["minibatch"] for r in aggregated}):
        rows = [r for r in aggregated if r["minibatch"] == mb]
        best = {mode: min((r for r in rows if r["mode"] == mode), key=lambda r: r["ms"])
                for mode in ("OriginalReady", "Ready", "Serial", "Fused")}
        ready, serial, fused = (best[x]["ms"] for x in ("OriginalReady", "Serial", "Fused"))
        delta = serial - ready
        fwd_flops = 6 * shape["tokens"] * shape["topk"] * shape["hidden"] * shape["intermediate"]
        all_flops = 6 * shape["tokens"] * (shape["topk"] + 1) * shape["hidden"] * shape["intermediate"]
        same = {mode: min((r for r in rows if r["mode"] == mode and r["comm_sms"] == best["Fused"]["comm_sms"]), key=lambda r: r["ms"])
                for mode in ("OriginalReady", "Serial")}
        matched_delta = same["Serial"]["ms"] - same["OriginalReady"]["ms"]
        result.append(dict(model=env["model"], precision=env["arguments"]["precision"], profile=env["profile"],
            ep=env["world_size"], tokens=shape["tokens"], hidden=shape["hidden"], intermediate=shape["intermediate"],
            topk=shape["topk"], logical_experts=shape["logical_experts"], minibatch=mb,
            original_ms=ready, serial_ms=serial, fused_ms=fused, modified_ready_ms=best["Ready"]["ms"],
            original_routed_tflops=fwd_flops/(ready*1e9), serial_routed_tflops=fwd_flops/(serial*1e9),
            fused_routed_tflops=fwd_flops/(fused*1e9), original_all_mlp_tflops=all_flops/(ready*1e9),
            fused_all_mlp_tflops=all_flops/(fused*1e9), recovered_serial_pct=100*(serial-fused)/delta if delta>0 else None,
            exposed_serial_pct=100*(fused-ready)/delta if delta>0 else None,
            hidden_same_sm_pct=100*(same["Serial"]["ms"]-fused)/matched_delta if matched_delta>0 else None,
            exposed_ms=fused-ready, serial_overhead_ms=delta, net_saving_pct=100*(serial-fused)/serial,
            original_comm_sms=best["OriginalReady"]["comm_sms"], serial_comm_sms=best["Serial"]["comm_sms"],
            fused_comm_sms=best["Fused"]["comm_sms"], prefix_fraction=best["Fused"]["prefix_fraction"],
            validation_records=len(checks), gpu=env["gpu"], torch=env["torch"], cuda=env["cuda"], source=path.as_posix()))
        result[-1].update(total_copies=plan["total_copies"],
                          source_job_id=env.get("job_id"),
                          mean_receive_mib=statistics.mean(payload["row_receive_bytes_by_rank"])/2**20,
                          maximum_receive_mib=max(payload["row_receive_bytes_by_rank"])/2**20,
                          maximum_send_mib=max(payload["row_send_bytes_by_rank"])/2**20)
    return result, checks


def cost_report(paths):
    rows = []
    seen = set()
    for path in paths:
        if path.exists():
            for line in path.read_text().splitlines():
                if line and line not in seen:
                    rows.append(json.loads(line)); seen.add(line)
    lines = ["## Cost Report", "", "| Kernel | Trials | JIT min/med s | GPU med us | Final |", "|---|---:|---:|---:|---|"]
    for kernel in sorted({r["kernel"] for r in rows}):
        group = [r for r in rows if r["kernel"] == kernel]
        jit = [r["jit_s"] for r in group]; gpu = [r["gpu_us"] for r in group if r["gpu_us"]>0]
        lines.append(f"| {kernel} | {len(group)} | {min(jit):.2f} / {statistics.median(jit):.2f} | {statistics.median(gpu) if gpu else 0:.2f} | {group[-1]['verify']} |")
    return lines


def write_report(metrics, output, checks, trial_paths):
    output.mkdir(parents=True, exist_ok=True)
    if metrics:
        with (output/"metrics.csv").open("w") as handle:
            writer=csv.DictWriter(handle,fieldnames=list(metrics[0]),lineterminator="\n");writer.writeheader();writer.writerows(metrics)
    (output/"validation.json").write_text(json.dumps(checks,ensure_ascii=False,indent=2)+"\n")
    lines=["# MOK Forward 原版与权重通信融合版", "",
        "本报告比较相同物理 token plan 下的原版 OriginalReady、Serial 和 Fused。默认分享版本是已验证的粗粒度 TMA pull 实现：先派发一批 home token，再复制 replica 权重，随后派发其余 token；shared/home 计算与复制并行。", "",
        "## 计时和吞吐定义", "",
        "OriginalReady 使用原 MOK 计算内核，replica 权重已就绪。Serial 为本 rank 完成权重复制后执行原 MOK，不额外增加全局权重 barrier。Fused 在 MOK 通信 CTA 中完成复制。OriginalReady 并不表示关闭负载均衡；三者使用同一个 physical plan。", "",
        "计时只含 Forward MegaKernel，以及 Serial 的权重复制：保留内核内 token dispatch/combine、shared expert、激活处理与训练 context 保存；排除 router/schedule、权重量化与 swizzle、源权重发布、workspace reset 和外部 forward epilogue。数值采用每个样本的 rank 最大值，再取样本中位数和各轮中位数。", "",
        "```text", "T: 每 rank 输入 tokens；K: top-k；H: hidden；I: routed intermediate",
        "本 fixture 有一个 intermediate 同为 I 的 BF16 shared expert。",
        "routed FLOPs = 6*T*K*H*I", "all MLP FLOPs = 6*T*(K+1)*H*I",
        "TFLOPS = FLOPs / (time_ms * 1e9)", "```", "",
        "主表用 routed FLOPs / 整个 MOK kernel 时间，便于核对此前 routed 吞吐口径；它没有从 MOK 时间中扣除 shared。CSV 另列包含 shared 的 all_mlp_tflops。MXFP8 情况的 shared 是 BF16，因此总 MLP TFLOPS 是混合精度有效指标。", "",
        "## 有效隐藏比例", "", "```text", "R = 最佳 OriginalReady；S = 最佳 Serial；F = 最佳 Fused",
        "新增串行开销 = S-R", "有效隐藏比例 = (S-F)/(S-R)", "暴露比例 = (F-R)/(S-R)",
        "融合相对 Serial 的净节省 = (S-F)/S", "```", "",
        "该比例表示回收了多少端到端串行增量，包含重新调节通信 SM 的效果，不是时间线上的物理重叠比例。CSV 同时给出固定在最佳 Fused SM 数下的 hidden_same_sm_pct。Copy-only 不能直接加到 OriginalReady：稀疏收发时，零接收 rank 可以计算并服务别人的 pull。比例不做 0/100 截断。", "",
        "## 实测结果", "",
        "| 模型 | 精度 | 合成 plan | 原版 ms | Serial ms | Fused ms | 原版 routed TFLOPS | Fused routed TFLOPS | 隐藏 | 暴露 | 净节省 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in metrics:
        hidden="—" if r["recovered_serial_pct"] is None else f"{r['recovered_serial_pct']:.1f}%"
        exposed="—" if r["exposed_serial_pct"] is None else f"{r['exposed_serial_pct']:.1f}%"
        lines.append(f"| {r['model']} | {r['precision'].upper()} | {r['profile']} | {r['original_ms']:.4f} | {r['serial_ms']:.4f} | {r['fused_ms']:.4f} | {r['original_routed_tflops']:.0f} | {r['fused_routed_tflops']:.0f} | {hidden} | {exposed} | {r['net_saving_pct']:.2f}% |")
    if not metrics:lines.append("\n此输入只有正确性检查，没有性能测量。")
    if (output/"forward_summary.png").exists():
        lines += ["", "![Forward throughput and recovered serial overhead](forward_summary.png)", ""]
    conditions = "历史报告来自 OCI AGA GB300、EP64/4K 的既有正式运行（811385）；每 rank 的 post-plan routed token 工作均衡，minibatch4096、单 macrobatch，50 warmup、100 samples、3 rounds。源码包重建不会自动产生新的 EP64 性能结论。" if metrics and all(str(r.get("source_job_id"))=="811385" for r in metrics) else "此表由指定输入重新计算。实际 GPU、EP、shape、minibatch、软件版本和运行参数以 CSV 与输入 JSONL 为准；小 shape smoke 仅检查可用性，不代表完整模型性能。"
    lines += ["", "## 范围和复现条件", "", conditions, "",
        "ECHO/UltraEP/MoonEP 标签表示按同事 DSv3 MLPerf EP32 截图边缘分布构造的 EP64 合成近似，不是真实 planner replay，不能用于给三个 planner 排名。所有 active replicas 每次调用都重传，收发边、fan-out 和 token 分配均见所附 plan 与假设。", "",
        "源权重已预先发布到对称内存；普通训练权重到对称 bank 的拷贝/量化不包含在数据内。Forward 权重使用 BF16 或 MXFP8 rowwise data+swizzled scales，源端真实供数包含在复制中。MXFP8 columnwise 空间保留但 Forward 不传，不应直接复用该 replica context 做 Backward。", "",
        "当前是 Forward benchmark 原型，采用 capture/configure/launch 入口，尚不是直接替换 MCore autograd 的生产接口。不支持跨 NVLink domain；固定 home、固定最大 spare 容量、每 rank 最多8个 active replicas；默认只验证单 macrobatch、串行调用的生命周期。", "",
        "[全部指标](metrics.csv) · [正确性检查](validation.json) · [成本记录](summary.md)", ""]
    (output/"BENCHMARK.zh.md").write_text("\n".join(lines).rstrip()+"\n")
    (output/"summary.md").write_text("# MOK Forward benchmark costs\n\n"+"\n".join(cost_report(trial_paths))+"\n")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs",type=Path,nargs="*")
    p.add_argument("--output",type=Path,default=Path("results/report"))
    args=p.parse_args()
    if not args.inputs:
        root=Path(__file__).resolve().parents[1]
        args.inputs=sorted((root/"data/history").rglob("timings.jsonl.gz"))
    if not args.inputs:raise SystemExit("No timing inputs found")
    metrics=[];checks={}
    for path in args.inputs:
        rows,validation=read_case(path);metrics.extend(rows);checks[path.as_posix()]=validation
    metrics.sort(key=lambda r:(r["model"],r["precision"],r["profile"],r["minibatch"]))
    write_report(metrics,args.output,checks,[path.parent/"trial_log.jsonl" for path in args.inputs])
    print(json.dumps({"cases":len(args.inputs),"rows":len(metrics),"checks":sum(map(len,checks.values())),"output":str(args.output)},ensure_ascii=False))


if __name__=="__main__":main()
