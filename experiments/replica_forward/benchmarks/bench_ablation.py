#!/usr/bin/env python3
"""MOK core-kernel ablations. No in-kernel timestamps or profiler instrumentation."""
import argparse
import datetime
import gc
import hashlib
import json
import os
import socket
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist
from mok import _C, functional, ops
from tests.utils import generate_inputs, run_reference_bf16, check_correctness, MXFP8_TOLERANCE
from random_exact_balanced_route import make_random_exact_balanced_topk

MODES = {'Full': 0, 'Comm-only': 1, 'Compute-only': 2}


def swizzle(scales, rows, cols):
    e = scales.shape[0]
    return (scales.reshape(e, rows//128, 128, cols//128, 4)
            .permute(0, 1, 3, 2, 4).reshape(e, rows//128, cols//128, 4, 32, 4)
            .transpose(-3, -2).reshape(e*rows//128, cols//128, 32, 16).contiguous())


def native_weight_view(weights):
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
    import transformer_engine_torch as tex
    q = MXFP8Quantizer(tex.DType.kFloat8E4M3, rowwise=True, columnwise=True)
    q.optimize_for_gemm = False
    owners = [q.quantize(w.contiguous()) for w in weights]
    row = [w._rowwise_data.view(torch.float8_e4m3fn) for w in owners]
    col = [w._columnwise_data.view(torch.float8_e4m3fn) for w in owners]
    m, k = weights[0].shape
    row_sc = [swizzle(w._rowwise_scale_inv.unsqueeze(0), m, k) for w in owners]
    col_sc = [swizzle(w._columnwise_scale_inv.transpose(-2,-1).unsqueeze(0), k, m) for w in owners]
    view = functional.SplitRoutedWeight(
        data=row[0], storage_table=ops.make_routed_weight_storage_table_mxfp8(row),
        scale=row_sc[0], scale_storage_table=ops.make_routed_scale_storage_table(row_sc),
        scale_tensors=tuple(row_sc), transposed_data=col[0], transposed_scale=col_sc[0],
        transposed_storage_table=ops.make_routed_weight_storage_table_mxfp8(col),
        transposed_scale_storage_table=ops.make_routed_scale_storage_table(col_sc),
        transposed_scale_tensors=tuple(col_sc), native_columnwise=True,
    )
    return view, owners


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--validate-only', action='store_true')
    p.add_argument('--ep', type=int, default=64)
    p.add_argument('--experts', type=int, default=384)
    p.add_argument('--tokens', default='4096,8192,16384')
    p.add_argument('--sms', default='4,8,12,16,24,32,40,48,56,64,80,96,112,128')
    p.add_argument('--minibatch', type=int, default=8192)
    p.add_argument('--macrobatch', type=int, default=131072)
    p.add_argument('--warmup', type=int, default=50)
    p.add_argument('--iterations', type=int, default=100)
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--output', required=True)
    p.add_argument('--grad-dtype', choices=['bf16','fp32'], default='bf16')
    a = p.parse_args()
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    dist.init_process_group('nccl', device_id=device, timeout=datetime.timedelta(minutes=5))
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == (4 if a.smoke else a.ep), world
    cpus = sorted(os.sched_getaffinity(0))
    local_world = int(os.environ.get('LOCAL_WORLD_SIZE','4'))
    if len(cpus) >= local_world:
        os.sched_setaffinity(0, cpus[len(cpus)*local_rank//local_world:len(cpus)*(local_rank+1)//local_world])
    h,i,e,k = (512,256,16,2) if a.smoke else (7168,3072,a.experts,6)
    token_cases = [512] if a.smoke else [int(x) for x in a.tokens.split(',')]
    if a.smoke:
        a.minibatch, a.macrobatch = 512, 8192
        a.warmup, a.iterations, a.rounds = 3, 5, 1
        a.sms = '4,16,32'
    sms_total = torch.cuda.get_device_properties(device).multi_processor_count
    sm_cases = [int(x) for x in a.sms.split(',') if int(x) < sms_total]
    assert all(x > 0 and x % 2 == 0 for x in sm_cases)
    output = Path(a.output)
    if rank == 0:
        output.parent.mkdir(parents=True, exist_ok=True)
    def emit(record):
        if rank == 0:
            record['timestamp'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            with output.open('a') as f: f.write(json.dumps(record, sort_keys=True)+'\n')
            print(json.dumps(record, sort_keys=True), flush=True)
    emit({'kind':'environment','world_size':world,'gpu':torch.cuda.get_device_name(device),
          'sm_count':sms_total,'torch':torch.__version__,'cuda':torch.version.cuda,
          'hostname':socket.gethostname(),'job_id':os.environ.get('SLURM_JOB_ID'),
          'arguments':vars(a),'scope':'megakernel only; setup/reset/EP completion barriers excluded',
          'source_base':'d85f7ce4ec1254d4473a595962e9e31467c17db2'})
    trial = 0
    for t in token_cases:
        if rank == 0: print('PREPARE tokens={}'.format(t), flush=True)
        inp = list(generate_inputs(rank,device,e,e//world,k,t,h,i))
        inp[1] = make_random_exact_balanced_topk(rank,world,device,num_local_tokens=t,num_experts=e,topk=k,seed=20260905)
        x,topk,probs,sg,su,sd,rg,ru,rd,dy = inp
        counts = torch.bincount(topk.flatten(),minlength=e)
        dist.all_reduce(counts)
        assert torch.equal(counts,torch.full_like(counts,world*t*k//e))
        assert bool((topk.sort(dim=1).values[:,1:] != topk.sort(dim=1).values[:,:-1]).all())
        config = functional.MoKConfig(fwd_num_comm_sms=32,bwd_num_comm_sms=32,
            minibatch_size=a.minibatch,macrobatch_size=a.macrobatch,schedule_capacity_multiplier=2.0/world)
        ws = functional.get_workspace(config,dist.group.WORLD,device=device,num_local_tokens=t,hidden_size=h,topk=k)
        schedule = functional.build_schedule(ws,config,topk,num_local_experts=e//world)
        routed_rows = int(schedule.num_tokens.item())
        assert routed_rows == t*k and routed_rows <= a.macrobatch
        fc1 = [torch.cat((rg[j],ru[j]),dim=0) for j in range(e//world)]
        fc2 = list(rd.unbind(0))
        w1, owner1 = native_weight_view(fc1)
        w2, owner2 = native_weight_view(fc2)
        grad_dtype = torch.bfloat16 if a.grad_dtype=='bf16' else torch.float32
        gs,gu,gd = [torch.zeros_like(w,dtype=grad_dtype) for w in (sg,su,sd)]
        g1 = [torch.zeros_like(w,dtype=grad_dtype) for w in fc1]
        g2 = [torch.zeros_like(w,dtype=grad_dtype) for w in fc2]
        all_grads = [gs,gu,gd]+g1+g2
        table1 = ops.make_routed_d_weight_storage_table(g1)
        table2 = ops.make_routed_d_weight_storage_table(g2)
        grad_args = (gs,g1[0],gu,g1[0],gd,g2[0])
        def clear_grads():
            for g in all_grads: g.zero_()
        def gpu_barrier():
            ops.barrier_all(ws.barrier_buffer,ws.barrier_buffer_ptrs,ws.barrier_buffer_multicast_ptr,ws.barrier_target)
        _C.ablation_capture(True)
        y,ctx = functional.forward(config,ws,schedule,x,probs,sg,su,sd,w1,w1,w2,swiglu_limit=10.0)
        grads = functional.backward(config,ws,schedule,ctx,dy,x,probs,sg,su,sd,w1,w1,w2,
            swiglu_limit=10.0,main_grads=grad_args,main_grad_storage_tables=(table1,table1,table2))
        _C.ablation_capture(False)
        torch.cuda.synchronize()
        if a.smoke:
            with torch.enable_grad():
                ref = run_reference_bf16(*inp,swiglu_limit=10.0)
            actual = (y,grads[0],grads[1],torch.stack([g[:i] for g in g1]),
                      torch.stack([g[i:] for g in g1]),torch.stack(g2),gs,gu,gd)
            for n,r,v in zip(('y','dx','dp','wg','wu','wd','wsg','wsu','wsd'),ref,actual):
                check_correctness(n,r,v,MXFP8_TOLERANCE,print_stats=rank==0)
            del ref,actual
        # Local compute outputs are independent from peer-facing communication destinations.
        obs_f = _C.ablation_observables(False)
        obs_b = _C.ablation_observables(True)
        refs_f = [obs_f[0].clone(),obs_f[1][:routed_rows].clone()]
        refs_b = [obs_b[0].clone(),obs_b[1][:routed_rows].clone()]
        refs_g = [g.clone() for g in all_grads]
        route_sha = hashlib.sha256(topk.cpu().numpy().tobytes()).hexdigest()
        shas = [None]*world
        dist.all_gather_object(shas,route_sha)
        emit({'kind':'input','tokens':t,'hidden':h,'intermediate':i,'experts':e,'topk':k,
              'routes_per_expert':world*t*k//e,'routed_rows_per_rank':routed_rows,
              'route_fingerprint':hashlib.sha256(''.join(shas).encode()).hexdigest(),
              'minibatch':a.minibatch,'macrobatch':a.macrobatch,'main_grad_dtype':a.grad_dtype})
        def step(backward,mode,sm):
            _C.ablation_reset(backward)
            if backward: clear_grads()
            _C.ablation_launch(backward,mode,sm)
            gpu_barrier()
        # Verify both real-compute modes against the prepared FULL reference;
        # comm-only must leave all gradients zero after the explicit reset.
        for backward in (False,True):
            for mode_name,mode in MODES.items():
                if rank==0: print('VALIDATE tokens={} backward={} mode={}'.format(t,backward,mode_name),flush=True)
                step(backward,mode,32)
                torch.cuda.synchronize()
                if mode != 1:
                    observations = obs_b if backward else obs_f
                    references = refs_b if backward else refs_f
                    for n,(got,expected) in enumerate(zip(observations[:2],references)):
                        if n==1: got=got[:routed_rows]
                        torch.testing.assert_close(got,expected,rtol=0,atol=0)
                    if backward:
                        for got,expected in zip(all_grads,refs_g):
                            torch.testing.assert_close(got,expected,rtol=0,atol=0)
                elif backward:
                    assert all(int(torch.count_nonzero(g).item())==0 for g in all_grads)
                emit({'kind':'validation','tokens':t,'backward':backward,'mode':mode_name,'status':'PASS'})
        # Poison peer-facing destinations: Comm-only must refill them, whereas
        # Compute-only must leave them untouched while still producing local math.
        for backward in (False,True):
            dest = [ws.d_x_routed_buffer,ws.d_router_weight_buffer] if backward else [ws.combine_buffer]
            step(backward,0,32)
            torch.cuda.synchronize()
            expected_dest = [buf.clone() for buf in dest]
            if a.validate_only and rank == world-1:
                time.sleep(0.05)  # Stress delayed initialization; outside all timing.
            for buf in dest: buf.fill_(float('nan'))
            gpu_barrier()  # All destinations initialized before any peer writes.
            step(backward,1,32)
            torch.cuda.synchronize()
            for got,expected in zip(dest,expected_dest):
                torch.testing.assert_close(got,expected,rtol=0,atol=0)
            if a.validate_only and rank == world-1:
                time.sleep(0.05)  # Stress delayed initialization; outside all timing.
            for buf in dest: buf.fill_(float('nan'))
            gpu_barrier()  # All destinations initialized before any peer writes.
            step(backward,2,32)
            torch.cuda.synchronize()
            assert all(bool(torch.isnan(buf).all().item()) for buf in dest)
            gpu_barrier()  # Do not restore peer outputs before all ranks finish inspecting.
            step(backward,0,32)
            torch.cuda.synchronize()
            del expected_dest,dest
            emit({'kind':'validation','tokens':t,'backward':backward,
                  'check':'peer-write poison test','status':'PASS'})
        del refs_f,refs_b,refs_g
        torch.cuda.synchronize()
        # Pair the order across rounds; no profiler or in-kernel instrumentation.
        for repeat in range(0 if a.validate_only else a.rounds):
            sms_order = sm_cases if repeat % 2 == 0 else list(reversed(sm_cases))
            mode_order = list(MODES.items())
            if repeat % 2: mode_order.reverse()
            for sm in sms_order:
                for backward in (False,True):
                    for mode_name,mode in mode_order:
                        if rank == 0:
                            print('MEASURE t={} sm={} {} {} round={}'.format(t,sm,'bwd' if backward else 'fwd',mode_name,repeat),flush=True)
                        for _ in range(a.warmup): step(backward,mode,sm)
                        events = [(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)) for _ in range(a.iterations)]
                        gpu_barrier()
                        for begin,end in events:
                            _C.ablation_reset(backward)
                            if backward: clear_grads()
                            begin.record()
                            _C.ablation_launch(backward,mode,sm)
                            end.record()
                            # Complete all peer writes before any rank resets/reuses state.
                            gpu_barrier()
                        torch.cuda.synchronize()
                        local_ms = torch.tensor([a.elapsed_time(b) for a,b in events],device=device,dtype=torch.float64)
                        per_rank = [torch.empty_like(local_ms) for _ in range(world)]
                        dist.all_gather(per_rank,local_ms)
                        matrix = torch.stack(per_rank)
                        worst = matrix.max(dim=0).values
                        rec={'kind':'timing','tokens':t,'direction':'bwd' if backward else 'fwd',
                             'mode':mode_name,'comm_sms':sm,'nominal_compute_sms':sms_total-sm,
                             'round':repeat,'iterations':a.iterations,'warmup':a.warmup,
                             'median_rank_max_ms':float(torch.quantile(worst,0.5).item()),
                             'p95_rank_max_ms':float(torch.quantile(worst,0.95).item()),
                             'min_rank_max_ms':float(worst.min().item()),
                             'per_rank_median_ms':torch.quantile(matrix,0.5,dim=1).cpu().tolist(),
                             'rank_max_samples_ms':worst.cpu().tolist()}
                        emit(rec)
                        trial += 1
                        if rank==0:
                            log={'ts':datetime.datetime.now().isoformat(timespec='seconds'),
                                 'kernel':'mok_'+rec['direction']+'_'+mode_name,'trial':trial,
                                 'jit_s':float(os.environ.get('BUILD_WALL_SECONDS','0')) if trial==1 else 0.0,
                                 'gpu_us':1000*rec['median_rank_max_ms'],'verify':'PASS',
                                 'note':'runtime ablation; T={} EP={} comm_sms={} round={}; native MXFP8/BF16 gradients'.format(t,world,sm,repeat)}
                            with (output.parent/'trial_log.jsonl').open('a') as f: f.write(json.dumps(log)+'\n')
        torch.cuda.synchronize()
        _C.ablation_clear()
        del obs_f,obs_b,grads,y,ctx,w1,w2,owner1,owner2,all_grads,grad_args,table1,table2,g1,g2,fc1,fc2
        del inp,x,topk,probs,sg,su,sd,rg,ru,rd,dy,gs,gu,gd,schedule,ws
        functional.clear_workspace_cache()
        gc.collect()
        torch.cuda.empty_cache()
    emit({'kind':'complete','status':'PASS'})
    dist.destroy_process_group()

if __name__=='__main__':
    with torch.no_grad(): main()
