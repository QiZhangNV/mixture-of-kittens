#!/usr/bin/env python3
"""Screenshot-distribution synthetic proxy: original/modified Ready, Serial, Fused and ablations."""
import argparse,collections,datetime,gc,hashlib,json,os,socket,statistics
from pathlib import Path
import torch
import numpy as np
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from mok import _C,functional,ops
from tests.utils import generate_inputs,check_correctness,BF16_TOLERANCE,MXFP8_TOLERANCE
from random_exact_balanced_route import make_random_exact_balanced_topk
import _replica_transport_distribution as transport
from distribution_plan import construct
import importlib.util
MODELS = {
    'dsv4_pro': dict(display='DSv4-Pro',h=7168,i=3072,experts=384,k=6,macro=131072),
    'glm52': dict(display='GLM-5.2',h=6144,i=2048,experts=256,k=8,macro=131072),
    'qwen35_397b': dict(display='Qwen3.5-397B-A17B',h=4096,i=1024,experts=512,k=10,macro=262144),
}


def payload_layout(h,i,precision):
    if precision=='bf16':
        specs=[('r1',(2*i,h),'data'),('r2',(h,i),'data')]
    else:
        specs=[('r1',(2*i,h),'data'),('r2',(h,i),'data'),
               ('rs1',(2*i//128,h//128,32,16),'scale'),('rs2',(h//128,i//128,32,16),'scale'),
               ('c1',(2*i,h),'data'),('c2',(h,i),'data'),
               ('cs1',(h//128,2*i//128,32,16),'scale'),('cs2',(i//128,h//128,32,16),'scale')]
    fields={};offset=0
    for name,shape,kind in specs:
        size=2 if precision=='bf16' else 1
        for n in shape:size*=n
        fields[name]={'offset':offset,'bytes':size,'shape':shape,'kind':kind,'dtype':precision if kind=='data' else 'uint8'};offset+=size
    row_bytes=offset if precision=='bf16' else fields['c1']['offset']
    assert row_bytes%128==0
    return fields,row_bytes,offset+256


def field(buf,expert,spec):
    dtype={'bf16':torch.bfloat16,'mxfp8':torch.float8_e4m3fn,'uint8':torch.uint8}[spec['dtype']]
    return buf[expert,spec['offset']:spec['offset']+spec['bytes']].view(dtype).view(spec['shape'])


def views(buffers,fields,which,precision):
    rr=[field(buf,e,fields['r'+which]) for buf in buffers for e in range(buf.shape[0])]
    if precision=='bf16':return functional.SplitRoutedWeight(data=rr[0],storage_table=ops.make_routed_weight_storage_table_bf16(rr))
    cc=[field(buf,e,fields['c'+which]) for buf in buffers for e in range(buf.shape[0])]
    rs=[field(buf,e,fields['rs'+which]) for buf in buffers for e in range(buf.shape[0])]
    cs=[field(buf,e,fields['cs'+which]) for buf in buffers for e in range(buf.shape[0])]
    return functional.SplitRoutedWeight(data=rr[0],storage_table=ops.make_routed_weight_storage_table_mxfp8(rr),
        scale=rs[0],scale_storage_table=ops.make_routed_scale_storage_table(rs),scale_tensors=tuple(rs),
        transposed_data=cc[0],transposed_scale=cs[0],transposed_storage_table=ops.make_routed_weight_storage_table_mxfp8(cc),
        transposed_scale_storage_table=ops.make_routed_scale_storage_table(cs),transposed_scale_tensors=tuple(cs),native_columnwise=True)


def pack_home(home,fields,rg,ru,rd,precision):
    home.fill_(0x3C)
    if precision=='mxfp8':
        from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
        import transformer_engine_torch as tex
        from bench_ablation import swizzle
        quantizer=MXFP8Quantizer(tex.DType.kFloat8E4M3,rowwise=True,columnwise=True);quantizer.optimize_for_gemm=False
    for e in range(home.shape[0]):
        for suffix,w in [('1',torch.cat((rg[e],ru[e]),dim=0)),('2',rd[e])]:
            if precision=='bf16':field(home,e,fields['r'+suffix]).copy_(w);continue
            q=quantizer.quantize(w.contiguous());m,k=w.shape
            components={'r'+suffix:q._rowwise_data.view(torch.uint8),'c'+suffix:q._columnwise_data.view(torch.uint8),
                        'rs'+suffix:swizzle(q._rowwise_scale_inv.unsqueeze(0),m,k),
                        'cs'+suffix:swizzle(q._columnwise_scale_inv.transpose(-2,-1).unsqueeze(0),k,m)}
            for name,v in components.items():
                spec=fields[name];assert v.numel()==spec['bytes']
                home[e,spec['offset']:spec['offset']+spec['bytes']].copy_(v.contiguous().view(torch.uint8).flatten())


def flip_epoch(home,fields,precision):
    for spec in fields.values():
        if spec['kind']=='data':
            v=home[:,spec['offset']:spec['offset']+spec['bytes']]
            if precision=='bf16':v.view(torch.int16).bitwise_xor_(-32768)
            else:v.bitwise_xor_(0x80)


def fingerprint(tensor):
    local=hashlib.sha256(tensor.cpu().numpy().tobytes()).hexdigest()
    values=[None]*dist.get_world_size();dist.all_gather_object(values,local)
    return hashlib.sha256(''.join(values).encode()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=tuple(MODELS),required=True);p.add_argument('--profile',choices=['ECHO','UltraEP','MoonEP'],required=True);p.add_argument('--plan-seed',type=int,default=20260921);p.add_argument('--precision',choices=['bf16','mxfp8'],default='bf16');p.add_argument('--tokens',type=int,default=4096);p.add_argument('--ep',type=int,default=64);p.add_argument('--smoke',action='store_true')
    p.add_argument('--validate-only',action='store_true');p.add_argument('--output',required=True)
    p.add_argument('--mok-sms',default='24,32,40,48,56,64')
    p.add_argument('--copy-sms',default='24,48,152')
    p.add_argument('--minibatches',default='4096');p.add_argument('--warmup',type=int,default=50)
    p.add_argument('--iterations',type=int,default=100);p.add_argument('--rounds',type=int,default=3)
    a=p.parse_args()
    build_root=Path(os.environ.get('MOK_REPLICA_BUILD_ROOT', str(Path(__file__).resolve().parents[1]/'.build')))
    baseline_so=next((build_root/('baseline_'+a.precision)/'mok').glob('_C*.so'))
    baseline_spec=importlib.util.spec_from_file_location('baseline_mok._C',str(baseline_so))
    baseline_C=importlib.util.module_from_spec(baseline_spec);baseline_spec.loader.exec_module(baseline_C)
    tolerance=BF16_TOLERANCE if a.precision=='bf16' else MXFP8_TOLERANCE
    local_rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(local_rank);device=torch.device('cuda',local_rank)
    dist.init_process_group('nccl',device_id=device,timeout=datetime.timedelta(minutes=8))
    rank,world=dist.get_rank(),dist.get_world_size();assert world==a.ep
    cpus=sorted(os.sched_getaffinity(0));n=int(os.environ.get('LOCAL_WORLD_SIZE','4'))
    if len(cpus)>=n:os.sched_setaffinity(0,cpus[len(cpus)*local_rank//n:len(cpus)*(local_rank+1)//n])
    model=MODELS[a.model];n_home=model['experts']//64
    capacity=8 if a.profile=='MoonEP' else 4;nslots=n_home+capacity
    h,i,t,k=(512,256,2048,model['k']) if a.smoke else (model['h'],model['i'],a.tokens,model['k'])
    e=world*n_home;mac=model['macro']
    assert t*k%(2*n_home*256)==0 and t*k<=mac
    minibatches=[int(v) for v in a.minibatches.split(',')]
    if a.smoke:
        minibatches=[1024];a.warmup=3;a.iterations=5;a.rounds=1
    sm_count=torch.cuda.get_device_properties(device).multi_processor_count
    mok_sms=[int(v) for v in a.mok_sms.split(',') if int(v)<sm_count]
    copy_sms=[int(v) for v in a.copy_sms.split(',') if int(v)<=sm_count]
    if a.smoke:mok_sms=[16,32];copy_sms=[4,32,sm_count]
    out=Path(a.output)
    if rank==0:out.parent.mkdir(parents=True,exist_ok=True)
    def emit(r):
        if rank==0:
            r['timestamp']=datetime.datetime.now(datetime.timezone.utc).isoformat()
            with out.open('a') as f:f.write(json.dumps(r,sort_keys=True)+'\n')
            print(json.dumps({k:v for k,v in r.items() if k not in ('rank_max_samples_ms','per_rank_median_ms','component_rank_max_samples_ms')},sort_keys=True),flush=True)
    counters=collections.Counter()
    def trial(name,us,note):
        if rank==0:
            counters[name]+=1
            r={'ts':datetime.datetime.now().isoformat(timespec='seconds'),'kernel':name,'trial':counters[name],'jit_s':0.0,'gpu_us':us,'verify':'PASS','note':note}
            with (out.parent/'trial_log.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
    emit({'kind':'environment','arguments':vars(a),'world_size':world,'gpu':torch.cuda.get_device_name(device),'sm_count':sm_count,
          'torch':torch.__version__,'numpy':np.__version__,'cuda':torch.version.cuda,'hostname':socket.gethostname(),'job_id':os.environ.get('SLURM_JOB_ID'),
          'model':a.model,'display_name':model['display'],'routed_precision':a.precision.upper(),'weight_payload':'BF16 FC1+FC2' if a.precision=='bf16' else 'MXFP8 FC1+FC2 rowwise and scales','shape':{'hidden':h,'intermediate':i,'tokens':t,'topk':k,'logical_experts':e},'home_experts':n_home,'replica_capacity':capacity,'source':'screenshot-marginal synthetic proxy; NOT real planner replay',
          'original_extension':str(baseline_so),'modified_extension':str(_C.__file__),'profile':a.profile,'minibatches':minibatches,'mok_sms':mok_sms,'copy_sms':copy_sms,'fused':True})
    inp=generate_inputs(rank,device,e,n_home,k,t,h,i)
    x,_,probs,sg,su,sd,rg,ru,rd,_=inp
    all_logical,all_physical,plan=construct(a.profile,world,n_home,t,k,seed=a.plan_seed,smoke=(world==4))
    logical=torch.as_tensor(all_logical[rank*t:(rank+1)*t].copy(),device=device)
    physical=torch.as_tensor(all_physical[rank*t:(rank+1)*t].copy(),device=device)
    del all_logical,all_physical
    n_received=plan['ingress'][rank];home_rows=plan['home_rows'][rank]
    incoming=sorted([edge for edge in plan['edges'] if edge['destination_rank']==rank],key=lambda edge:edge['replica_slot'])
    if rank==0:
        plan_text=json.dumps(plan,indent=2)+'\n'
        (out.parent/'plan.json').write_text(plan_text)
        (out.parent/(out.stem+'_plan.json')).write_text(plan_text)
    emit({'kind':'plan','assumptions':plan['assumptions'],'total_copies':plan['total_copies'],
          'mean_copies':plan['total_copies']/world,'ingress':plan['ingress'],'egress':plan['egress'],
          'replica_tokens_per_copy':plan['replica_tokens_per_copy'],'home_rows':plan['home_rows']})
    logical_fp,physical_fp=fingerprint(logical),fingerprint(physical)
    fields,row_bytes,stride=payload_layout(h,i,a.precision)
    symm_mem.enable_symm_mem_for_group(dist.group.WORLD.group_name)
    home=symm_mem.empty(n_home,stride,dtype=torch.uint8,device=device)
    handle=symm_mem.rendezvous(home,dist.group.WORLD.group_name)
    pack_home(home,fields,rg,ru,rd,a.precision)
    replica=torch.empty((capacity,stride),dtype=torch.uint8,device=device)
    src_ptrs=[int(handle.buffer_ptrs[edge['source_rank']])+edge['source_expert']*stride for edge in incoming]
    copies={'row':transport.CopyContext(src_ptrs,replica,row_bytes)}
    seed_copy=copies['row']
    weight_ready=torch.zeros(1,dtype=torch.int32,device=device)
    resources=copies['row'].resources();assert resources['active_blocks_per_sm']==1,resources
    cfg=functional.MoKConfig(fwd_num_comm_sms=32,bwd_num_comm_sms=32,minibatch_size=minibatches[0],macrobatch_size=mac,schedule_capacity_multiplier=2.0/world)
    ws=functional.get_workspace(cfg,dist.group.WORLD,device=device,num_local_tokens=t,hidden_size=h,topk=k)
    def barrier():ops.barrier_all(ws.barrier_buffer,ws.barrier_buffer_ptrs,ws.barrier_buffer_multicast_ptr,ws.barrier_target)
    torch.cuda.synchronize();dist.barrier()
    emit({'kind':'input','logical_route_fingerprint':logical_fp,'physical_route_fingerprint':physical_fp,'routes_per_rank':t*k,
          'physical_counts':plan['physical_counts'],'row_bytes_per_expert':row_bytes,'allocation_stride_bytes_per_expert':stride,
          'row_receive_bytes_by_rank':[n*row_bytes for n in plan['ingress']],'row_send_bytes_by_rank':[n*row_bytes for n in plan['egress']],'guard_bytes_per_expert':256,'fields':fields,'copy_resources':resources})
    def independent_expected():
        expected=torch.full_like(replica,0xA5);requests=[]
        for edge in plan['edges']:
            if edge['destination_rank']==rank:
                requests.append(dist.P2POp(dist.irecv,expected[edge['replica_slot'],:row_bytes],edge['source_rank']))
            if edge['source_rank']==rank:
                requests.append(dist.P2POp(dist.isend,home[edge['source_expert'],:row_bytes],edge['destination_rank']))
        if requests:
            for work in dist.batch_isend_irecv(requests):work.wait()
        torch.cuda.synchronize();dist.barrier()
        return expected
    expected=independent_expected()
    for payload,copy in copies.items():
        size=row_bytes if payload=='row' else stride
        for sm in copy_sms:
            replica.fill_(0xA5);copy.run(sm);torch.cuda.synchronize()
            assert torch.equal(replica,expected),(payload,sm)
            if payload=='row':assert bool((replica[:,row_bytes:]==0xA5).all().item())
            barrier()
        emit({'kind':'validation','check':'copy bytes and untouched suffix/guard','payload':payload,'copy_sms':copy_sms,'status':'PASS'})
    seed_copy.run(32);barrier();torch.cuda.synchronize()
    home1,home2=views([home],fields,'1',a.precision),views([home],fields,'2',a.precision)
    all1,all2=views([home,replica],fields,'1',a.precision),views([home,replica],fields,'2',a.precision)
    logical_schedule=functional.build_schedule(ws,cfg,logical,num_local_experts=n_home)
    # build_schedule reads a shared all-gather buffer asynchronously. All ranks
    # must finish before any rank overwrites its multicast slice for the next plan.
    barrier()
    physical_schedule=functional.build_schedule(ws,cfg,physical,num_local_experts=nslots)
    barrier()
    def validate_schedule(schedule,requested,nslots,expected_counts):
        gathered=torch.empty((world*t,k),dtype=requested.dtype,device=device)
        dist.all_gather_into_tensor(gathered,requested)
        gathered=gathered.view(world,t*k)
        nrows=int(schedule.num_tokens.item())
        assert nrows==sum(expected_counts)
        assert torch.equal(schedule.tokens_per_expert,torch.tensor(expected_counts,device=device,dtype=schedule.tokens_per_expert.dtype))
        peers=schedule.peer_rank[:nrows].long();indices=schedule.peer_token_idx[:nrows].long()
        assert bool(((peers>=0)&(peers<world)&(indices>=0)&(indices<t*k)).all().item())
        expected_experts=torch.repeat_interleave(torch.arange(nslots,device=device,dtype=requested.dtype)+rank*nslots,schedule.tokens_per_expert.long())
        torch.testing.assert_close(gathered[peers,indices],expected_experts,rtol=0,atol=0)
        assert torch.unique(peers*(t*k)+indices).numel()==nrows
        barrier()
    validate_schedule(logical_schedule,logical,n_home,plan['logical_counts'][rank])
    validate_schedule(physical_schedule,physical,nslots,plan['physical_counts'][rank])
    emit({'kind':'validation','check':'both scheduler outputs match independent routes','status':'PASS'})
    assert int(physical_schedule.num_tokens.item())==t*k
    assert torch.equal(physical_schedule.tokens_per_expert,torch.tensor(plan['physical_counts'][rank],device=device,dtype=physical_schedule.tokens_per_expert.dtype))
    def full(schedule,w1,w2,config):
        y,ctx=functional.forward(config,ws,schedule,x,probs,sg,su,sd,w1,w1,w2,swiglu_limit=10.0 if a.model=='dsv4_pro' else None)
        torch.cuda.synchronize()
        return y,ctx
    ops._C=baseline_C
    ref0,ctx0=full(logical_schedule,home1,home2,cfg);del ctx0
    ops._C=_C
    got,ctx0=full(physical_schedule,all1,all2,cfg)
    check_correctness('replica logical-model parity',ref0,got,tolerance,print_stats=rank==0);del got,ctx0
    # Change the source weight epoch by flipping the payload sign bits.
    flip_epoch(home,fields,a.precision)
    torch.cuda.synchronize();dist.barrier()
    expected=independent_expected()
    ref1,ctx0=full(logical_schedule,home1,home2,cfg);del ctx0
    assert not torch.equal(ref0,ref1),'epoch did not change reference output'
    for payload,copy in copies.items():
        size=row_bytes if payload=='row' else stride
        replica.fill_(0xA5);copy.run(32);barrier();torch.cuda.synchronize()
        assert torch.equal(replica,expected)
        if payload=='row':assert bool((replica[:,row_bytes:]==0xA5).all().item())
        got,ctx0=full(physical_schedule,all1,all2,cfg)
        check_correctness('new epoch '+payload,ref1,got,tolerance,print_stats=rank==0);del got,ctx0
        emit({'kind':'validation','check':'new weight epoch and logical parity','payload':payload,'status':'PASS'})
    flip_epoch(home,fields,a.precision)
    torch.cuda.synchronize();dist.barrier();seed_copy.run(32);barrier();torch.cuda.synchronize()
    del expected,ref1
    def measure(run,setup,parts=False):
        for _ in range(a.warmup):setup();run(None);barrier()
        events=[(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)) for _ in range(a.iterations)]
        barrier()
        for begin,middle,end in events:
            setup();begin.record();run(middle if parts else None);end.record();barrier()
        torch.cuda.synchronize()
        total=[b.elapsed_time(e) for b,m,e in events]
        arrays=[total]
        if parts:arrays.extend(([b.elapsed_time(m) for b,m,e in events],[m.elapsed_time(e) for b,m,e in events]))
        local=torch.tensor(arrays,device=device,dtype=torch.float64)
        gathered=[torch.empty_like(local) for _ in range(world)];dist.all_gather(gathered,local)
        matrix=torch.stack(gathered);worst=matrix.max(dim=0).values
        rec={'median_rank_max_ms':float(torch.quantile(worst[0],.5).item()),'p95_rank_max_ms':float(torch.quantile(worst[0],.95).item()),
             'rank_max_samples_ms':worst[0].cpu().tolist(),'per_rank_median_ms':torch.quantile(matrix[:,0,:],.5,dim=1).cpu().tolist(),
             'warmup':a.warmup,'iterations':a.iterations}
        if parts:
            rec.update(copy_median_rank_max_ms=float(torch.quantile(worst[1],.5).item()),mok_median_rank_max_ms=float(torch.quantile(worst[2],.5).item()),component_rank_max_samples_ms=worst[1:].cpu().tolist())
        return rec
    selected={p:32 for p in copies}
    if not a.validate_only:
        tuning=collections.defaultdict(list)
        for rep in range(a.rounds):
            order=copy_sms if rep%2==0 else list(reversed(copy_sms))
            for sm in order:
                for payload in (list(copies) if rep%2==0 else list(reversed(copies))):
                    rec=measure(lambda middle,payload=payload,sm=sm:copies[payload].run(sm),lambda:None)
                    rec.update(kind='timing',mode='Copy-only',payload=payload,copy_sms=sm,round=rep,receive_bytes_by_rank=[n*row_bytes for n in plan['ingress']])
                    emit(rec);tuning[payload,sm].append(rec['median_rank_max_ms'])
                    trial(a.profile+'_'+a.model+'_'+a.precision+'_copy_'+payload,1000*rec['median_rank_max_ms'],'copy_sms={} round={}'.format(sm,rep))
        for payload in copies:selected[payload]=min(copy_sms,key=lambda sm:statistics.median(tuning[payload,sm]))
        emit({'kind':'selected_copy_sms','values':selected})
    for mb in minibatches:
        # Each rank has a different home boundary. Parameterize the global
        # experiment by prefix fraction, then take complete home minibatches.
        prefix_fractions=[0.0,0.5,1.0]
        def prefix_rows(fraction):
            return int((home_rows//mb)*fraction)*mb
        config=functional.MoKConfig(fwd_num_comm_sms=32,bwd_num_comm_sms=32,minibatch_size=mb,macrobatch_size=mac,schedule_capacity_multiplier=2.0/world)
        # Capture the unchanged binary as a concurrent baseline in this process.
        ops._C=baseline_C;baseline_C.ablation_capture(True)
        old_y,old_ctx=full(physical_schedule,all1,all2,config)
        baseline_C.ablation_capture(False);ops._C=_C
        _C.ablation_capture(True)
        y,ctx=full(physical_schedule,all1,all2,config)
        _C.ablation_capture(False)
        _C.replica_configure(src_ptrs,replica,weight_ready,row_bytes,home_rows)
        resource_values=_C.replica_resources()
        labels=['registers','static_shared_bytes','dynamic_shared_bytes','active_blocks_per_sm','local_bytes','cluster_size']
        resource_info={kind:dict(zip(labels,resource_values[j*6:(j+1)*6])) for j,kind in enumerate(['plain','fused'])}
        assert resource_info['fused']['active_blocks_per_sm']==1,resource_info
        emit({'kind':'resources','minibatch':mb,'values':resource_info})
        torch.testing.assert_close(y,old_y,rtol=0,atol=0)
        torch.testing.assert_close(y,ref0,rtol=0,atol=0)
        local_obs=_C.ablation_observables(False)
        old_obs=baseline_C.ablation_observables(False)
        gold=[local_obs[0].clone(),local_obs[1][:t*k].clone()]
        expected_combine=ws.combine_buffer.clone()
        expected=independent_expected()
        def setup():
            _C.ablation_reset(False);_C.replica_reset()
        def check_core(obs):
            torch.testing.assert_close(obs[0],gold[0],rtol=0,atol=0)
            torch.testing.assert_close(obs[1][:t*k],gold[1],rtol=0,atol=0)
            torch.testing.assert_close(ws.combine_buffer,expected_combine,rtol=0,atol=0)
        for sm in mok_sms:
            setup();_C.ablation_launch(False,0,sm);barrier();torch.cuda.synchronize();check_core(local_obs)
            baseline_C.ablation_reset(False);baseline_C.ablation_launch(False,0,sm)
            barrier();torch.cuda.synchronize();check_core(old_obs)
            for fraction in prefix_fractions:
                prefix=prefix_rows(fraction)
                replica.fill_(0xA5);setup();_C.replica_launch(sm,prefix);barrier();torch.cuda.synchronize()
                assert int(weight_ready.item())==sm
                assert torch.equal(replica,expected)
                assert bool((replica[:,row_bytes:]==0xA5).all().item())
                check_core(local_obs)
        emit({'kind':'validation','check':'original/modified Ready and Fused bitwise parity; poisoned weights; all SM/prefix combinations',
              'minibatch':mb,'mok_sms':mok_sms,'prefix_fractions':prefix_fractions,'prefix_rows_rank0':[prefix_rows(f) for f in prefix_fractions],'status':'PASS'})
        # Force two distinct source epochs through the fused copy, then restore.
        for epoch in (1,0):
            flip_epoch(home,fields,a.precision)
            torch.cuda.synchronize();dist.barrier()
            expected=independent_expected()
            copies['row'].run(32);barrier()
            baseline_C.ablation_reset(False);baseline_C.ablation_launch(False,0,32);barrier();torch.cuda.synchronize()
            epoch_gold=[old_obs[0].clone(),old_obs[1][:t*k].clone()];epoch_combine=ws.combine_buffer.clone()
            if epoch:assert not torch.equal(epoch_combine,expected_combine)
            for sm in mok_sms:
                for fraction in prefix_fractions:
                    prefix=prefix_rows(fraction)
                    replica.fill_(0xA5);setup();_C.replica_launch(sm,prefix);barrier();torch.cuda.synchronize()
                    assert int(weight_ready.item())==sm
                    assert torch.equal(replica,expected)
                    assert bool((replica[:,row_bytes:]==0xA5).all().item())
                    torch.testing.assert_close(local_obs[0],epoch_gold[0],rtol=0,atol=0)
                    torch.testing.assert_close(local_obs[1][:t*k],epoch_gold[1],rtol=0,atol=0)
                    torch.testing.assert_close(ws.combine_buffer,epoch_combine,rtol=0,atol=0)
            emit({'kind':'validation','check':'fused new weight epoch and reset reuse','epoch':epoch,'minibatch':mb,'status':'PASS'})
        # Replay a captured Fused graph repeatedly with changing weight epochs.
        # Reset is part of the graph; a stale completion counter cannot pass it.
        graph=torch.cuda.CUDAGraph()
        setup();_C.replica_launch(32,prefix_rows(1.0));barrier();torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            setup();_C.replica_launch(32,prefix_rows(1.0))
        for epoch in (1,0):
            flip_epoch(home,fields,a.precision)
            torch.cuda.synchronize();dist.barrier()
            copies['row'].run(32);barrier()
            baseline_C.ablation_reset(False);baseline_C.ablation_launch(False,0,32);barrier();torch.cuda.synchronize()
            graph_gold=ws.combine_buffer.clone()
            for replay in range(3):
                replica.fill_(0xA5);graph.replay();barrier();torch.cuda.synchronize()
                assert int(weight_ready.item())==32
                torch.testing.assert_close(ws.combine_buffer,graph_gold,rtol=0,atol=0)
        emit({'kind':'validation','check':'CUDA graph replay with reset and two weight epochs','minibatch':mb,'replays':6,'status':'PASS'})
        del graph,graph_gold,epoch_gold,epoch_combine,expected,gold,expected_combine
        seed_copy.run(32);barrier();torch.cuda.synchronize()
        if not a.validate_only:
            for rep in range(a.rounds):
                sm_order=mok_sms if rep%2==0 else list(reversed(mok_sms))
                for sm in sm_order:
                    modes=[('OriginalReady',None),('Ready',None),('Serial',None),('Comm-only',None),('Compute-only',None)]+[('Fused',f) for f in prefix_fractions]
                    if rep%2:modes.reverse()
                    for mode,prefix in modes:
                        # Fresh real activations for ablation consumers; outside timing.
                        setup();_C.ablation_launch(False,0,sm);barrier()
                        def reset(mode=mode):
                            if mode in ('OriginalReady','Serial'):baseline_C.ablation_reset(False)
                            else:_C.ablation_reset(False)
                            _C.replica_reset()
                        def run(middle,mode=mode,prefix=prefix,sm=sm):
                            if mode=='Serial':copies['row'].run(selected['row']);middle.record() if middle is not None else None
                            if mode in ('OriginalReady','Serial'):baseline_C.ablation_launch(False,0,sm)
                            elif mode=='Fused':_C.replica_launch(sm,prefix_rows(prefix))
                            else:_C.ablation_launch(False,{'Ready':0,'Comm-only':1,'Compute-only':2}[mode],sm)
                        rec=measure(run,reset,parts=mode=='Serial')
                        rec.update(kind='timing',mode=mode,payload='row' if mode in ('Fused','Serial') else 'preloaded',
                                   minibatch=mb,mok_sms=sm,copy_sms=selected['row'] if mode=='Serial' else None,
                                   prefix_fraction=prefix,prefix_rows_rank0=prefix_rows(prefix) if prefix is not None else None,round=rep,active_minibatches=(t*k+mb-1)//mb)
                        emit(rec);trial(a.profile+'_'+a.model+'_'+a.precision+'_'+mode,1000*rec['median_rank_max_ms'],'minibatch={} mok_sms={} prefix={} round={}'.format(mb,sm,prefix,rep))
        _C.ablation_clear();baseline_C.ablation_clear();del local_obs,old_obs,ctx,old_ctx,y,old_y;gc.collect();torch.cuda.empty_cache()
    emit({'kind':'complete','status':'PASS','fused':True});dist.destroy_process_group()

if __name__=='__main__':
    with torch.no_grad():main()
