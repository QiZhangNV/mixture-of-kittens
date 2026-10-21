"""Synthetic plans constrained by screenshot marginals, not real planner replay."""
import json
from pathlib import Path
from collections import deque
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DIGITIZED=ROOT/'docs/planner_dispatch/dsv3_mlperf_ep32_20260921/digitized_marginals.json'


def fit_degrees(shares,mean,n,tail_cap):
    """Fit integer bucket counts; exact rank count and rounded total copies."""
    target=int(round(mean*n));dp={(0,0):(0.0,())}
    for bucket in range(5):
        nxt={};expected=n*shares[bucket]
        for (used,base),(cost,counts) in dp.items():
            max_count=n-used
            if bucket:max_count=min(max_count,target//bucket)
            if shares[bucket]==0:max_count=0
            for count in range(max_count+1):
                new_base=base+bucket*count
                if new_base>target:break
                score=cost+(count-expected)**2/max(expected,1.0)
                key=(used+count,new_base)
                if key not in nxt or score<nxt[key][0]:nxt[key]=(score,counts+(count,))
        dp=nxt
    candidates=[]
    for (used,base),(cost,counts) in dp.items():
        if used==n and base<=target<=base+max(0,tail_cap-4)*counts[4]:candidates.append((cost,counts,base))
    assert candidates,('cannot fit',shares,mean,n,tail_cap)
    cost,counts,base=min(candidates);degrees=[]
    for bucket,count in enumerate(counts):degrees.extend([bucket]*count)
    extra=target-base;tail=[i for i,d in enumerate(degrees) if d>=4]
    # Unknown 4+ tail: minimally spread additional copies across this bucket.
    for j in range(extra):degrees[tail[j%len(tail)]]+=1
    assert sum(degrees)==target and max(degrees,default=0)<=tail_cap
    return np.array(degrees,dtype=np.int64),dict(bucket_counts=list(counts),target_total_copies=target,actual_mean=target/n,fit_cost=cost,tail_assumption='minimum-spread integers, capped at '+str(tail_cap))


def make_edges(tx,rx,rng):
    """Simple directed bipartite realization, no self/duplicate rank pairs."""
    n=len(tx);source=2*n;sink=source+1;adj=[[] for _ in range(2*n+2)]
    def edge(a,b,c):
        adj[a].append([b,len(adj[b]),c]);adj[b].append([a,len(adj[a])-1,0]);return len(adj[a])-1
    refs=[]
    for r in range(n):edge(source,r,int(tx[r]));edge(n+r,sink,int(rx[r]))
    for src in rng.permutation(n):
        for dst in rng.permutation(n):
            if src!=dst:refs.append((int(src),int(dst),edge(int(src),n+int(dst),1)))
    total=0
    while True:
        level=[-1]*len(adj);level[source]=0;q=deque([source])
        while q:
            v=q.popleft()
            for u,rev,cap in adj[v]:
                if cap and level[u]<0:level[u]=level[v]+1;q.append(u)
        if level[sink]<0:break
        it=[0]*len(adj)
        def push(v,f):
            if v==sink:return f
            while it[v]<len(adj[v]):
                e=adj[v][it[v]];u,rev,cap=e
                if cap and level[u]==level[v]+1:
                    sent=push(u,min(f,cap))
                    if sent:e[2]-=sent;adj[u][rev][2]+=sent;return sent
                it[v]+=1
            return 0
        while True:
            f=push(source,10**9)
            if not f:break
            total+=f
    assert total==sum(tx)==sum(rx),(total,tx,rx)
    return [(src,dst) for src,dst,index in refs if adj[src][index][2]==0]


def construct(profile,world,home_experts,tokens,topk,seed=20260921,smoke=False):
    rng=np.random.default_rng(seed);capacity=8 if profile=='MoonEP' else 4
    rows=tokens*topk;quantum=256;replica_tokens=rows//(2*home_experts)
    assert replica_tokens%quantum==0
    if smoke:
        assert world==4
        pairs={'ECHO':[(0,1),(0,2),(1,2)],'UltraEP':[(0,1)],'MoonEP':[(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]}[profile]
        tx=np.bincount([s for s,d in pairs],minlength=world);rx=np.bincount([d for s,d in pairs],minlength=world)
        fits={'kind':'EP4 correctness toy, not distribution fit'}
    else:
        data=json.loads(DIGITIZED.read_text())['rows'];hist={}
        for direction in ['egress','ingress']:
            r=next(x for x in data if x['planner']==profile and x['step']==45 and x['direction']==direction and (profile=='MoonEP' or x['budget_value']==4))
            hist[direction]=r
        out_degree,fo=fit_degrees(hist['egress']['shares_0_1_2_3_4plus'],hist['egress']['mean_from_label'],world,8)
        in_degree,fi=fit_degrees(hist['ingress']['shares_0_1_2_3_4plus'],hist['ingress']['mean_from_label'],world,capacity)
        order=rng.permutation(world);tx=np.zeros(world,dtype=np.int64);rx=np.zeros_like(tx)
        tx[order]=np.sort(out_degree)[::-1];rx[order]=np.sort(in_degree)
        pairs=make_edges(tx,rx,rng)
        fits={'egress':fo,'ingress':fi,'screenshot_marginals':hist}
    assert max(rx)<=capacity
    slots=home_experts+capacity;counts=np.zeros((world,slots),dtype=np.int64)
    for rank in range(world):
        home_rows=rows-int(rx[rank])*replica_tokens;assert home_rows>=0
        tiles=home_rows//quantum;q,rem=divmod(tiles,home_experts)
        counts[rank,:home_experts]=(q+(np.arange(home_experts)<rem))*quantum
        counts[rank,home_experts:home_experts+int(rx[rank])]=replica_tokens
    logical_counts=counts[:,:home_experts].copy();incoming=[[] for _ in range(world)];edges=[]
    # Hot-rank fan-out: use one home expert where top-k uniqueness permits it.
    for src,dst in sorted(pairs):
        anchor=(src+seed)%home_experts
        choices=[(anchor+j)%home_experts for j in range(home_experts)]
        expert=next(j for j in choices if logical_counts[src,j]+replica_tokens<=world*tokens)
        logical_counts[src,expert]+=replica_tokens
        slot=len(incoming[dst]);e=dict(source_rank=int(src),source_expert=int(expert),destination_rank=int(dst),replica_slot=slot,tokens=replica_tokens)
        incoming[dst].append(e);edges.append(e)
    assert np.all(counts.sum(axis=1)==rows)
    assert np.max(logical_counts)<=world*tokens and np.max(logical_counts.sum(axis=1))<=2*rows
    n_tokens=world*tokens;assignments=[];physical=[]
    for src in range(world):
        for expert in range(home_experts):
            logical_id=src*home_experts+expert
            destinations=[(src*slots+expert,int(counts[src,expert]))]
            destinations += [(e['destination_rank']*slots+home_experts+e['replica_slot'],e['tokens']) for e in edges if e['source_rank']==src and e['source_expert']==expert]
            assert sum(v for p,v in destinations)==logical_counts[src,expert]
            for physical_id,n in destinations:
                assignments.append(np.full(n,logical_id,dtype=np.int32));physical.append(np.full(n,physical_id,dtype=np.int32))
    logical=np.concatenate(assignments).reshape(topk,n_tokens).T
    physical=np.concatenate(physical).reshape(topk,n_tokens).T
    permutation=rng.permutation(n_tokens);logical=logical[permutation];physical=physical[permutation]
    order=np.argsort(rng.random((n_tokens,topk)),axis=1)
    logical=np.ascontiguousarray(np.take_along_axis(logical,order,axis=1));physical=np.ascontiguousarray(np.take_along_axis(physical,order,axis=1))
    assert np.all(np.diff(np.sort(logical,axis=1),axis=1)>0)
    mapping=np.full(world*slots,-1,dtype=np.int32)
    for rank in range(world):mapping[rank*slots:rank*slots+home_experts]=np.arange(rank*home_experts,(rank+1)*home_experts)
    for e in edges:mapping[e['destination_rank']*slots+home_experts+e['replica_slot']]=e['source_rank']*home_experts+e['source_expert']
    assert np.array_equal(mapping[physical],logical)
    assert np.array_equal(np.bincount(physical.ravel(),minlength=world*slots).reshape(world,slots),counts)
    assert np.array_equal(np.bincount(logical.ravel(),minlength=world*home_experts).reshape(world,home_experts),logical_counts)
    metadata=dict(profile=profile,source='synthetic proxy fitted to screenshots; NOT real planner replay',step=45,chart_budget='B=EPN=8' if profile=='MoonEP' else 'n_slot=4',world=world,seed=seed,home_experts=home_experts,replica_capacity=capacity,ingress=rx.tolist(),egress=tx.tolist(),total_copies=len(edges),replica_tokens_per_copy=replica_tokens,post_plan_rows_per_rank=rows,home_rows=counts[:,:home_experts].sum(axis=1).tolist(),physical_counts=counts.tolist(),logical_counts=logical_counts.tolist(),edges=edges,fit=fits,
                  assumptions=['per-rank post-plan token work balanced','per-replica tokens = half original average home-expert tokens','egress/ingress ranks paired in opposite sorted order (assumed, not observed)','single-hot-expert fan-out where logical top-k uniqueness permits','4+ tail minimally spread with cap8','fixed sampled plan repeated; all active replicas copied each call'])
    return logical,physical,metadata
