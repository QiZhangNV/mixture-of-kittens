#!/bin/bash
set -euo pipefail
EXP_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
REPO_ROOT=$(cd -- "$EXP_ROOT/../.." && pwd)
BENCH_PYTHON=${PYTHON:-python3}
BENCH_NNODES=${NNODES:-1}
BENCH_NPROC=${NPROC_PER_NODE:-4}
BENCH_NODE_RANK=${NODE_RANK:-${SLURM_NODEID:-0}}
export MOK_REPLICA_BUILD_ROOT=${MOK_REPLICA_BUILD_ROOT:-$EXP_ROOT/.build}
MOK_REPLICA_BUILD_ROOT=$("$BENCH_PYTHON" -c 'import os; from pathlib import Path; print(Path(os.environ["MOK_REPLICA_BUILD_ROOT"]).resolve())')
export PYTHONPATH="$EXP_ROOT/replica_transport_distribution:$REPO_ROOT:$EXP_ROOT/benchmarks${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export NVLINK_DOMAIN_SIZE=${NVLINK_DOMAIN_SIZE:-72}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NVSHMEM_REMOTE_TRANSPORT=${NVSHMEM_REMOTE_TRANSPORT:-none}
export PYTHONUNBUFFERED=1
BENCH_LAUNCH=(--nnodes="$BENCH_NNODES" --nproc-per-node="$BENCH_NPROC")
if [[ "$BENCH_NNODES" == 1 ]]; then
    BENCH_LAUNCH+=(--standalone)
else
    BENCH_LAUNCH+=(--node-rank="$BENCH_NODE_RANK" --master-addr="${MASTER_ADDR:?Set MASTER_ADDR for multi-node}" --master-port="${MASTER_PORT:-29500}")
fi
exec "$BENCH_PYTHON" -m torch.distributed.run "${BENCH_LAUNCH[@]}" \
    "$EXP_ROOT/benchmarks/bench_distribution_replica.py" --ep "$((BENCH_NNODES*BENCH_NPROC))" "$@"
