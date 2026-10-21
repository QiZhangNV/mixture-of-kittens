#!/bin/bash
set -euo pipefail
EXP_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
REPO_ROOT=$(cd -- "$EXP_ROOT/../.." && pwd)
BENCH_PYTHON=${PYTHON:-python3}
BENCH_ARCH=${ARCH:-SM103}
BENCH_PRECISION=${1:-all}
export MOK_REPLICA_BUILD_ROOT=${MOK_REPLICA_BUILD_ROOT:-$EXP_ROOT/.build}
MOK_REPLICA_BUILD_ROOT=$("$BENCH_PYTHON" -c 'import os; from pathlib import Path; print(Path(os.environ["MOK_REPLICA_BUILD_ROOT"]).resolve())')
case "$BENCH_PRECISION" in
    all) BENCH_TARGETS=(baseline_mxfp8 baseline_bf16 fused transport);;
    bf16) BENCH_TARGETS=(baseline_bf16 fused transport);;
    mxfp8) BENCH_TARGETS=(baseline_mxfp8 fused transport);;
    *) echo 'Usage: build.sh [all|bf16|mxfp8]' >&2; exit 2;;
esac
"$BENCH_PYTHON" "$EXP_ROOT/scripts/verify.py"
"$BENCH_PYTHON" "$EXP_ROOT/scripts/prepare.py" "$BENCH_PRECISION"
mkdir -p "$EXP_ROOT/build_logs"
for BENCH_TARGET in "${BENCH_TARGETS[@]}"; do
    case "$BENCH_TARGET" in
        fused) BENCH_DIRECTORY=$REPO_ROOT;;
        transport) BENCH_DIRECTORY=$EXP_ROOT/replica_transport_distribution;;
        *) BENCH_DIRECTORY=$MOK_REPLICA_BUILD_ROOT/$BENCH_TARGET;;
    esac
    BENCH_STARTED=$SECONDS
    if make -C "$BENCH_DIRECTORY" ARCH="$BENCH_ARCH" PYTHON="$BENCH_PYTHON" \
        THUNDERKITTENS_ROOT="$MOK_REPLICA_BUILD_ROOT/ThunderKittens" \
        > "$EXP_ROOT/build_logs/$BENCH_TARGET.log" 2>&1; then
        "$BENCH_PYTHON" "$EXP_ROOT/scripts/record_trial.py" --directory "$EXP_ROOT/build_logs" \
            --kernel "share_build_$BENCH_TARGET" --seconds "$((SECONDS-BENCH_STARTED))" \
            --status PASS --note 'Byte-identical historical device source; branch rebuild'
        echo "PASS $BENCH_TARGET"
    else
        "$BENCH_PYTHON" "$EXP_ROOT/scripts/record_trial.py" --directory "$EXP_ROOT/build_logs" \
            --kernel "share_build_$BENCH_TARGET" --seconds "$((SECONDS-BENCH_STARTED))" \
            --status 'FAIL: build' --note "See build_logs/$BENCH_TARGET.log"
        tail -n 60 "$EXP_ROOT/build_logs/$BENCH_TARGET.log"
        exit 1
    fi
done
