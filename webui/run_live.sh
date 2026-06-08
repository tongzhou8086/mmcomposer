#!/usr/bin/env bash
# Serve mmcomposer on a cluster jump node with ON-THE-FLY B200 benchmarking.
#
# On your LAPTOP (only it has a browser), tunnel to the jump node:
#     ssh -L 8501:localhost:8501 <user>@<jumpnode>
# then on the JUMP NODE run this script, and open http://localhost:8501 locally.
#
# The "Benchmark on a B200 (live)" button compiles + runs the generated kernel
# (and cuBLAS) on a real B200 via srun and reports measured TFLOPS — no GPU is
# needed on the jump node itself.
set -euo pipefail
cd "$(dirname "$0")"

export MMCOMPOSER_LIVE=1
# srun flags for the live bench (override if your partition/gres differ):
export MMCOMPOSER_SRUN_ARGS="${MMCOMPOSER_SRUN_ARGS:---partition=dedicated --gres=gpu:nvidia_b200:1 --time=00:10:00}"
PORT="${PORT:-8501}"

exec python -m streamlit run app.py \
    --server.address 127.0.0.1 --server.port "$PORT" --server.headless true
