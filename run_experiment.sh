#!/bin/bash
# Recurrent GPT vs Baseline GPT comparison experiment
# Model: d8 (512 dim, 8 layers, ~58.7M params)
# Hardware: 2x RTX A6000 (49GB each)
# Logs to wandb project "nanochat"
#
# Comparison strategy:
#   Exp 1: Baseline GPT d8 — standard transformer, compiled, 2 GPUs, data:param=8
#   Exp 2: RecurrentGPT d8 (no memory) — same arch as baseline but through RecurrentGPT path
#   Exp 3: RecurrentGPT d8 (with memory) — recurrence enabled, shorter seq due to speed constraints
#
# All experiments log to wandb for easy comparison of val/bpb curves.

set -e

export PATH="$HOME/miniconda3/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate nanochat_recurrent

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export PYTORCH_ALLOC_CONF="expandable_segments:True"

cd /home/ubuntu/nanochat

DEPTH=8

# =============================================================================
echo "[$(date)] EXPERIMENT 1: Baseline GPT d${DEPTH} (2 GPU, compiled)"
echo "=============================================="
# ~58.7M params, data:param=8 => ~470M tokens => 896 iters
# Expected: ~2 min

python -m torch.distributed.run --standalone --nproc_per_node=2 \
    -m scripts.baseline_train -- \
    --depth=$DEPTH \
    --target_param_data_ratio=8 \
    --device_batch_size=16 \
    --max_seq_len=2048 \
    --eval_every=100 \
    --run=baseline_d${DEPTH} \
    --model_tag=baseline_d${DEPTH} \
    2>&1 | tee /tmp/baseline_d${DEPTH}.log

echo ""
# =============================================================================
echo "[$(date)] EXPERIMENT 2: RecurrentGPT d${DEPTH} no-memory (2 GPU, compiled)"
echo "=============================================="
# Same params, same tokens. Tests RecurrentGPT code path without recurrence.
# Expected: ~4 min (SDPA with explicit mask is slower than is_causal=True)

python -m torch.distributed.run --standalone --nproc_per_node=2 \
    -m scripts.recurrent_train -- \
    --depth=$DEPTH \
    --n_memory_tokens=0 \
    --memory_window=8 \
    --target_param_data_ratio=8 \
    --device_batch_size=16 \
    --max_seq_len=2048 \
    --eval_every=100 \
    --sample_every=-1 \
    --core_metric_every=-1 \
    --run=recurrent_d${DEPTH}_nomem \
    --model_tag=recurrent_d${DEPTH}_nomem \
    2>&1 | tee /tmp/recurrent_d${DEPTH}_nomem.log

echo ""
# =============================================================================
echo "[$(date)] EXPERIMENT 3: RecurrentGPT d${DEPTH} with memory (1 GPU, uncompiled)"
echo "=============================================="
# Recurrence enabled: n_memory_tokens=2, window=4, seq_len=64
# Sequential pass is O(T*layers) per micro-step — fundamentally slow without compile.
# Uses small batch to keep grad_accum reasonable.
# 200 iterations * 8192 tokens/iter = 1.6M tokens (much less than exps 1&2).
# We compare val_bpb at matched steps to see if memory helps convergence speed.
# Expected: ~60-90 min

python -m scripts.recurrent_train \
    --depth=$DEPTH \
    --n_memory_tokens=2 \
    --memory_window=4 \
    --num_iterations=200 \
    --device_batch_size=4 \
    --total_batch_size=8192 \
    --max_seq_len=64 \
    --eval_every=25 \
    --eval_tokens=$((10*8192)) \
    --sample_every=-1 \
    --core_metric_every=-1 \
    --run=recurrent_d${DEPTH}_m2_w4 \
    --model_tag=recurrent_d${DEPTH}_m2_w4 \
    2>&1 | tee /tmp/recurrent_d${DEPTH}_m2_w4.log

echo ""
echo "=============================================="
echo "[$(date)] ALL EXPERIMENTS COMPLETE"
echo "=============================================="
echo ""
echo "=== FINAL RESULTS ==="
echo "Baseline (d8, 2048 seq, ~470M tok):"
grep "val bpb" /tmp/baseline_d${DEPTH}.log | tail -1
echo ""
echo "RecurrentGPT no-mem (d8, 2048 seq, ~470M tok):"
grep "val bpb" /tmp/recurrent_d${DEPTH}_nomem.log | tail -1
echo ""
echo "RecurrentGPT with-mem (d8, 64 seq, ~1.6M tok):"
grep "val bpb" /tmp/recurrent_d${DEPTH}_m2_w4.log | tail -1
echo ""
echo "NOTE: Experiments 1&2 are directly comparable (same tokens, same params)."
echo "Experiment 3 uses much fewer tokens due to sequential pass overhead."
echo "Compare val_bpb curves on wandb for the fairest analysis."
