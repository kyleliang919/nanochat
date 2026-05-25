#!/bin/bash
# Sweep: find M, W config that matches baseline val_bpb (1.076)
# Using 200 iterations each — enough to rank configs via learning curve
# Baseline at step 200: ~1.42 bpb → final 1.076 (ratio 0.76)
# Our M=2 W=8 at step 200: 1.404 → final 1.170 (ratio 0.83)
#
# Configs:
#   1. M=4 W=8:  more memory slots (2.8x slower than M=2 W=8)
#   2. M=2 W=32: wider direct window (3.7x slower)
#   3. M=4 W=16: both dimensions up (5.2x slower)

set -e

export PATH="$HOME/miniconda3/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate nanochat_recurrent

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export PYTORCH_ALLOC_CONF="expandable_segments:True"

cd /home/ubuntu/nanochat

DEPTH=8
ITERS=200

# =============================================================================
echo "[$(date)] SWEEP 1: M=4 W=8 chunk=128"
echo "=============================================="

python -m torch.distributed.run --standalone --nproc_per_node=2 \
    -m scripts.recurrent_train -- \
    --depth=$DEPTH \
    --n_memory_tokens=4 \
    --memory_window=8 \
    --max_seq_len=2048 \
    --chunk_size=128 \
    --device_batch_size=128 \
    --total_batch_size=524288 \
    --num_iterations=$ITERS \
    --eval_every=50 \
    --sample_every=-1 \
    --core_metric_every=-1 \
    --run=sweep_m4_w8 \
    --model_tag=sweep_m4_w8 \
    2>&1 | tee /tmp/sweep_m4_w8.log

echo ""
# =============================================================================
echo "[$(date)] SWEEP 2: M=2 W=32 chunk=128"
echo "=============================================="

python -m torch.distributed.run --standalone --nproc_per_node=2 \
    -m scripts.recurrent_train -- \
    --depth=$DEPTH \
    --n_memory_tokens=2 \
    --memory_window=32 \
    --max_seq_len=2048 \
    --chunk_size=128 \
    --device_batch_size=128 \
    --total_batch_size=524288 \
    --num_iterations=$ITERS \
    --eval_every=50 \
    --sample_every=-1 \
    --core_metric_every=-1 \
    --run=sweep_m2_w32 \
    --model_tag=sweep_m2_w32 \
    2>&1 | tee /tmp/sweep_m2_w32.log

echo ""
# =============================================================================
echo "[$(date)] SWEEP 3: M=4 W=16 chunk=128"
echo "=============================================="

python -m torch.distributed.run --standalone --nproc_per_node=2 \
    -m scripts.recurrent_train -- \
    --depth=$DEPTH \
    --n_memory_tokens=4 \
    --memory_window=16 \
    --max_seq_len=2048 \
    --chunk_size=128 \
    --device_batch_size=128 \
    --total_batch_size=524288 \
    --num_iterations=$ITERS \
    --eval_every=50 \
    --sample_every=-1 \
    --core_metric_every=-1 \
    --run=sweep_m4_w16 \
    --model_tag=sweep_m4_w16 \
    2>&1 | tee /tmp/sweep_m4_w16.log

echo ""
echo "=============================================="
echo "[$(date)] SWEEP COMPLETE"
echo "=============================================="
echo ""
echo "=== RESULTS AT STEP $ITERS ==="
echo "M=2 W=8  (reference): final bpb = 1.170 (from full run)"
echo "M=4 W=8:"
grep "val bpb" /tmp/sweep_m4_w8.log | tail -1
echo "M=2 W=32:"
grep "val bpb" /tmp/sweep_m2_w32.log | tail -1
echo "M=4 W=16:"
grep "val bpb" /tmp/sweep_m4_w16.log | tail -1
echo ""
echo "Baseline at step $ITERS: ~1.42 bpb (final 1.076)"
echo "Lower step-$ITERS bpb = better expected final bpb"
