#!/bin/bash
# Re-run of Experiment 2: RecurrentGPT d8 no-memory (FIXED)
# The bug was: memory_window=8 was being used as attention window even with M=0
# Fix: when M=0, use full causal attention (is_causal=True)

set -e

export PATH="$HOME/miniconda3/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate nanochat_recurrent

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export PYTORCH_ALLOC_CONF="expandable_segments:True"

cd /home/ubuntu/nanochat

DEPTH=8

echo "[$(date)] EXPERIMENT 2 (FIXED): RecurrentGPT d${DEPTH} no-memory (2 GPU, compiled)"
echo "=============================================="

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
    --run=recurrent_d${DEPTH}_nomem_fixed \
    --model_tag=recurrent_d${DEPTH}_nomem_fixed \
    2>&1 | tee /tmp/recurrent_d${DEPTH}_nomem_fixed.log

echo ""
echo "Done! Final result:"
grep "val bpb" /tmp/recurrent_d${DEPTH}_nomem_fixed.log | tail -1
