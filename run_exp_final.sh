#!/bin/bash
# Final experiment: RecurrentGPT d8 M=2 W=8, compiled sequential pass
#
# Fair comparison with baseline:
#   - Same model size (58.7M params)
#   - Same total tokens (470M, data:param=8)
#   - Same context length (2048 tokens per sequence)
#   - Same dataloader (B sequences of 2048 tokens)
#
# The recurrent model chunks each 2048-token sequence into 16 chunks of 128,
# carrying memory across chunks within the same sequence. Memory resets at
# sequence boundaries. RoPE positions 0..2047 per sequence.
#
# Expected: ~2.9 hours at ~45k tok/sec on 2 GPUs
# Baseline comparison: 1.076 bpb in 18 min at 165k tok/sec

set -e

export PATH="$HOME/miniconda3/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate nanochat_recurrent

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export PYTORCH_ALLOC_CONF="expandable_segments:True"

cd /home/ubuntu/nanochat

DEPTH=8

echo "[$(date)] RecurrentGPT d${DEPTH} M=2 W=8 chunk=128 seq=2048 (2 GPU, compiled)"
echo "=============================================="

python -m torch.distributed.run --standalone --nproc_per_node=2 \
    -m scripts.recurrent_train -- \
    --depth=$DEPTH \
    --n_memory_tokens=2 \
    --memory_window=8 \
    --max_seq_len=2048 \
    --chunk_size=128 \
    --device_batch_size=128 \
    --target_param_data_ratio=8 \
    --eval_every=100 \
    --sample_every=-1 \
    --core_metric_every=-1 \
    --run=recurrent_d${DEPTH}_m2_w8_ctx2048 \
    --model_tag=recurrent_d${DEPTH}_m2_w8_ctx2048 \
    2>&1 | tee /tmp/recurrent_d${DEPTH}_m2_w8_ctx2048.log

echo ""
echo "Done! Final result:"
grep "val bpb" /tmp/recurrent_d${DEPTH}_m2_w8_ctx2048.log | tail -1
