#!/bin/bash
# Experiment: RecurrentGPT d8 with memory, compiled sequential pass
# Compare fairly with baseline: same model size, same total tokens (data:param=8)
#
# Config: seq_len=512, M=2, W=8, B=128, 1 GPU (38 GB memory)
# At ~17k tok/sec on 1 GPU, total ~470M tokens:
#   470M / 17k = ~27,600 seconds = ~460 min ← too slow!
#
# Better: seq_len=128, M=2, W=8, B=512, 2 GPU DDP
# At ~45k tok/sec per GPU, 2 GPUs = ~90k tok/sec
#   470M / 90k = ~5200s = ~87 min
#
# Even better: just match tokens with total_batch=524288 and let it run.
# With 2 GPU: each GPU does B=512, seq=128 → tokens/fwdbwd = 512*128*2 = 131072
# grad_accum = 524288/131072 = 4
# Estimated: ~50k tok/sec per GPU with DDP → ~30 min
#
# NOTE: The key comparison is val_bpb at matched training tokens.
# The baseline reached 1.076 bpb after 470M tokens at seq_len=2048.
# We train at seq_len=128 with the same 470M tokens.
# If memory helps, we should see comparable or better bpb despite shorter context.

set -e

export PATH="$HOME/miniconda3/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate nanochat_recurrent

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export PYTORCH_ALLOC_CONF="expandable_segments:True"

cd /home/ubuntu/nanochat

DEPTH=8

echo "[$(date)] RecurrentGPT d${DEPTH} M=2 W=8 compiled (2 GPU)"
echo "=============================================="

python -m torch.distributed.run --standalone --nproc_per_node=2 \
    -m scripts.recurrent_train -- \
    --depth=$DEPTH \
    --n_memory_tokens=2 \
    --memory_window=8 \
    --max_seq_len=128 \
    --device_batch_size=512 \
    --target_param_data_ratio=8 \
    --eval_every=100 \
    --sample_every=-1 \
    --core_metric_every=-1 \
    --run=recurrent_d${DEPTH}_m2_w8_compiled \
    --model_tag=recurrent_d${DEPTH}_m2_w8_compiled \
    2>&1 | tee /tmp/recurrent_d${DEPTH}_m2_w8_compiled.log

echo ""
echo "Done! Final result:"
grep "val bpb" /tmp/recurrent_d${DEPTH}_m2_w8_compiled.log | tail -1
