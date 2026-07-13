"""
Train RecurrentGPT model. From root directory of the project, run as:

    python -m scripts.recurrent_train

or distributed as:

    torchrun --nproc_per_node=8 -m scripts.recurrent_train

Quick smoke-test (CPU):
    python -m scripts.recurrent_train --depth=4 --max_seq_len=128 \
        --device_batch_size=1 --eval_tokens=512 --core_metric_every=-1 \
        --total_batch_size=512 --num_iterations=20 \
        --n_memory_tokens=2 --memory_window=4
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import argparse
import time
from contextlib import nullcontext

import wandb
import torch

from nanochat.recurrent_gpt import RecurrentGPT, RecurrentGPTConfig
from nanochat.dataloader import tokenizing_distributed_data_loader, tokenizing_distributed_data_loader_with_state
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir, autodetect_device_type
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine
from scripts.base_eval import evaluate_model
print_banner()

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Pretrain RecurrentGPT model")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb)")
# Runtime
parser.add_argument("--device_type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
# Model architecture
parser.add_argument("--depth", type=int, default=20)
parser.add_argument("--aspect_ratio", type=int, default=64)
parser.add_argument("--head_dim", type=int, default=128)
parser.add_argument("--max_seq_len", type=int, default=2048)
# Recurrent memory
parser.add_argument("--n_memory_tokens", type=int, default=4,
                    help="number of recurrent memory slots per step (0 = disabled)")
parser.add_argument("--memory_window", type=int, default=8,
                    help="sliding window size in real-token units")
parser.add_argument("--chunk_size", type=int, default=128,
                    help="chunk size for sequential pass (max_seq_len must be divisible by this)")
# Split cross-attention
parser.add_argument("--split_cross_attn", action="store_true", default=False,
                    help="second half of layers cross-attend to first half output")
# Parallel K-pass unroll
parser.add_argument("--parallel_unroll", action="store_true", default=False,
                    help="use K parallel encoder passes + non-recurrent decoder instead of sequential two-pass")
parser.add_argument("--n_memory_passes", type=int, default=2,
                    help="K: number of parallel encoder passes (hops of recurrence)")
parser.add_argument("--no_detach_memory", dest="detach_memory", action="store_false", default=True,
                    help="backprop through all K passes (default: detach memory carry between passes)")
parser.add_argument("--encoder_window", type=int, default=0,
                    help="first-half memory attention window in real-token blocks (0 = full attention)")
parser.add_argument("--decoder_window", type=int, default=0,
                    help="second-half decoder attention window in tokens (0 = full attention)")
# Training horizon
parser.add_argument("--num_iterations", type=int, default=-1)
parser.add_argument("--target_flops", type=float, default=-1.0)
parser.add_argument("--target_param_data_ratio", type=int, default=8)
# Optimization
parser.add_argument("--device_batch_size", type=int, default=32)
parser.add_argument("--total_batch_size", type=int, default=524288)
parser.add_argument("--embedding_lr", type=float, default=0.3)
parser.add_argument("--unembedding_lr", type=float, default=0.004)
parser.add_argument("--weight_decay", type=float, default=0.2)
parser.add_argument("--matrix_lr", type=float, default=0.02)
parser.add_argument("--scalar_lr", type=float, default=0.5)
parser.add_argument("--adam_beta1", type=float, default=0.8)
parser.add_argument("--adam_beta2", type=float, default=0.95)
parser.add_argument("--warmup_ratio", type=float, default=0.0)
parser.add_argument("--warmdown_ratio", type=float, default=0.4)
parser.add_argument("--final_lr_frac", type=float, default=0.0)
parser.add_argument("--resume_from_step", type=int, default=-1)
# Evaluation
parser.add_argument("--eval_every", type=int, default=250)
parser.add_argument("--eval_tokens", type=int, default=20*524288)
parser.add_argument("--core_metric_every", type=int, default=2000)
parser.add_argument("--core_metric_max_per_task", type=int, default=500)
parser.add_argument("--sample_every", type=int, default=2000)
parser.add_argument("--save_every", type=int, default=-1)
# Output
parser.add_argument("--model_tag", type=str, default=None)
args = parser.parse_args()
user_config = vars(args).copy()
# -----------------------------------------------------------------------------

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0

use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat", name=args.run, config=user_config)

tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

num_layers = args.depth
model_dim = args.depth * args.aspect_ratio

def find_num_heads(model_dim, target_head_dim):
    ideal = max(1, round(model_dim / target_head_dim))
    for offset in range(model_dim):
        for candidate in [ideal + offset, ideal - offset]:
            if candidate > 0 and model_dim % candidate == 0:
                return candidate
    return 1

num_heads = find_num_heads(model_dim, args.head_dim)
num_kv_heads = num_heads
print0(f"num_layers: {num_layers}, model_dim: {model_dim}, num_heads: {num_heads}")
print0(f"n_memory_tokens: {args.n_memory_tokens}, memory_window: {args.memory_window}, split_cross_attn: {args.split_cross_attn}")
if args.parallel_unroll:
    print0(f"parallel_unroll: K={args.n_memory_passes}, detach_memory={args.detach_memory}, "
           f"encoder_window={args.encoder_window}, decoder_window={args.decoder_window}")

if args.n_memory_tokens > 0:
    assert args.max_seq_len % args.chunk_size == 0, \
        f"max_seq_len ({args.max_seq_len}) must be divisible by chunk_size ({args.chunk_size})"
    chunks_per_seq = args.max_seq_len // args.chunk_size
    print0(f"Chunked recurrent: {chunks_per_seq} chunks of {args.chunk_size} per sequence of {args.max_seq_len}")
else:
    chunks_per_seq = 1

tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
assert args.total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = args.total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Total batch size {args.total_batch_size:,} => grad accum steps: {grad_accum_steps}")

batch_lr_scale = 1.0
reference_batch_size = 2**19
batch_ratio = args.total_batch_size / reference_batch_size
if batch_ratio != 1.0:
    batch_lr_scale = batch_ratio ** 0.5
    print0(f"Scaling LRs by {batch_lr_scale:.4f} for batch size {args.total_batch_size:,}")

weight_decay_scaled = args.weight_decay * (12 / args.depth) ** 2
if args.depth != 12:
    print0(f"Scaling weight decay from {args.weight_decay:.6f} to {weight_decay_scaled:.6f}")

# -----------------------------------------------------------------------------
# Model
model_config_kwargs = dict(
    sequence_len=args.max_seq_len,
    vocab_size=vocab_size,
    n_layer=num_layers,
    n_head=num_heads,
    n_kv_head=num_kv_heads,
    n_embd=model_dim,
    n_memory_tokens=args.n_memory_tokens,
    memory_window=args.memory_window,
    split_cross_attn=args.split_cross_attn,
    parallel_unroll=args.parallel_unroll,
    n_memory_passes=args.n_memory_passes,
    detach_memory=args.detach_memory,
    encoder_window=args.encoder_window,
    decoder_window=args.decoder_window,
)
with torch.device("meta"):
    model_config = RecurrentGPTConfig(**model_config_kwargs)
    model = RecurrentGPT(model_config)
model.to_empty(device=device)
model.init_weights()

base_dir = get_base_dir()
output_dirname = args.model_tag if args.model_tag else f"recurrent_d{args.depth}"
checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
resuming = args.resume_from_step != -1
if resuming:
    print0(f"Resuming from step {args.resume_from_step}")
    model_data, optimizer_data, meta_data = load_checkpoint(
        checkpoint_dir, args.resume_from_step, device, load_optimizer=True, rank=ddp_rank)
    model.load_state_dict(model_data, strict=True, assign=True)
    del model_data

orig_model = model
if args.n_memory_tokens == 0:
    model = torch.compile(model, dynamic=False)
else:
    # Sequential pass now uses static-shape KV buffers — compile the inner step
    # function (fixed shapes every call) and the parallel pass separately.
    orig_model._seq_step_static = torch.compile(orig_model._seq_step_static, dynamic=False)
    orig_model.forward_parallel = torch.compile(orig_model.forward_parallel, dynamic=True)
num_params = sum(p.numel() for p in model.parameters())
print0(f"Number of parameters: {num_params:,} (scaling: {orig_model.num_scaling_params():,})")
num_flops_per_token = orig_model.estimate_flops()
print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

assert args.num_iterations > 0 or args.target_param_data_ratio > 0 or args.target_flops > 0
if args.num_iterations > 0:
    num_iterations = args.num_iterations
    print0(f"Using user-provided number of iterations: {num_iterations:,}")
elif args.target_flops > 0:
    num_iterations = round(args.target_flops / (num_flops_per_token * args.total_batch_size))
    print0(f"Calculated num_iterations from target FLOPs: {num_iterations:,}")
else:
    target_tokens = args.target_param_data_ratio * orig_model.num_scaling_params()
    num_iterations = target_tokens // args.total_batch_size
    print0(f"Calculated num_iterations from data:param ratio: {num_iterations:,}")
total_tokens = args.total_batch_size * num_iterations
print0(f"Total tokens: {total_tokens:,} | tokens:params = {total_tokens / orig_model.num_scaling_params():.2f}")

# -----------------------------------------------------------------------------
# Optimizers
adam_betas = (args.adam_beta1, args.adam_beta2)
optimizers = orig_model.setup_optimizers(
    unembedding_lr=args.unembedding_lr * batch_lr_scale,
    embedding_lr=args.embedding_lr * batch_lr_scale,
    matrix_lr=args.matrix_lr * batch_lr_scale,
    weight_decay=weight_decay_scaled,
    adam_betas=adam_betas,
    scalar_lr=args.scalar_lr * batch_lr_scale,
)
adamw_optimizer, muon_optimizer = optimizers

if resuming:
    for opt, dat in zip(optimizers, optimizer_data):
        opt.load_state_dict(dat)
    del optimizer_data

# -----------------------------------------------------------------------------
# DataLoaders
tokens_dir = os.path.join(base_dir, "tokenized_data")
dataloader_resume_state_dict = None if not resuming else meta_data["dataloader_state_dict"]
train_loader = tokenizing_distributed_data_loader_with_state(
    tokenizer, args.device_batch_size, args.max_seq_len,
    split="train", device=device, resume_state_dict=dataloader_resume_state_dict)
build_val_loader = lambda: tokenizing_distributed_data_loader(
    tokenizer, args.device_batch_size, args.max_seq_len, split="val", device=device)
x, y, dataloader_state_dict = next(train_loader)

# -----------------------------------------------------------------------------
# LR / momentum / weight-decay schedulers (identical to base_train)
def get_lr_multiplier(it):
    warmup_iters   = round(args.warmup_ratio   * num_iterations)
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * args.final_lr_frac

def get_muon_momentum(it):
    frac = min(it / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95

def get_weight_decay(it):
    return weight_decay_scaled * (1 - it / num_iterations)

# -----------------------------------------------------------------------------
# Loop state
carry_state = None  # Persistent memory state carried across micro-batches
if not resuming:
    step = 0
    val_bpb = None
    min_val_bpb = float("inf")
    smooth_train_loss = 0
    total_training_time = 0
else:
    step = meta_data["step"]
    loop_state = meta_data["loop_state"]
    val_bpb = meta_data["val_bpb"]
    min_val_bpb = loop_state["min_val_bpb"]
    smooth_train_loss = loop_state["smooth_train_loss"]
    total_training_time = loop_state["total_training_time"]

# -----------------------------------------------------------------------------
# Training loop
while True:
    last_step = step == num_iterations
    flops_so_far = num_flops_per_token * args.total_batch_size * step

    # Validation bpb
    if args.eval_every > 0 and (last_step or step % args.eval_every == 0):
        model.eval()
        val_loader = build_val_loader()
        eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size)
        with autocast_ctx:
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(f"Step {step:05d} | val bpb: {val_bpb:.6f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log({"step": step, "total_training_flops": flops_so_far,
                       "total_training_time": total_training_time, "val/bpb": val_bpb})
        model.train()

    # CORE metric
    results = {}
    if args.core_metric_every > 0 and (last_step or (step > 0 and step % args.core_metric_every == 0)):
        model.eval()
        with autocast_ctx:
            results = evaluate_model(orig_model, tokenizer, device,
                                     max_per_task=args.core_metric_max_per_task)
        print0(f"Step {step:05d} | CORE: {results['core_metric']:.4f}")
        wandb_run.log({"step": step, "total_training_flops": flops_so_far,
                       "core_metric": results["core_metric"],
                       "centered_results": results["centered_results"]})
        model.train()

    # Sampling
    if args.sample_every > 0 and master_process and (last_step or (step > 0 and step % args.sample_every == 0)):
        model.eval()
        prompts = [
            "The capital of France is",
            "The chemical symbol of gold is",
            "If yesterday was Friday, then tomorrow will be",
            "The opposite of hot is",
        ]
        engine = Engine(orig_model, tokenizer)
        for prompt in prompts:
            tokens_list = tokenizer(prompt, prepend="<|bos|>")
            with autocast_ctx:
                sample, _ = engine.generate_batch(tokens_list, num_samples=1,
                                                   max_tokens=16, temperature=0)
            print0(tokenizer.decode(sample[0]))
        model.train()

    # Checkpoint (save at most 2: midpoint and final)
    midpoint = num_iterations // 2
    should_save = last_step or (step == midpoint and step > 0)
    if should_save:
        save_checkpoint(
            checkpoint_dir, step,
            orig_model.state_dict(),
            [opt.state_dict() for opt in optimizers],
            {
                "step": step,
                "val_bpb": val_bpb,
                "model_config": model_config_kwargs,
                "user_config": user_config,
                "device_batch_size": args.device_batch_size,
                "max_seq_len": args.max_seq_len,
                "dataloader_state_dict": dataloader_state_dict,
                "loop_state": {
                    "min_val_bpb": min_val_bpb,
                    "smooth_train_loss": smooth_train_loss,
                    "total_training_time": total_training_time,
                },
            },
            rank=ddp_rank,
        )

    if last_step:
        break

    # -------------------------------------------------------------------------
    # Training step: two-pass (sequential no_grad + parallel with grad)
    synchronize()
    t0 = time.time()
    total_chunks = grad_accum_steps * chunks_per_seq
    for micro_step in range(grad_accum_steps):
        if args.n_memory_tokens > 0:
            # Chunk the (B, max_seq_len) batch on T dimension, carry memory within
            carry_state = None  # reset at start of each 2048-token sequence
            for chunk_idx in range(chunks_per_seq):
                t_start = chunk_idx * args.chunk_size
                t_end = t_start + args.chunk_size
                x_chunk = x[:, t_start:t_end].contiguous()
                y_chunk = y[:, t_start:t_end].contiguous()
                with autocast_ctx:
                    loss, carry_state = orig_model.forward_with_carry(
                        x_chunk, y_chunk, carry_state=carry_state)
                train_loss = loss.detach()
                (loss / total_chunks).backward()
        else:
            with autocast_ctx:
                loss = model(x, y)
            train_loss = loss.detach()
            (loss / grad_accum_steps).backward()
        x, y, dataloader_state_dict = next(train_loader)

    lrm = get_lr_multiplier(step)
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm
    for group in muon_optimizer.param_groups:
        group["momentum"] = get_muon_momentum(step)
        group["weight_decay"] = get_weight_decay(step)
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
    synchronize()
    t1 = time.time()
    dt = t1 - t0
    # -------------------------------------------------------------------------

    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item()
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(args.total_batch_size / dt)
    flops_per_sec = num_flops_per_token * args.total_batch_size / dt
    promised_flops = 989e12 * ddp_world_size
    mfu = 100 * flops_per_sec / promised_flops
    if step > 10:
        total_training_time += dt
    steps_done = step - 10
    if steps_done > 0:
        avg_time = total_training_time / steps_done
        eta_str = f" | eta: {(num_iterations - step) * avg_time / 60:.1f}m"
    else:
        eta_str = ""
    print0(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | "
           f"lrm: {lrm:.2f} | dt: {dt*1000:.2f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.2f}{eta_str}")
    if step % 100 == 0:
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
        })

    step += 1

print0(f"Peak memory: {get_max_memory() / 1024 / 1024:.2f} MiB")
print0(f"Total training time: {total_training_time / 60:.2f}m")
if val_bpb is not None:
    print0(f"Final val bpb: {val_bpb:.6f} | Best val bpb: {min_val_bpb:.6f}")
compute_cleanup()
