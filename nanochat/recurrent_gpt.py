"""
RecurrentGPT: GPT with two orthogonal architectural extensions.

1. Recurrent memory tokens (n_memory_tokens > 0):
   Each token attends to M memory slots from the previous step, giving each
   token effective access to a full-depth computation over prior context.
   The sequence is interleaved: [mem_0^t, ..., mem_{M-1}^t, token_t, ...].
   Memory slots share RoPE position with their real token (same time step).
   ALiBi bias on memory keys breaks slot symmetry without learned parameters.

   Training uses a two-pass procedure:
     Pass 1 (sequential, no_grad): process block-by-block with rolling KV
       buffer, collect exact recurrent memory states.
     Pass 2 (parallel, with grad): build full interleaved sequence with
       stop-grad memory states, run one forward pass, compute loss.
   No train/test discrepancy; no truncation of the recurrent dependency.

2. Split cross-attention (split_cross_attn=True):
   Second-half layers cross-attend to the first-half's final hidden states
   instead of self-attending. Halves KV cache memory at inference; no change
   in compute. Each second-half layer still has its own Q/K/V projections;
   K and V are simply projected from the first-half snapshot instead of x.

Both features default to off and are fully independent.
"""

from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

from nanochat.common import get_dist_info, print0
from nanochat.muon import Muon, DistMuon
from nanochat.adamw import DistAdamW
from nanochat.fp8_static import LinearFP8


@dataclass
class RecurrentGPTConfig:
    sequence_len: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    # Recurrent memory tokens
    n_memory_tokens: int = 4   # M: memory slots per step (0 = disabled)
    memory_window: int = 8     # W: sliding window size in real-token units
    # Split cross-attention
    split_cross_attn: bool = False  # second half cross-attends to first half output


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(F.relu(self.c_fc(x)).square())


class RecurrentCausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        assert config.n_embd % config.n_head == 0
        assert config.n_kv_head <= config.n_head and config.n_head % config.n_kv_head == 0
        self.c_q = nn.Linear(config.n_embd, config.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

    def forward(self, x_normed, cos_sin, block_mask, score_mod, kv_src_normed=None):
        """Parallel-pass attention via flex_attention.

        kv_src_normed: if provided, K/V are projected from this instead of
        x_normed. Used by second-half layers when split_cross_attn=True.
        """
        B, T_ext, _ = x_normed.size()
        kv_in = kv_src_normed if kv_src_normed is not None else x_normed

        q = self.c_q(x_normed).view(B, T_ext, self.n_head, self.head_dim)
        k = self.c_k(kv_in).view(B, T_ext, self.n_kv_head, self.head_dim)
        v = self.c_v(kv_in).view(B, T_ext, self.n_kv_head, self.head_dim)

        cos, sin = cos_sin
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)

        # flex_attention layout: (B, heads, seq, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        y = flex_attention(
            q, k, v,
            score_mod=score_mod,
            block_mask=block_mask,
            enable_gqa=(self.n_kv_head < self.n_head),
        )
        y = y.transpose(1, 2).contiguous().view(B, T_ext, self.n_embd)
        return self.c_proj(y)


class RecurrentBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = RecurrentCausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin, block_mask, score_mod, kv_src=None):
        kv_src_normed = norm(kv_src) if kv_src is not None else None
        x = x + self.attn(norm(x), cos_sin, block_mask, score_mod, kv_src_normed)
        x = x + self.mlp(norm(x))
        return x


class RecurrentGPT(nn.Module):
    def __init__(self, config: RecurrentGPTConfig, pad_vocab_size_to=64):
        """NOTE: runs in meta device context during __init__; real init in init_weights()."""
        super().__init__()
        self.config = config
        padded_vocab = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab != config.vocab_size:
            print0(f"Padding vocab from {config.vocab_size} to {padded_vocab}")
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab, config.n_embd),
            "h": nn.ModuleList([RecurrentBlock(config) for _ in range(config.n_layer)]),
        })
        self.lm_head = LinearFP8(config.n_embd, padded_vocab, bias=False,
                                  x_scale=100/448, w_scale=1.6/448, monitor=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))

        # RoPE buffers (fake meta tensors here, real data set in init_weights)
        self.rotary_seq_len = config.sequence_len * 10
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        # ALiBi slopes for per-slot bias in memory token attention
        alibi = self._compute_alibi_slopes(config.n_head)
        self.register_buffer("alibi_slopes", alibi, persistent=False)

        self._block_mask_cache: dict = {}

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_alibi_slopes(n_head):
        slopes = [2.0 ** (-(8.0 * (h + 1) / n_head)) for h in range(n_head)]
        return torch.tensor(slopes, dtype=torch.float32)

    def _precompute_rotary(self, seq_len, head_dim, base=10000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        ch = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (ch / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos = freqs.cos().bfloat16()[None, :, None, :]
        sin = freqs.sin().bfloat16()[None, :, None, :]
        return cos, sin

    def init_weights(self):
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        n_embd = self.config.n_embd
        s = 3 ** 0.5 * n_embd ** -0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
        with torch.no_grad():
            self.resid_lambdas.fill_(1.0)
            self.x0_lambdas.fill_(0.0)
        head_dim = self.config.n_embd // self.config.n_head
        self.cos, self.sin = self._precompute_rotary(self.rotary_seq_len, head_dim)
        self.alibi_slopes = self._compute_alibi_slopes(self.config.n_head).to(
            self.transformer.wte.weight.device)
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)

    def get_device(self):
        return self.transformer.wte.weight.device

    def num_scaling_params(self):
        return sum(p.numel() for p in self.parameters())

    def estimate_flops(self):
        M = self.config.n_memory_tokens
        S = M + 1
        W = self.config.memory_window
        T = self.config.sequence_len
        T_ext = T * S
        nparams = sum(p.numel() for p in self.parameters())
        nparams_excl = (self.transformer.wte.weight.numel() +
                        self.resid_lambdas.numel() + self.x0_lambdas.numel())
        h = self.config.n_head
        d = self.config.n_embd // h
        eff_attn = min(W * S, T_ext)
        attn_flops = self.config.n_layer * 12 * h * d * eff_attn
        return 6 * (nparams - nparams_excl) + attn_flops

    # ------------------------------------------------------------------
    # Flex-attention helpers (parallel pass)
    # ------------------------------------------------------------------

    def _get_block_mask(self, T, device):
        key = (T, str(device))
        if key not in self._block_mask_cache:
            self._block_mask_cache[key] = self._create_block_mask(T, device)
        return self._block_mask_cache[key]

    def _create_block_mask(self, T, device):
        M = self.config.n_memory_tokens
        W = self.config.memory_window
        S = M + 1

        if M == 0:
            # Standard causal sliding window
            def mask_fn(b, h, q_idx, kv_idx):
                return (q_idx >= kv_idx) & (q_idx - kv_idx <= W)
            return create_block_mask(mask_fn, B=None, H=None,
                                     Q_LEN=T, KV_LEN=T, device=device)

        def mask_fn(b, h, q_idx, kv_idx):
            q_block = q_idx // S
            q_local = q_idx % S
            kv_block = kv_idx // S
            kv_local = kv_idx % S
            in_window = (q_block - kv_block >= 0) & (q_block - kv_block <= W)
            # memory tokens at step t cannot attend to the real token at step t
            not_blocked = ~((q_block == kv_block) & (q_local < M) & (kv_local == M))
            return in_window & not_blocked

        return create_block_mask(mask_fn, B=None, H=None,
                                 Q_LEN=T * S, KV_LEN=T * S, device=device)

    def _make_score_mod(self):
        """ALiBi per-slot bias: attending to memory key at slot j incurs -slope_h * j."""
        M = self.config.n_memory_tokens
        if M == 0:
            return None
        S = M + 1
        slopes = self.alibi_slopes  # (n_head,) buffer, captured by reference

        def score_mod(score, b, h, q_idx, kv_idx):
            kv_local = kv_idx % S
            is_memory = kv_local < M
            return score - slopes[h] * kv_local * is_memory

        return score_mod

    # ------------------------------------------------------------------
    # Sequential-pass attention helpers
    # ------------------------------------------------------------------

    def _seq_attn_bias(self, ctx_len, S, M, device):
        """
        Additive attention bias for one sequential-pass block step.
        Shape: (1, n_head, S, ctx_len).

        Encodes:
          - Intra-block causality: memory queries (local < M) cannot attend to
            the real token of the same block (last position in current block).
          - ALiBi: memory keys get a per-slot per-head negative bias.
        """
        n_head = self.config.n_head
        bias = torch.zeros(1, n_head, S, ctx_len, device=device)

        if M > 0:
            # Memory queries (0..M-1) cannot see the real token (ctx_len-1)
            bias[:, :, :M, -1] = float('-inf')

        # ALiBi for all K positions: slot index = position % S if < M, else 0
        k_local = torch.arange(ctx_len, device=device) % S          # (ctx_len,)
        is_mem  = (k_local < M).float()                              # (ctx_len,)
        alibi_slot = k_local.float() * is_mem                        # (ctx_len,)
        slopes = self.alibi_slopes                                    # (n_head,)
        # (1, n_head, 1, ctx_len) broadcasts cleanly with bias (1, n_head, S, ctx_len)
        alibi_bias = -(slopes[None, :, None, None] * alibi_slot[None, None, None, :])
        bias = bias + alibi_bias

        return bias  # (1, n_head, S, ctx_len)

    def _seq_block_step(self, x, x0, t, kv_bufs, first_half_out):
        """
        Run one block (M+1 tokens) through all transformer layers in the
        sequential pass, updating kv_bufs in-place and returning the new
        hidden state plus first_half_out snapshot.

        x:             (B, S, n_embd)
        kv_bufs:       list of (K, V) or None, length n_layer
        first_half_out: None or (B, S, n_embd), set after layer half-1

        Returns: x_out, updated_kv_bufs, first_half_out
        """
        B, S, _ = x.shape
        M = self.config.n_memory_tokens
        W = self.config.memory_window
        n_layer = self.config.n_layer
        half = n_layer // 2
        n_head = self.config.n_head
        n_kv_head = self.config.n_kv_head
        head_dim = self.config.n_embd // n_head
        device = x.device

        # RoPE: all S positions in this block share time-step position t
        cos_t = self.cos[:, t:t+1].expand(-1, S, -1, -1)
        sin_t = self.sin[:, t:t+1].expand(-1, S, -1, -1)

        new_kv_bufs = []
        first_half_out_new = None

        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            x_normed = norm(x)

            if self.config.split_cross_attn and i >= half:
                kv_src_normed = norm(first_half_out)
            else:
                kv_src_normed = x_normed

            attn = block.attn

            # Project Q, K, V
            q = attn.c_q(x_normed).view(B, S, n_head, head_dim)
            k = attn.c_k(kv_src_normed).view(B, S, n_kv_head, head_dim)
            v = attn.c_v(kv_src_normed).view(B, S, n_kv_head, head_dim)

            # RoPE + QK-norm
            q = norm(apply_rotary_emb(q, cos_t, sin_t))
            k = norm(apply_rotary_emb(k, cos_t, sin_t))

            # Concatenate with rolling KV buffer
            if kv_bufs[i] is not None:
                K_buf, V_buf = kv_bufs[i]
                k_full = torch.cat([K_buf, k], dim=1)
                v_full = torch.cat([V_buf, v], dim=1)
            else:
                k_full, v_full = k, v

            ctx_len = k_full.size(1)
            attn_bias = self._seq_attn_bias(ctx_len, S, M, device)

            # Expand GQA for SDPA
            if n_kv_head < n_head:
                reps = n_head // n_kv_head
                k_sdpa = k_full.repeat_interleave(reps, dim=2)
                v_sdpa = v_full.repeat_interleave(reps, dim=2)
            else:
                k_sdpa, v_sdpa = k_full, v_full

            # Scaled dot-product attention: (B, heads, seq, head_dim)
            attn_out = F.scaled_dot_product_attention(
                q.transpose(1, 2),
                k_sdpa.transpose(1, 2),
                v_sdpa.transpose(1, 2),
                attn_mask=attn_bias,
                is_causal=False,
            )
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, self.config.n_embd)
            x = x + attn.c_proj(attn_out)
            x = x + block.mlp(norm(x))

            # Roll the KV buffer: keep at most W blocks
            max_buf = W * S
            new_K = k_full[:, -max_buf:] if k_full.size(1) > max_buf else k_full
            new_V = v_full[:, -max_buf:] if v_full.size(1) > max_buf else v_full
            new_kv_bufs.append((new_K, new_V))

            if i == half - 1 and self.config.split_cross_attn:
                first_half_out_new = x

        return x, new_kv_bufs, first_half_out_new

    # ------------------------------------------------------------------
    # Two-pass training forward
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward_sequential(self, idx):
        """
        Sequential pass: process one (M+1)-token block at a time with a rolling
        KV buffer per layer. Returns memory_states (B, T, M, n_embd) where
        memory_states[:, t] holds the M memory tokens to be placed before
        token_t in the parallel pass.

        Returns None when n_memory_tokens == 0 (but still runs split_cross_attn
        KV buffer logic if needed — which is a no-op when M=0 since the parallel
        pass handles split_cross_attn differently).
        """
        B, T = idx.size()
        M = self.config.n_memory_tokens
        S = M + 1
        device = idx.device

        # Initial memory state: zeros
        mem_state = torch.zeros(B, M, self.config.n_embd, dtype=torch.bfloat16, device=device)
        kv_bufs = [None] * self.config.n_layer
        first_half_out = None

        memory_states = []

        for t in range(T):
            # Save memory state for this position (memory BEFORE token t)
            memory_states.append(mem_state.clone() if M > 0 else None)

            # Build block input: [norm(mem_state), norm(wte(token_t))]
            tok_emb = norm(self.transformer.wte(idx[:, t:t+1]))  # (B, 1, n_embd)
            if M > 0:
                x = torch.cat([norm(mem_state), tok_emb], dim=1)  # (B, S, n_embd)
            else:
                x = tok_emb
            x0 = x

            x, kv_bufs, first_half_out = self._seq_block_step(
                x, x0, t, kv_bufs, first_half_out)

            # Extract new memory state from the M memory positions
            if M > 0:
                mem_state = x[:, :M, :]  # (B, M, n_embd), raw residual stream

        if M == 0:
            return None
        return torch.stack(memory_states, dim=1)  # (B, T, M, n_embd)

    def forward_parallel(self, idx, memory_states=None, targets=None, loss_reduction='mean'):
        """
        Parallel pass over the full interleaved sequence.

        memory_states: (B, T, M, n_embd) stop-grad memory from forward_sequential,
                       or None when n_memory_tokens == 0.
        """
        B, T = idx.size()
        M = self.config.n_memory_tokens
        S = M + 1
        n_layer = self.config.n_layer
        half = n_layer // 2
        device = idx.device

        # Build embeddings
        token_embeds = norm(self.transformer.wte(idx))  # (B, T, n_embd)

        if M > 0 and memory_states is not None:
            # Interleave: [mem_0^t, ..., mem_{M-1}^t, token_t] for each t
            mem_normed = norm(memory_states)                        # (B, T, M, n_embd)
            combined = torch.cat(
                [mem_normed, token_embeds.unsqueeze(2)], dim=2)    # (B, T, S, n_embd)
            x = combined.reshape(B, T * S, self.config.n_embd)
            T_ext = T * S
        else:
            x = token_embeds
            T_ext = T

        x0 = x

        # RoPE: all S positions in block t share time-step position t
        assert T <= self.cos.size(1), f"T={T} exceeds rotary cache {self.cos.size(1)}"
        if M > 0:
            block_pos = torch.arange(T, device=device).repeat_interleave(S)
            cos_ext = self.cos[:, block_pos]
            sin_ext = self.sin[:, block_pos]
        else:
            cos_ext = self.cos[:, :T]
            sin_ext = self.sin[:, :T]
        cos_sin = (cos_ext, sin_ext)

        block_mask = self._get_block_mask(T, device)
        score_mod = self._make_score_mod()

        first_half_snapshot = None
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            if self.config.split_cross_attn and i >= half:
                kv_src = first_half_snapshot
            else:
                kv_src = None
            x = block(x, cos_sin, block_mask, score_mod, kv_src=kv_src)
            if i == half - 1 and self.config.split_cross_attn:
                first_half_snapshot = x

        # Extract real-token positions: index M within each block → M, S+M, 2S+M, ...
        if M > 0:
            real_idx = torch.arange(T, device=device) * S + M
            x_real = x[:, real_idx, :]
        else:
            x_real = x

        softcap = 15
        logits = self.lm_head(norm(x_real))
        logits = logits[..., :self.config.vocab_size].float()
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            return F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction=loss_reduction,
            )
        return logits

    def forward(self, idx, targets=None, loss_reduction='mean'):
        """Two-pass training: sequential (no_grad) then parallel (with grad)."""
        memory_states = None
        if self.config.n_memory_tokens > 0:
            memory_states = self.forward_sequential(idx).detach()
        return self.forward_parallel(idx, memory_states, targets, loss_reduction)

    # ------------------------------------------------------------------
    # Optimizers
    # ------------------------------------------------------------------

    def setup_optimizers(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02,
                         weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5):
        model_dim = self.config.n_embd
        ddp, _, _, _ = get_dist_info()

        matrix_params   = list(self.transformer.h.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params  = list(self.lm_head.parameters())
        resid_params    = [self.resid_lambdas]
        x0_params       = [self.x0_lambdas]
        assert (len(list(self.parameters())) ==
                len(matrix_params) + len(embedding_params) + len(lm_head_params) +
                len(resid_params) + len(x0_params))

        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling LR ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")
        adam_groups = [
            dict(params=lm_head_params,  lr=unembedding_lr * dmodel_lr_scale),
            dict(params=embedding_params, lr=embedding_lr   * dmodel_lr_scale),
            dict(params=resid_params,    lr=scalar_lr * 0.01),
            dict(params=x0_params,       lr=scalar_lr),
        ]
        AdamWFactory = DistAdamW if ddp else partial(torch.optim.AdamW, fused=True)
        adamw = AdamWFactory(adam_groups, betas=adam_betas, eps=1e-10, weight_decay=0.0)

        MuonFactory = DistMuon if ddp else Muon
        muon = MuonFactory(matrix_params, lr=matrix_lr, momentum=0.95,
                           weight_decay=weight_decay)

        optimizers = [adamw, muon]
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]
        return optimizers

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """Streaming autoregressive generation via sequential block-by-block forward."""
        assert isinstance(tokens, list) and len(tokens) > 0
        device = self.get_device()
        M = self.config.n_memory_tokens
        S = M + 1

        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)

        mem_state = torch.zeros(1, M, self.config.n_embd, dtype=torch.bfloat16, device=device)
        kv_bufs = [None] * self.config.n_layer
        first_half_out = None

        def step(token_id, t_pos):
            nonlocal mem_state, kv_bufs, first_half_out
            tok_idx = torch.tensor([[token_id]], dtype=torch.long, device=device)
            tok_emb = norm(self.transformer.wte(tok_idx))  # (1, 1, n_embd)
            if M > 0:
                x = torch.cat([norm(mem_state), tok_emb], dim=1)
            else:
                x = tok_emb
            x0 = x
            x, kv_bufs, first_half_out = self._seq_block_step(
                x, x0, t_pos, kv_bufs, first_half_out)
            # Logits from real-token position
            x_real = x[:, M:M+1, :] if M > 0 else x
            softcap = 15
            logits = self.lm_head(norm(x_real))[:, 0, :self.config.vocab_size].float()
            logits = softcap * torch.tanh(logits / softcap)
            if M > 0:
                mem_state = x[:, :M, :]
            return logits  # (1, vocab_size)

        # Process prefix (all but last token, discarding logits)
        for t, tok in enumerate(tokens[:-1]):
            step(tok, t)

        # Last prefix token gives first generation logits
        logits = step(tokens[-1], len(tokens) - 1)
        t_pos = len(tokens)

        for _ in range(max_tokens):
            if top_k is not None:
                v_topk, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v_topk[:, [-1]]] = float('-inf')
            if temperature > 0:
                next_tok = torch.multinomial(
                    F.softmax(logits / temperature, dim=-1),
                    num_samples=1, generator=rng).item()
            else:
                next_tok = torch.argmax(logits, dim=-1).item()
            yield next_tok
            logits = step(next_tok, t_pos)
            t_pos += 1
