"""A ground-up decoder-only Transformer.

The default attention is grouped-query attention (GQA). Setting n_kv_heads to
n_heads gives MHA; setting it to 1 gives MQA, without changing the algorithm.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from config import ModelConfig
from quantization import QATLinear


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accumulate the variance in fp32 even during mixed-precision training.
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = x * torch.rsqrt(variance + self.eps).to(dtype=x.dtype)
        return normalized * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, theta: float) -> None:
        super().__init__()
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        inverse_frequencies = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        frequencies = torch.outer(positions, inverse_frequencies)
        self.register_buffer("cos", frequencies.cos(), persistent=False)
        self.register_buffer("sin", frequencies.sin(), persistent=False)

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cos[:seq_len], self.sin[:seq_len]


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to x shaped [batch, heads, time, head_dim]."""
    x_float = x.float().reshape(*x.shape[:-1], -1, 2)
    even, odd = x_float.unbind(dim=-1)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
    return rotated.flatten(-2).to(dtype=x.dtype)


def repeat_kv(x: torch.Tensor, repeats: int) -> torch.Tensor:
    """Map KV heads to query heads without storing learned duplicate projections."""
    if repeats == 1:
        return x
    batch, kv_heads, time, head_dim = x.shape
    return (
        x[:, :, None, :, :]
        .expand(batch, kv_heads, repeats, time, head_dim)
        .reshape(batch, kv_heads * repeats, time, head_dim)
    )


def _linear(config: ModelConfig, in_dim: int, out_dim: int) -> nn.Linear:
    if not config.qat:
        return nn.Linear(in_dim, out_dim, bias=False)
    return QATLinear(
        in_dim,
        out_dim,
        bias=False,
        weight_bits=config.qat_weight_bits,
        activation_bits=config.qat_activation_bits,
        group_size=config.qat_group_size,
    )


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.dim // config.n_heads
        self.kv_repeats = config.n_heads // config.n_kv_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = _linear(config, config.dim, config.n_heads * self.head_dim)
        self.k_proj = _linear(config, config.dim, config.n_kv_heads * self.head_dim)
        self.v_proj = _linear(config, config.dim, config.n_kv_heads * self.head_dim)
        self.out_proj = _linear(config, config.dim, config.dim)
        self.q_norm = RMSNorm(self.head_dim, config.norm_eps) if config.qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim, config.norm_eps) if config.qk_norm else nn.Identity()
        self.attention_dropout = nn.Dropout(config.dropout)
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, time, _ = x.shape
        q = self.q_proj(x).view(batch, time, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, time, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, time, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(self.q_norm(q), cos, sin)
        k = apply_rope(self.k_norm(k), cos, sin)
        k = repeat_kv(k, self.kv_repeats)
        v = repeat_kv(v, self.kv_repeats)

        backend = self.config.attention_backend
        if backend == "auto":
            backend = "sdpa" if q.is_cuda else "explicit"
        if backend == "sdpa":
            # PyTorch dispatches to flash/memory-efficient SDPA on CUDA when
            # available. The operation has a native backward, unlike the
            # inference-only paged KV-cache kernel in miniVLLM.
            context = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=~causal_mask,
                dropout_p=self.attention_dropout.p if self.training else 0.0,
                is_causal=False,
            )
        else:
            # Kept explicit for study: [B,H,T,D] @ [B,H,D,T] -> [B,H,T,T].
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            scores = scores.masked_fill(causal_mask, torch.finfo(scores.dtype).min)
            probabilities = F.softmax(scores.float(), dim=-1).to(dtype=q.dtype)
            probabilities = self.attention_dropout(probabilities)
            context = torch.matmul(probabilities, v)
        context = context.transpose(1, 2).contiguous().view(batch, time, -1)
        return self.residual_dropout(self.out_proj(context))


def _ffn_hidden_dim(config: ModelConfig) -> int:
    if config.ffn_hidden_dim is not None:
        return config.ffn_hidden_dim
    # SwiGLU has three matrices; 8d/3 roughly matches the parameter count of a
    # classic two-matrix 4d GELU MLP. Round for accelerator-friendly shapes.
    hidden = int(8 * config.dim / 3)
    return config.ffn_multiple_of * math.ceil(hidden / config.ffn_multiple_of)


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        hidden_dim = _ffn_hidden_dim(config)
        self.gate_proj = _linear(config, config.dim, hidden_dim)
        self.up_proj = _linear(config, config.dim, hidden_dim)
        self.down_proj = _linear(config, hidden_dim, config.dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class SparseMoE(nn.Module):
    """Simple token-choice top-k MoE, intentionally written without fused kernels."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.num_experts = config.moe_num_experts
        self.top_k = config.moe_top_k
        self.router = nn.Linear(config.dim, self.num_experts, bias=False)
        self.experts = nn.ModuleList(SwiGLU(config) for _ in range(self.num_experts))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        router_probabilities = F.softmax(self.router(flat).float(), dim=-1)
        routing_weights, selected_experts = torch.topk(
            router_probabilities, self.top_k, dim=-1
        )
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        output = torch.zeros_like(flat)

        # This loop exposes routing mechanics. A production MoE would use a
        # grouped GEMM, capacity management, expert parallelism, and fused kernels.
        for expert_id, expert in enumerate(self.experts):
            token_index, slot_index = torch.where(selected_experts == expert_id)
            if token_index.numel() == 0:
                continue
            expert_output = expert(flat[token_index])
            weights = routing_weights[token_index, slot_index].to(expert_output.dtype)
            output.index_add_(0, token_index, expert_output * weights[:, None])

        assignment = F.one_hot(selected_experts, self.num_experts).float().sum(dim=1)
        assignment = assignment / self.top_k
        load_balance_loss = self.num_experts * torch.sum(
            router_probabilities.mean(dim=0) * assignment.mean(dim=0)
        )
        return output.view(shape), load_balance_loss


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig, layer_index: int) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)
        self.attention = GroupedQueryAttention(config)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        use_moe = (
            config.moe_num_experts > 0
            and (layer_index + 1) % config.moe_every_n_layers == 0
        )
        self.feed_forward = SparseMoE(config) if use_moe else SwiGLU(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + self.attention(self.attention_norm(x), cos, sin, causal_mask)
        ffn_input = self.ffn_norm(x)
        if isinstance(self.feed_forward, SparseMoE):
            ffn_output, auxiliary_loss = self.feed_forward(ffn_input)
        else:
            ffn_output = self.feed_forward(ffn_input)
            auxiliary_loss = x.new_zeros(())
        return x + ffn_output, auxiliary_loss


@dataclass
class GPTOutput:
    logits: torch.Tensor
    auxiliary_loss: torch.Tensor


class GPT(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.vocab_size = config.vocab_size
        head_dim = config.dim // config.n_heads
        self.token_embedding = nn.Embedding(config.vocab_size, config.dim)
        self.rotary = RotaryEmbedding(
            head_dim, config.max_seq_len, config.rope_theta
        )
        self.blocks = nn.ModuleList(
            TransformerBlock(config, index) for index in range(config.n_layers)
        )
        self.final_norm = RMSNorm(config.dim, config.norm_eps)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.gradient_checkpointing = config.gradient_checkpointing
        self.register_buffer(
            "causal_mask",
            torch.ones(config.max_seq_len, config.max_seq_len, dtype=torch.bool).triu(1),
            persistent=False,
        )
        self.apply(self._initialize_weights)

    def _initialize_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor) -> GPTOutput:
        _, time = input_ids.shape
        if time > self.config.max_seq_len:
            raise ValueError(
                f"sequence length {time} exceeds max_seq_len={self.config.max_seq_len}"
            )
        x = self.token_embedding(input_ids)
        cos, sin = self.rotary(time)
        causal_mask = self.causal_mask[:time, :time][None, None, :, :]
        total_auxiliary_loss = x.new_zeros(())

        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x, auxiliary_loss = checkpoint(
                    block, x, cos, sin, causal_mask, use_reentrant=False
                )
            else:
                x, auxiliary_loss = block(x, cos, sin, causal_mask)
            total_auxiliary_loss = total_auxiliary_loss + auxiliary_loss

        logits = self.lm_head(self.final_norm(x))
        return GPTOutput(logits=logits, auxiliary_loss=total_auxiliary_loss)

    def parameter_count(self, exclude_embeddings: bool = False) -> int:
        count = sum(parameter.numel() for parameter in self.parameters())
        if exclude_embeddings:
            count -= self.token_embedding.weight.numel()
        return count
