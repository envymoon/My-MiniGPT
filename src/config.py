"""Small, dependency-free configuration helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, TypeVar


@dataclass
class ModelConfig:
    vocab_size: int = 50_304
    max_seq_len: int = 1_024
    dim: int = 1024
    n_layers: int = 18
    n_heads: int = 16
    n_kv_heads: int = 4
    ffn_hidden_dim: int | None = 2_816
    ffn_multiple_of: int = 256
    rope_theta: float = 10_000.0
    norm_eps: float = 1e-5
    dropout: float = 0.0
    qk_norm: bool = True
    # auto selects PyTorch's fused SDPA on CUDA and the explicit reference path
    # on CPU. Set explicit when studying the score/mask/softmax mechanics.
    attention_backend: str = "auto"  # explicit | sdpa | auto
    gradient_checkpointing: bool = True

    # Optional, educational sparse feed-forward path.
    moe_num_experts: int = 0
    moe_top_k: int = 2
    moe_every_n_layers: int = 2
    moe_aux_loss_weight: float = 0.01

    # QAT inserts fake quantization in Linear layers. It does not make training
    # faster; it teaches the weights to tolerate the requested quantization.
    qat: bool = False
    qat_weight_bits: int = 8
    qat_activation_bits: int = 8
    qat_group_size: int = 64

    def validate(self) -> None:
        if self.dim % self.n_heads != 0:
            raise ValueError("dim must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if (self.dim // self.n_heads) % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        if self.attention_backend not in {"explicit", "sdpa", "auto"}:
            raise ValueError("attention_backend must be explicit, sdpa, or auto")
        if self.moe_num_experts and not 1 <= self.moe_top_k <= self.moe_num_experts:
            raise ValueError("moe_top_k must be in [1, moe_num_experts]")
        if self.moe_every_n_layers < 1:
            raise ValueError("moe_every_n_layers must be positive")
        if self.qat_weight_bits < 2 or self.qat_activation_bits < 2:
            raise ValueError("QAT bit widths must be at least 2")


@dataclass
class TrainConfig:
    train_tokens: str = "data/processed/train.bin"
    val_tokens: str = "data/processed/val.bin"
    output_dir: str = "runs/minigpt-255m"
    tokenizer_name: str = "tokenizers/minigpt-bilingual-50304"
    sequence_length: int = 512
    batch_size: int = 2
    gradient_accumulation_steps: int = 32
    total_steps: int = 20_000
    warmup_steps: int = 500
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    precision: str = "auto"  # auto | fp32 | fp16 | bf16
    num_workers: int = 2
    seed: int = 42
    log_every: int = 1
    eval_every: int = 250
    eval_batches: int = 50
    save_every: int = 50_000
    keep_checkpoint_history: bool = False
    qat_start_step: int = 15_000
    # CUDA Graphs are enabled as a policy, but training falls back to eager
    # autograd because this loop has accumulation, checkpointing, and streams.
    cuda_graphs: str = "auto"  # auto | off

    def validate(self) -> None:
        if self.precision not in {"auto", "fp32", "fp16", "bf16"}:
            raise ValueError("precision must be auto, fp32, fp16, or bf16")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if self.cuda_graphs not in {"auto", "off"}:
            raise ValueError("cuda_graphs must be auto or off")
        if self.sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        if self.total_steps < 1 or self.eval_batches < 1:
            raise ValueError("total_steps and eval_batches must be positive")
        if self.log_every < 1 or self.eval_every < 1 or self.save_every < 1:
            raise ValueError("log_every, eval_every, and save_every must be positive")
        if not 0 <= self.warmup_steps < self.total_steps:
            raise ValueError("warmup_steps must be smaller than total_steps")


ConfigT = TypeVar("ConfigT", ModelConfig, TrainConfig)


def from_dict(cls: type[ConfigT], values: dict[str, Any]) -> ConfigT:
    known = {field.name for field in fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {sorted(unknown)}")
    config = cls(**values)
    config.validate()
    return config


def load_config(cls: type[ConfigT], path: str | Path) -> ConfigT:
    with Path(path).open("r", encoding="utf-8") as handle:
        return from_dict(cls, json.load(handle))


def config_dict(config: ModelConfig | TrainConfig) -> dict[str, Any]:
    return asdict(config)
