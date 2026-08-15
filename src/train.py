"""Single-device pretraining with AMP, accumulation, evaluation, and resume."""

from __future__ import annotations

import argparse
import json
import math
import random
import signal
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

from config import ModelConfig, TrainConfig, load_config
from dataset import get_dataloaders
from model import GPT
from quantization import set_qat_enabled


_STOP_REQUESTED = False


def request_graceful_stop(_signum: int, _frame: object) -> None:
    """Finish the current optimizer step, then write a resumable checkpoint."""

    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    print("stop requested; finishing the current optimizer step before checkpointing")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", default="configs/model_small.json")
    parser.add_argument("--train-config", default="configs/train.json")
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--weights-only",
        action="store_true",
        help="load model weights but reset optimizer/scheduler/step for a new phase",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_precision(requested: str, device: torch.device) -> str:
    if requested != "auto":
        return requested
    if device.type == "cuda":
        return "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    return "fp32"


def autocast_context(device: torch.device, precision: str):
    if precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


def make_lr_scheduler(optimizer: torch.optim.Optimizer, config: TrainConfig) -> LambdaLR:
    minimum_ratio = config.min_learning_rate / config.learning_rate

    def multiplier(step: int) -> float:
        if step < config.warmup_steps:
            return (step + 1) / max(1, config.warmup_steps)
        progress = (step - config.warmup_steps) / max(
            1, config.total_steps - config.warmup_steps
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return minimum_ratio + (1.0 - minimum_ratio) * cosine

    return LambdaLR(optimizer, multiplier)


def make_optimizer(model: GPT, config: TrainConfig) -> torch.optim.AdamW:
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (decay if parameter.ndim >= 2 and "token_embedding" not in name else no_decay).append(parameter)
    groups = [
        {"params": decay, "weight_decay": config.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        groups,
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        fused=torch.cuda.is_available(),
    )


@torch.no_grad()
def evaluate(
    model: GPT,
    loader,
    device: torch.device,
    precision: str,
    max_batches: int,
    moe_weight: float,
) -> float:
    was_training = model.training
    model.eval()
    losses = []
    for batch_index, tokens in enumerate(loader):
        if batch_index >= max_batches:
            break
        tokens = tokens.to(device, non_blocking=True)
        with autocast_context(device, precision):
            output = model(tokens[:, :-1])
            language_loss = F.cross_entropy(
                output.logits.reshape(-1, model.vocab_size), tokens[:, 1:].reshape(-1)
            )
            loss = language_loss + moe_weight * output.auxiliary_loss
        losses.append(loss.float())
    model.train(was_training)
    return torch.stack(losses).mean().item()


def save_checkpoint(
    path: Path,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler: torch.amp.GradScaler,
    step: int,
    model_config: ModelConfig,
    train_config: TrainConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "step": step,
            "model_config": asdict(model_config),
            "train_config": asdict(train_config),
            "rng_state": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "python": random.getstate(),
                "numpy": np.random.get_state(),
            },
        },
        temporary,
    )
    temporary.replace(path)


def main() -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = False
    signal.signal(signal.SIGINT, request_graceful_stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, request_graceful_stop)
    arguments = parse_args()
    model_config = load_config(ModelConfig, arguments.model_config)
    train_config = load_config(TrainConfig, arguments.train_config)
    if train_config.sequence_length > model_config.max_seq_len:
        raise ValueError("train sequence_length cannot exceed model max_seq_len")
    seed_everything(train_config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        # Keep GEMM selection explicit in this low-level implementation. This
        # enables TF32 tensor-core math for FP32 utility paths without changing
        # the requested AMP dtype or model parameter storage.
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    precision = resolve_precision(train_config.precision, device)
    if device.type == "cpu" and precision == "fp16":
        raise ValueError("fp16 training is not supported on CPU; use fp32 or bf16")

    train_loader, val_loader = get_dataloaders(
        train_config.train_tokens,
        train_config.val_tokens,
        train_config.sequence_length,
        train_config.batch_size,
        train_config.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = GPT(model_config).to(device)
    optimizer = make_optimizer(model, train_config)
    scheduler = make_lr_scheduler(optimizer, train_config)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and precision == "fp16"
    )
    start_step = 0

    if arguments.weights_only and not arguments.resume:
        raise ValueError("--weights-only requires --resume")
    if arguments.resume:
        checkpoint_state = torch.load(arguments.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint_state["model"])
        if arguments.weights_only:
            print("loaded checkpoint weights; optimizer, scheduler, and step were reset")
        else:
            optimizer.load_state_dict(checkpoint_state["optimizer"])
            scheduler.load_state_dict(checkpoint_state["scheduler"])
            scaler.load_state_dict(checkpoint_state["scaler"])
            start_step = checkpoint_state["step"] + 1
            if "rng_state" in checkpoint_state:
                rng_state = checkpoint_state["rng_state"]
                if isinstance(rng_state, dict):
                    torch.set_rng_state(rng_state["torch"])
                    random.setstate(rng_state["python"])
                    np.random.set_state(rng_state["numpy"])
                    if torch.cuda.is_available() and rng_state["cuda"] is not None:
                        torch.cuda.set_rng_state_all(rng_state["cuda"])
                else:  # compatibility with the first refactored checkpoint format
                    torch.set_rng_state(rng_state)

    output_dir = Path(train_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    tokens_per_step = (
        train_config.batch_size
        * train_config.gradient_accumulation_steps
        * train_config.sequence_length
    )
    print(
        f"device={device} precision={precision} parameters={model.parameter_count():,} "
        f"attention={model_config.attention_backend} context={train_config.sequence_length} "
        f"batch={train_config.batch_size} accumulation={train_config.gradient_accumulation_steps} "
        f"tokens/step={tokens_per_step:,} workers={train_config.num_workers} "
        f"cuda_graphs={train_config.cuda_graphs}"
    )
    if train_config.cuda_graphs == "auto":
        print(
            "cuda_graphs=auto: eager training fallback is active because the "
            "loop uses gradient accumulation and activation checkpointing; "
            "fixed-shape inference capture is available in src/cuda_graph.py"
        )
    model.train()
    train_iterator = iter(train_loader)
    optimizer.zero_grad(set_to_none=True)
    last_time = time.perf_counter()
    last_logged_step = start_step - 1
    last_completed_step = start_step - 1

    for step in range(start_step, train_config.total_steps):
        completed_steps = step + 1
        qat_active = model_config.qat and completed_steps >= train_config.qat_start_step
        set_qat_enabled(model, qat_active)
        # Keep these reductions on-device. Calling .item() for every
        # micro-batch forces a CUDA synchronization 64 times per optimizer
        # step in the previous configuration.
        accumulated_language_loss = torch.zeros((), device=device, dtype=torch.float32)
        accumulated_auxiliary_loss = torch.zeros((), device=device, dtype=torch.float32)

        for _ in range(train_config.gradient_accumulation_steps):
            try:
                tokens = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                tokens = next(train_iterator)
            tokens = tokens.to(device, non_blocking=True)
            with autocast_context(device, precision):
                output = model(tokens[:, :-1])
                language_loss = F.cross_entropy(
                    output.logits.reshape(-1, model.vocab_size), tokens[:, 1:].reshape(-1)
                )
                loss = language_loss + model_config.moe_aux_loss_weight * output.auxiliary_loss
                scaled_loss = loss / train_config.gradient_accumulation_steps
            scaler.scale(scaled_loss).backward()
            accumulated_language_loss += language_loss.detach().float()
            accumulated_auxiliary_loss += output.auxiliary_loss.detach().float()

        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        last_completed_step = step

        if completed_steps % train_config.log_every == 0:
            now = time.perf_counter()
            elapsed = now - last_time
            last_time = now
            steps_in_interval = step - last_logged_step
            last_logged_step = step
            language_loss_value = (
                accumulated_language_loss / train_config.gradient_accumulation_steps
            ).item()
            auxiliary_loss_value = (
                accumulated_auxiliary_loss / train_config.gradient_accumulation_steps
            ).item()
            metric = {
                "step": completed_steps,
                "train_loss": language_loss_value,
                "aux_loss": auxiliary_loss_value,
                "total_loss": language_loss_value
                + model_config.moe_aux_loss_weight * auxiliary_loss_value,
                "learning_rate": scheduler.get_last_lr()[0],
                "gradient_norm": float(gradient_norm),
                "qat": qat_active,
                "tokens_per_second": tokens_per_step * steps_in_interval / max(elapsed, 1e-9),
                "peak_vram_gb": (
                    torch.cuda.max_memory_allocated() / 1024**3
                    if device.type == "cuda"
                    else 0.0
                ),
            }
            print(json.dumps(metric))
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metric) + "\n")

        if completed_steps % train_config.eval_every == 0:
            validation_loss = evaluate(
                model,
                val_loader,
                device,
                precision,
                train_config.eval_batches,
                model_config.moe_aux_loss_weight,
            )
            validation_metric = {
                "step": completed_steps,
                "validation_loss": validation_loss,
            }
            print(json.dumps(validation_metric))
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(validation_metric) + "\n")

        if completed_steps % train_config.save_every == 0:
            save_checkpoint(
                output_dir / "latest.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                step,
                model_config,
                train_config,
            )
            if train_config.keep_checkpoint_history:
                save_checkpoint(
                    output_dir / f"step_{completed_steps:07d}.pt",
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    step,
                    model_config,
                    train_config,
                )

        if _STOP_REQUESTED:
            break

    if last_completed_step >= start_step:
        save_checkpoint(
            output_dir / "latest.pt",
            model,
            optimizer,
            scheduler,
            scaler,
            last_completed_step,
            model_config,
            train_config,
        )


if __name__ == "__main__":
    main()
