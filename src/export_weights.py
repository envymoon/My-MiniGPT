"""Export a compact, weights-only artifact for inference or Hub hosting.

Training checkpoints also contain AdamW, scheduler, AMP, and RNG state.  This
script deliberately removes those fields and optionally stores floating-point
weights as FP16/BF16 so the published artifact stays small without changing
the architecture or tokenizer.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--dtype",
        choices=("fp32", "fp16", "bf16"),
        default="fp16",
        help="storage dtype for floating-point model tensors (default: fp16)",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    checkpoint_path = Path(arguments.checkpoint)
    output_path = Path(arguments.output)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model" not in state or "model_config" not in state:
        raise ValueError("checkpoint must contain model and model_config fields")

    dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[arguments.dtype]
    model_state = {
        name: value.detach().cpu().to(dtype=dtype)
        if torch.is_floating_point(value)
        else value.detach().cpu()
        for name, value in state["model"].items()
    }
    artifact = {
        "format": "minigpt-weights-v1",
        "dtype": arguments.dtype,
        "model_config": state["model_config"],
        "train_config": state.get("train_config", {}),
        "source_step": state.get("step"),
        "model": model_state,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output_path)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"exported {arguments.dtype} weights-only artifact: {output_path} ({size_mb:.1f} MiB)")


if __name__ == "__main__":
    main()
