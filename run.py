"""One-command data preparation and training entry point."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from config import ModelConfig, TrainConfig, load_config
from model import GPT
from prepare_data import manifest_fingerprint, prepare
from train_tokenizer import train_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="configs/data_sources.json")
    parser.add_argument("--model-config", default="configs/model_small.json")
    parser.add_argument("--train-config", default="configs/train.json")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--weights-only", action="store_true")
    parser.add_argument("--rebuild-data", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    manifest_path = ROOT / arguments.manifest
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    model_config = load_config(ModelConfig, ROOT / arguments.model_config)
    train_config = load_config(TrainConfig, ROOT / arguments.train_config)
    manifest_tokenizer = manifest.get("tokenizer_name", train_config.tokenizer_name)
    if manifest_tokenizer != train_config.tokenizer_name:
        raise ValueError(
            f"manifest tokenizer={manifest_tokenizer!r} but training tokenizer="
            f"{train_config.tokenizer_name!r}"
        )
    # The launcher only needs the parameter count here. Building the full model
    # on CPU would initialize 255M parameters and then immediately discard them
    # before train.py constructs the real CUDA model.
    with torch.device("meta"):
        model = GPT(model_config)
    print(
        f"model parameters={model.parameter_count():,} layers={model_config.n_layers} "
        f"GQA={model_config.n_heads}Q/{model_config.n_kv_heads}KV "
        f"train_context={train_config.sequence_length} max_context={model_config.max_seq_len}"
    )
    del model

    if not torch.cuda.is_available() and not arguments.allow_cpu:
        cuda_build = torch.version.cuda or "none (CPU-only PyTorch)"
        raise RuntimeError(
            "CUDA is unavailable. Training this corpus on CPU is impractical. "
            f"Current PyTorch CUDA build: {cuda_build}. Install a CUDA-enabled "
            "PyTorch build, then run the same command again. Use --allow-cpu only "
            "for a tiny smoke test."
        )

    tokenizer_path = ROOT / manifest_tokenizer
    if manifest.get("train_tokenizer_if_missing", False) and not (
        tokenizer_path / "tokenizer.json"
    ).exists():
        print("bilingual tokenizer not found; training it from balanced source samples")
        train_tokenizer(manifest_path, tokenizer_path)

    train_tokens = ROOT / train_config.train_tokens
    val_tokens = ROOT / train_config.val_tokens
    metadata_path = train_tokens.parent / "metadata.json"
    data_is_current = False
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            existing_metadata = json.load(handle)
        data_is_current = (
            existing_metadata.get("tokenizer_name") == manifest_tokenizer
            and existing_metadata.get("vocab_size") == model_config.vocab_size
            and existing_metadata.get("manifest_fingerprint")
            == manifest_fingerprint(manifest)
        )
    if arguments.rebuild_data or not (
        train_tokens.exists() and val_tokens.exists() and metadata_path.exists()
    ) or not data_is_current:
        print("packed tokens not found; starting data preparation")
        prepare(str(manifest_path), str(train_tokens.parent))
    else:
        print("using existing packed token files; pass --rebuild-data to regenerate")

    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if metadata["vocab_size"] != model_config.vocab_size:
            raise ValueError(
                f"tokenizer vocab_size={metadata['vocab_size']} but model "
                f"vocab_size={model_config.vocab_size}"
            )
        print(
            f"data train_tokens={metadata['stats']['train_tokens']:,} "
            f"val_tokens={metadata['stats']['val_tokens']:,}"
        )

    command = [
        sys.executable,
        str(ROOT / "src" / "train.py"),
        "--model-config",
        str(ROOT / arguments.model_config),
        "--train-config",
        str(ROOT / arguments.train_config),
    ]
    if arguments.resume:
        command.extend(("--resume", str(ROOT / arguments.resume)))
    if arguments.weights_only:
        if not arguments.resume:
            raise ValueError("--weights-only requires --resume")
        command.append("--weights-only")
    print("starting training; metrics are also written to runs/*/metrics.jsonl")
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
