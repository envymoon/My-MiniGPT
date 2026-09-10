"""Autoregressive sampling from a training checkpoint."""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from config import ModelConfig, from_dict
from model import GPT
from quantization import set_qat_enabled


def apply_repetition_penalty(
    logits: torch.Tensor, previous_tokens: torch.Tensor, penalty: float
) -> torch.Tensor:
    token_ids = torch.unique(previous_tokens)
    selected = logits[:, token_ids]
    selected = torch.where(selected < 0, selected * penalty, selected / penalty)
    logits[:, token_ids] = selected
    return logits


def filter_logits(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    if top_k > 0:
        threshold = torch.topk(logits, min(top_k, logits.shape[-1]), dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < threshold, float("-inf"))
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cumulative > top_p
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False
        original_order_remove = torch.zeros_like(remove).scatter(1, sorted_indices, remove)
        logits = logits.masked_fill(original_order_remove, float("-inf"))
    return logits


@torch.inference_mode()
def generate(
    model: GPT,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    eos_token_id: int | None,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    for _ in range(max_new_tokens):
        model_input = input_ids[:, -model.config.max_seq_len :]
        logits = model(model_input).logits[:, -1, :] / temperature
        logits = apply_repetition_penalty(logits, input_ids, repetition_penalty)
        logits = filter_logits(logits, top_k, top_p)
        next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
        input_ids = torch.cat((input_ids, next_token), dim=1)
        if eos_token_id is not None and torch.all(next_token == eos_token_id):
            break
    return input_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--qat", action="store_true", help="simulate QAT numerics while sampling")
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(arguments.checkpoint, map_location=device, weights_only=False)
    model_config = from_dict(ModelConfig, state["model_config"])
    stored_dtype = state.get("dtype")
    model_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }.get(stored_dtype)
    # Keep CPU inference in fp32 for broad operator support. CUDA can retain
    # the published storage dtype and therefore the smaller runtime footprint.
    if device.type == "cuda" and model_dtype is not None:
        model = GPT(model_config).to(device=device, dtype=model_dtype)
    else:
        model = GPT(model_config).to(device)
    model.load_state_dict(state["model"])
    set_qat_enabled(model, arguments.qat)
    model.eval()
    tokenizer_name = state.get("train_config", {}).get("tokenizer_name", "gpt2")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    input_ids = tokenizer.encode(arguments.prompt, return_tensors="pt").to(device)
    result = generate(
        model,
        input_ids,
        arguments.max_new_tokens,
        arguments.temperature,
        arguments.top_k,
        arguments.top_p,
        arguments.repetition_penalty,
        tokenizer.eos_token_id,
    )
    print(tokenizer.decode(result[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
