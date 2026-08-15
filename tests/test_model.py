import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from config import ModelConfig
from model import GPT
from quantization import set_qat_enabled


def tiny_config(**overrides):
    config = ModelConfig(
        vocab_size=101,
        max_seq_len=16,
        dim=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        ffn_hidden_dim=64,
        ffn_multiple_of=16,
        dropout=0.0,
        qk_norm=True,
        gradient_checkpointing=False,
        **overrides,
    )
    config.validate()
    return config


def test_gqa_forward_and_backward():
    model = GPT(tiny_config())
    tokens = torch.randint(0, 101, (2, 12))
    output = model(tokens)
    assert output.logits.shape == (2, 12, 101)
    output.logits.mean().backward()
    assert model.blocks[0].attention.q_proj.weight.grad is not None


def test_sdpa_backend_preserves_causal_forward_and_autograd():
    explicit = GPT(tiny_config(attention_backend="explicit")).eval()
    fused = GPT(tiny_config(attention_backend="sdpa")).eval()
    fused.load_state_dict(explicit.state_dict())
    tokens = torch.randint(0, 101, (2, 12))
    with torch.no_grad():
        expected = explicit(tokens).logits
        actual = fused(tokens).logits
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
    fused.train()
    fused(tokens).logits.mean().backward()
    assert fused.blocks[0].attention.q_proj.weight.grad is not None


def test_causal_mask_prevents_future_information():
    model = GPT(tiny_config()).eval()
    first = torch.randint(0, 101, (1, 10))
    second = first.clone()
    second[:, 6:] = torch.randint(0, 101, (1, 4))
    with torch.no_grad():
        first_logits = model(first).logits
        second_logits = model(second).logits
    torch.testing.assert_close(first_logits[:, :6], second_logits[:, :6])


def test_qat_path_has_finite_gradients():
    model = GPT(tiny_config(qat=True, qat_group_size=16))
    set_qat_enabled(model, True)
    output = model(torch.randint(0, 101, (2, 8)))
    output.logits.square().mean().backward()
    gradient = model.blocks[0].attention.q_proj.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()


def test_sparse_moe_returns_auxiliary_loss():
    model = GPT(
        tiny_config(moe_num_experts=4, moe_top_k=2, moe_every_n_layers=1)
    )
    output = model(torch.randint(0, 101, (2, 8)))
    assert output.auxiliary_loss.item() > 0
