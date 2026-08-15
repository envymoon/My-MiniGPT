"""Small CUDA Graph helper for static-shape inference experiments.

CUDA Graphs replay a previously captured launch sequence. They require fixed
tensor addresses and shapes, so this helper deliberately targets inference
(``no_grad``) rather than the variable-length, checkpointed training loop.
Training still gets native autograd through the selected PyTorch attention
backend.
"""

from __future__ import annotations

from collections.abc import Callable

import torch


class CUDAGraphForward:
    """Capture and replay a tensor-only forward function at one fixed shape."""

    def __init__(
        self,
        forward: Callable[[torch.Tensor], torch.Tensor],
        sample_input: torch.Tensor,
        warmup: int = 3,
    ) -> None:
        if not torch.cuda.is_available() or sample_input.device.type != "cuda":
            raise RuntimeError("CUDAGraphForward requires a CUDA tensor")
        if sample_input.requires_grad:
            raise ValueError("CUDA graph helper is inference-only; disable gradients")
        if warmup < 1:
            raise ValueError("warmup must be positive")
        self.forward = forward
        self.static_input = torch.empty_like(sample_input)
        self.static_input.copy_(sample_input)
        self.graph = torch.cuda.CUDAGraph()

        warmup_stream = torch.cuda.Stream()
        current_stream = torch.cuda.current_stream()
        warmup_stream.wait_stream(current_stream)
        with torch.cuda.stream(warmup_stream), torch.no_grad():
            for _ in range(warmup):
                self.forward(self.static_input)
        current_stream.wait_stream(warmup_stream)

        with torch.no_grad():
            self.static_input.copy_(sample_input)
            self.graph.capture_begin()
            self.static_output = self.forward(self.static_input)
            self.graph.capture_end()

    def __call__(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.shape != self.static_input.shape:
            raise ValueError(
                "CUDA Graph replay requires the captured input shape "
                f"{tuple(self.static_input.shape)}, got {tuple(input_ids.shape)}"
            )
        if input_ids.device != self.static_input.device:
            raise ValueError("CUDA Graph replay requires the captured CUDA device")
        with torch.no_grad():
            self.static_input.copy_(input_ids)
            self.graph.replay()
        return self.static_output


def capture_gpt_logits(model: torch.nn.Module, sample_input: torch.Tensor) -> CUDAGraphForward:
    """Capture ``model(input_ids).logits`` for a fixed inference shape."""

    model.eval()
    return CUDAGraphForward(lambda ids: model(ids).logits, sample_input)
