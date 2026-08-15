"""Optional loader for the miniVLLM native attention extension.

The miniVLLM CUDA kernels use a paged KV-cache ABI for inference. They are not
used by the training model: their output buffers are written in-place and no
autograd backward is registered. Keeping the loader separate makes that
boundary explicit while allowing a local operator experiment when CUDA/NVCC
are available.
"""

from __future__ import annotations

import os
from pathlib import Path


def locate_sources(root: str | Path | None = None) -> tuple[Path, Path]:
    configured = root or os.environ.get("MINIVLLM_ROOT")
    if configured is None:
        raise FileNotFoundError(
            "Set MINIVLLM_ROOT to the miniVLLM project before loading its CUDA extension"
        )
    base = Path(configured)
    cpp = base / "runtime" / "torch_attention.cpp"
    cuda = base / "runtime" / "torch_attention_cuda.cu"
    if not cpp.is_file() or not cuda.is_file():
        raise FileNotFoundError(f"miniVLLM attention sources not found under {base}")
    return cpp, cuda


def load_minivllm_attention(root: str | Path | None = None) -> object:
    """Compile/load miniVLLM's inference-only paged attention extension on demand."""

    import torch
    from torch.utils.cpp_extension import load

    if not torch.cuda.is_available():
        raise RuntimeError("miniVLLM CUDA operators require a CUDA-enabled PyTorch")
    cpp, cuda = locate_sources(root)
    return load(
        name="minivllm_attention_bridge",
        sources=[str(cpp), str(cuda)],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=True,
    )
