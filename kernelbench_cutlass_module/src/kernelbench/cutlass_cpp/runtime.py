"""Shared build settings for CUTLASS C++ extensions (JIT via ``torch.utils.cpp_extension``)."""

from __future__ import annotations

import os
from typing import List


def require_cutlass_root(explicit: str | None = None) -> str:
    root = explicit or os.environ.get("CUTLASS_ROOT", "").strip()
    if not root:
        raise RuntimeError(
            "Set CUTLASS_ROOT to the root of a CUTLASS source tree "
            "(directory containing the `include/cutlass/` headers)."
        )
    include = os.path.join(root, "include", "cutlass")
    if not os.path.isdir(include):
        raise RuntimeError(
            f"CUTLASS_ROOT={root!r} does not look like a CUTLASS tree "
            f"(missing {include})."
        )
    return root


def cutlass_include_paths(cutlass_root: str | None = None) -> List[str]:
    root = require_cutlass_root(cutlass_root)
    return [os.path.join(root, "include")]


def default_extra_cuda_cflags(
    *,
    gpu_arch: str | None = None,
    fast_math: bool = True,
) -> List[str]:
    """
    Extra nvcc flags for CUTLASS device code.

    ``gpu_arch`` examples: ``\"80\"`` for Ampere (sm_80). If unset, nvcc uses the default
    for the current driver GPU during JIT.
    """
    flags = ["-O3"]
    if fast_math:
        flags.append("--use_fast_math")
    if gpu_arch:
        g = gpu_arch.lstrip("sm_").lstrip("compute_")
        flags.append(f"-gencode=arch=compute_{g},code=sm_{g}")
    return flags
