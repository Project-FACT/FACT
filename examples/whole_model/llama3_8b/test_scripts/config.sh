#!/usr/bin/env bash
# =============================================================================
# Whole-model test pipeline: CUDA, CUTLASS, kernelbench, repo paths.
# =============================================================================

# Directory paths (derived from script location)
_THIS_CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WHOLE_MODEL="$(cd "$_THIS_CONFIG_DIR/.." && pwd)"
export FACT_ROOT="$(cd "$_THIS_CONFIG_DIR/../../.." && pwd)"
export REPO_ROOT="$FACT_ROOT"

# CUDA setup
if [[ -z "${CUDA_HOME:-}" ]] && command -v nvcc &>/dev/null; then
  _nvcc_path="$(command -v nvcc)"
  export CUDA_HOME="$(cd "$(dirname "$_nvcc_path")/.." && pwd)"
fi
: "${CUDA_HOME:=/usr/local/cuda}"
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# CUTLASS headers
if [[ -z "${CUTLASS_ROOT:-}" && -d "$REPO_ROOT/cutlass/include/cutlass" ]]; then
  export CUTLASS_ROOT="$REPO_ROOT/cutlass"
fi
if [[ -n "${CUTLASS_ROOT:-}" ]]; then
  export CUTLASS_ROOT
fi

# Python path
export PYTHONPATH="$REPO_ROOT/kernelbench_cutlass_module/src${PYTHONPATH:+:$PYTHONPATH}"

# GPU architecture
export GPU_ARCH="${GPU_ARCH:=sm_80}"

# Verbose JIT
export KERNELBENCH_CUTLASS_VERBOSE="${KERNELBENCH_CUTLASS_VERBOSE:=1}"
