#!/usr/bin/env bash
# =============================================================================
# Whole-model test pipeline: CUDA, CUTLASS, kernelbench, repo paths.
# Source this file from any script in this directory (bash only).
# Do not enable `set -e` here — parent scripts control error handling.
# =============================================================================

_THIS_CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WHOLE_MODEL="$(cd "$_THIS_CONFIG_DIR/.." && pwd)"
export FACT_ROOT="$(cd "$_THIS_CONFIG_DIR/../../.." && pwd)"
export REPO_ROOT="$FACT_ROOT"

# --- CUDA (optional defaults; nvcc on PATH wins if CUDA_HOME unset) ---
if [[ -z "${CUDA_HOME:-}" ]] && command -v nvcc &>/dev/null; then
  _nvcc_path="$(command -v nvcc)"
  export CUDA_HOME="$(cd "$(dirname "$_nvcc_path")/.." && pwd)"
fi
: "${CUDA_HOME:=/usr/local/cuda}"
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# --- CUTLASS headers (default: checkout next to agent_work) ---
if [[ -z "${CUTLASS_ROOT:-}" && -d "$REPO_ROOT/cutlass/include/cutlass" ]]; then
  export CUTLASS_ROOT="$REPO_ROOT/cutlass"
fi
if [[ -n "${CUTLASS_ROOT:-}" ]]; then
  export CUTLASS_ROOT
fi

# --- kernelbench (JIT helper package in this repo) ---
_KB_SRC="$REPO_ROOT/kernelbench_cutlass_module/src"
if [[ -d "$_KB_SRC" ]]; then
  export PYTHONPATH="$_KB_SRC${PYTHONPATH:+:$PYTHONPATH}"
fi

# --- Optional: GPU arch string for future tooling ---
: "${GPU_ARCH:=sm_80}"
export GPU_ARCH

# --- Verbose JIT (scripts may override) ---
: "${KERNELBENCH_CUTLASS_VERBOSE:=0}"
export KERNELBENCH_CUTLASS_VERBOSE
