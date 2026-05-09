# Composed Llama 3 8B Block - Multi-Pattern Optimization

## Overview

This directory contains the composed model that combines two optimized patterns for the Llama 3 8B Single Transformer Block:
- **Pattern 1 (FMHA)**: Fused Multi-Head Attention with GQA support
- **Pattern 2 (SwiGLU MLP)**: Gated MLP with SiLU activation fusion

**Target GPU**: NVIDIA A100 (SM80, Ampere)
**Target Model**: `KernelBench/KernelBench/level3/51_Llama3_8B_Block.py`

## Expected Performance

| Pattern | Speedup |
|---------|---------|
| FMHA (attention only) | ~5.29x |
| SwiGLU MLP (MLP only) | ~8.42x |
| Combined (end-to-end) | ~2.98x |

## File Structure

```
agent_work/whole_model/llama3_8b/
├── cutlass_kernels/
│   ├── fmha/                     # FMHA include headers
│   ├── fmha_launch.cu            # FMHA kernel wrapper
│   └── swiglu_mlp_fusion.cu      # SwiGLU kernel wrapper
├── ModelNew.py                   # Composed model with pattern switches
├── test.py                       # Test & benchmark script
├── README.md                     # This file
└── test_scripts/
    ├── config.sh                 # Environment configuration
    ├── 1_check_env.sh            # Environment check
    ├── 2_compile.sh              # Compile extensions
    ├── 3_test_correctness.sh     # Correctness testing
    ├── 4_benchmark.sh            # Multi-mode benchmark
    └── run_all.sh                # Run all steps sequentially
```

## Pattern Configurations

### FMHA Pattern
- **Directory**: `pattern_table/fmha/fp16/sm80/fmha_llama3_gqa/`
- **Replaces**: Core attention computation (Q@K^T + softmax + attn@V)
- **Config** (hardcoded in `fmha_launch.cu`):
  - `queries_per_block`: 32
  - `keys_per_block`: 128
  - `aligned`: true

### SwiGLU MLP Pattern
- **Directory**: `pattern_table/gemm/fp16/sm80/swiglu_mlp_fusion/`
- **Replaces**: gate_proj + SiLU + up_proj + multiply + down_proj (entire MLP)
- **Config** (hardcoded in `swiglu_mlp_fusion.cu`):
  - `ThreadblockShape`: [128, 128, 32]
  - `WarpShape`: [64, 64, 32]
  - `pipeline_stages`: 3

## Pattern Switches

The composed model supports independent pattern enable/disable via constructor parameters:

```python
from ModelNew import Model

# Baseline (PyTorch only)
model_baseline = Model(enable_fmha=False, enable_swiglu=False)

# FMHA-only
model_fmha = Model(enable_fmha=True, enable_swiglu=False)

# SwiGLU-only
model_swiglu = Model(enable_fmha=False, enable_swiglu=True)

# Full optimization (both patterns)
model_full = Model(enable_fmha=True, enable_swiglu=True)
```

## Requirements

- **GPU**: NVIDIA A100 (SM80) or compatible
- **CUDA**: 12.4 or later
- **PyTorch**: with CUDA support
- **Compute Capability**: >= 8.0

## Running Tests on GPU

### Option 1: Run All Steps Automatically

```bash
cd examples/whole_model/llama3_8b/test_scripts
./run_all.sh
```

This will execute all validation steps in sequence with prompts before each step.

### Option 2: Run Steps Individually

```bash
cd examples/whole_model/llama3_8b/test_scripts

# Step 1: Check environment
./1_check_env.sh

# Step 2: Compile extensions
./2_compile.sh

# Step 3: Test correctness
./3_test_correctness.sh

# Step 4: Benchmark performance
./4_benchmark.sh
```

### Option 3: Run Python Test Script Directly

```bash
cd examples/whole_model/llama3_8b

# Run correctness tests only
python3 test.py --correctness-only --tolerance 1e-2

# Run all benchmark modes
python3 test.py --benchmark-mode all --warmup 5 --trials 20

# Run single benchmark mode
python3 test.py --benchmark-mode full --warmup 5 --trials 20
```

## GPU Node Instructions

### 1. Transfer Files to GPU Node

Copy the entire composed model directory and pattern directories:

```bash
# On local machine
cd <FACT_ROOT>
tar -czf llama3_composed.tar.gz agent_work/whole_model/llama3_8b agent_work/pattern_table

# Transfer to GPU node
scp llama3_composed.tar.gz user@gpu-node:/path/to/destination/

# On GPU node
tar -xzf llama3_composed.tar.gz
cd agent_work/whole_model/llama3_8b/test_scripts
```

### 2. Set Up Environment on GPU Node

The test scripts source **`test_scripts/config.sh`**, which sets paths for:
- `REPO_ROOT` (repo root directory)
- `AGENT_WORK` (agent_work directory)
- `WHOLE_MODEL` (composed model directory)
- `CUTLASS_ROOT` (CUTLASS headers)
- `PYTHONPATH` (for kernelbench_cutlass_module)

**Required directory structure on GPU node**:
```
/path/to/destination/
├── cutlass/                        # CUTLASS library
├── kernelbench_cutlass_module/     # JIT helper
├── KernelBench/                    # Baseline models
└── agent_work/
    ├── pattern_table/              # Pattern directories
    │   ├── fmha/fp16/sm80/fmha_llama3_gqa/
    │   └── gemm/fp16/sm80/swiglu_mlp_fusion/
    └── whole_model/llama3_8b/      # Composed model
```

Load modules and verify PyTorch:

```bash
# Load required modules (adjust based on your cluster)
module load cuda/12.4
module load python/3.11
module load pytorch/2.0.0

# Verify GPU availability
python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 3. Run Tests

```bash
# Run all tests
./run_all.sh

# Or run individual steps
./1_check_env.sh
./2_compile.sh
./3_test_correctness.sh
./4_benchmark.sh
```

### 4. Collect Results

Save output for documentation:

```bash
# Run tests and save output
./run_all.sh 2>&1 | tee results_$(date +%Y%m%d_%H%M%S).log
```

## Benchmark Modes

The test script supports 4 benchmark configurations:

| Mode | FMHA | SwiGLU | Description |
|------|------|--------|-------------|
| `baseline` | ❌ | ❌ | Pure PyTorch (reference) |
| `fmha_only` | ✅ | ❌ | Measure FMHA contribution |
| `swiglu_only` | ❌ | ✅ | Measure SwiGLU contribution |
| `full` | ✅ | ✅ | End-to-end combined speedup |
| `all` | — | — | Run all 4 modes automatically |

### Expected Output Format

```
=== PERFORMANCE BENCHMARK - MULTI-MODE ===

Configuration               Time (ms)    Speedup
------------------------------------------------------------
Baseline (PyTorch)          656.02       1.00x
FMHA-only                   585.26       1.12x
SwiGLU-only                 293.76       2.23x
Full (both)                 220.00       2.98x
```

## Correctness Testing

Test sizes for correctness verification:
- `(2, 128, 4096)` - Small
- `(4, 512, 4096)` - Medium
- `(8, 1024, 4096)` - Large
- `(16, 2048, 4096)` - Full problem size

**Tolerance**: `1e-2` (FP16 precision)

## Troubleshooting

### Extension Compilation Fails

1. Ensure CUDA toolkit is properly installed
2. Check compute capability matches your GPU (SM80 for A100)
3. Verify PyTorch CUDA version compatibility
4. Check CUTLASS library paths in `config.sh`

### Correctness Tests Fail

1. Verify both extensions are compiled successfully
2. Check tolerance setting (FP16 requires ~1e-2 tolerance)
3. Ensure pattern switches are correctly set
4. Check for numerical precision issues

### Poor Performance

1. Verify GPU is being used (check `CUDA_VISIBLE_DEVICES`)
2. Ensure both extensions are loaded (check pattern status)
3. Verify correct configurations are being used
4. Check no other processes are using GPU resources

## Notes

- **No modifications to pattern directories**: Extensions are loaded directly from pattern directories
- **Fallback to PyTorch**: If CUTLASS extensions fail to load, model automatically falls back to PyTorch
- **Lazy loading**: Extensions are compiled on-demand when ModelNew is first imported
- **Hardcoded configurations**: Default parameters from .cu files are used (no auto-tuning)

## Contact

For issues or questions about this composed model, refer to the individual pattern documentation:
- FMHA: `agent_work/pattern_table/fmha/fp16/sm80/fmha_llama3_gqa/`
- SwiGLU MLP: `agent_work/pattern_table/gemm/fp16/sm80/swiglu_mlp_fusion/`
