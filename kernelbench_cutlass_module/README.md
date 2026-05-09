# KernelBench CUTLASS Setup - Square Matrix Multiplication (Module-Based)

This directory contains a minimal KernelBench-style setup for testing CUTLASS-accelerated GEMM kernels against PyTorch's baseline matmul implementation.

**This setup uses the cluster's `nvidia-cutlass/3.8.0.0` module** instead of local CUTLASS installation.

## Problem: Square Matrix Multiplication (Problem 1, Level 1)

Reference: `KernelBench/KernelBench/level1/1_Square_matrix_multiplication_.py`

## Directory Structure

```
kernelbench_cutlass_module/
├── src/
│   └── kernelbench/
│       ├── __init__.py
│       ├── cutlass_cpp/
│       │   ├── __init__.py
│       │   └── runtime.py          # CUTLASS build utilities
│       ├── eval.py                 # Evaluation helpers
│       ├── dataset.py              # Dataset utilities
│       ├── timing.py               # Timing utilities
│       └── kernel_static_checker.py
├── work/
│   ├── Model.py                    # Baseline PyTorch matmul
│   ├── ModelNew.py                 # CUTLASS GEMM implementation
│   ├── cutlass_kernels/            # CUDA/C++ source files
│   │   ├── ampere_tf32_gemm.cu     # CUTLASS kernel
│   │   └── main.cpp                # Python bindings
│   └── test.py                     # Test script
├── sbatch/
│   ├── step2_compile_cutlass.sh    # Compile CUTLASS extension
│   ├── step3_test_correctness.sh   # Test correctness
│   └── step4_benchmark.sh          # Benchmark performance
└── results/                        # Output files from slurm jobs
```

## Prerequisites

1. **CUDA/PyTorch**: Available on compute nodes with A100 GPUs
   - PyTorch with CUDA support (PyTorch 2.8.0+cu128 available)
   - CUDA toolkit module: `CUDA/12.6.0` (loaded by sbatch scripts)

2. **nvidia-cutlass module**: `nvidia-cutlass/3.8.0.0`
   - Provides CUTLASS headers and libraries
   - Required for compilation and NCU profiling
   - Loaded by all sbatch scripts

3. **Slurm**: Access to the cluster with account `pearl`

## Module Requirements

The sbatch scripts automatically load:
- `CUDA/12.6.0` - CUDA toolkit and nvcc compiler
- `nvidia-cutlass/3.8.0.0` - CUTLASS headers and NCU support

These modules provide:
- nvcc compiler (required for CUTLASS JIT compilation)
- CUTLASS headers (from module, not local path)
- CUDA toolkit compatible with CUTLASS and PyTorch
- NCU profiling support

## Usage: Step-by-Step Jobs

Each step requires submitting a slurm job. Run them sequentially:

### Step 1: Compile CUTLASS Extension

```bash
sbatch sbatch/step2_compile_cutlass.sh
```

This tests JIT compilation of the CUTLASS GEMM kernel via PyTorch's cpp_extension using the nvidia-cutlass module.

**Expected output:** Compilation messages from nvcc, followed by successful test.

### Step 2: Test Correctness

```bash
sbatch sbatch/step3_test_correctness.sh
```

This verifies that ModelNew (CUTLASS) produces the same results as Model (PyTorch) within FP32 tolerance (1e-4).

**Test sizes:** 512x512, 1024x1024, 2048x2048

### Step 3: Benchmark Performance

```bash
sbatch sbatch/step4_benchmark.sh
```

This benchmarks performance comparing PyTorch matmul vs CUTLASS GEMM.

**Test sizes:** 1024x1024, 2048x2048, 4096x4096 (problem size)

**Settings:** 5 warmup runs, 20 timed runs per size

## Output Files

Results are written to the `results/` directory:

- `step2_compile_<JOBID>.out` - Compilation output
- `step3_correctness_<JOBID>.out` - Correctness test results
- `step4_benchmark_<JOBID>.out` - Benchmark results

## Running Locally (for debugging)

You can also run tests directly on a GPU node (interactive session):

```bash
# Start interactive session
srun --account=pearl --partition=a100_normal_q --gres=gpu:a100:1 --time=01:00:00 --pty bash

# Load modules
module load CUDA/12.6.0
module load nvidia-cutlass/3.8.0.0

# Set environment
export PYTHONPATH=<FACT_ROOT>/kernelbench_cutlass_module/src:$PYTHONPATH

# Run tests
cd <FACT_ROOT>/kernelbench_cutlass_module
python work/test.py --help                    # Show options
python work/test.py --compile-only            # Test compilation
python work/test.py --correctness-only        # Test correctness
python work/test.py --benchmark-only          # Benchmark only
python work/test.py                           # Run all tests
```

## Implementation Details

### Model (Baseline - PyTorch)
- Uses PyTorch's `torch.matmul()` for square matrix multiplication
- Problem size: N = 2048 * 2 = 4096
- **TF32 tensor cores ENABLED** for optimal A100 performance
- Uses `torch.backends.cuda.matmul.allow_tf32 = True`

### ModelNew (CUTLASS TF32 Tensor Cores)
- Uses CUTLASS `device::Gemm` with **TF32 tensor cores** (OpClassTensorOp)
- Target: Ampere (SM80) architecture (A100 GPUs)
- Precision: FP32 input, TF32 computation, FP32 output
- Layout: Row-major for all matrices (A, B, C)
- **CUTLASS Source**: nvidia-cutlass/3.8.0.0 module (cluster-managed)
- **Separate CUDA files** (not inline source):
  - `cutlass_kernels/ampere_tf32_gemm.cu` - CUTLASS kernel implementation
  - `cutlass_kernels/main.cpp` - Python bindings
- Compiled via `torch.utils.cpp_extension.load()`
- **NCU Compatible**: Works with NCU profiling when module is loaded

### Correctness Tolerance
- FP32: 1e-4 (following KernelBench/PyTorch Benchmark standards)

## Troubleshooting

### Compilation fails
- Check that CUTLASS_ROOT is set correctly
- Verify CUDA/PyTorch versions are compatible
- Look for error messages in the compilation output

### Correctness test fails
- Check that both models use the same precision (fp32)
- Verify input tensors are on the same device (GPU)
- Check for numerical issues with extreme tensor values

### Performance issues
- Ensure GPU is properly utilized (check `nvidia-smi`)
- Verify tensor layouts match (row-major)
- Check that compilation used appropriate GPU architecture flags

## References

- KernelBench: `KernelBench/`
- CUTLASS: https://github.com/NVIDIA/cutlass
- Problem: `KernelBench/level1/1_Square_matrix_multiplication_.py`
