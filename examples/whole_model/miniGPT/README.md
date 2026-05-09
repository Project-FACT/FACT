# Composed MiniGPTBlock Model - Pattern Composition

## Overview

This directory contains the composed model that combines two optimized patterns for the MiniGPTBlock:
- **p1: Fused Multi-Head Attention (FMHA)** - 3.618x speedup
- **p2: MLP GEMM Fusion with GELU** - 2.852x speedup

**Expected overall speedup**: ~3.47x

## File Structure

```
agent_work/whole_model/
├── ModelNew.py              # Composed optimized model
├── test.py                  # Test script for correctness and benchmarking
└── test_scripts/
    ├── config.sh            # CUDA / CUTLASS / PYTHONPATH (sourced by all steps)
    ├── 1_check_env.sh       # Step 1: Environment check
    ├── 2_compile.sh         # Step 2: Compile extensions
    ├── 3_test_correctness.sh # Step 3: Correctness testing
    ├── 4_benchmark.sh       # Step 4: Performance benchmarking
    └── run_all.sh           # Run all steps sequentially
```

## Pattern Configurations

### p1 (FMHA)
- **queries_per_block**: 64
- **keys_per_block**: 128
- **aligned**: true
- **Autotune results**: `pattern_table/fmha/fp32/sm80/fused_multi_head_attention/autotune_results.json`

### p2 (MLP GEMM Fusion)
- **tile**: [128, 256, 32]
- **warp**: [64, 64, 32]
- **stages**: 4
- **Autotune results**: `pattern_table/gemm/tf32/sm80/mlp_gemm_fusion_gelu/autotune_results.json`

## Requirements

- **GPU**: NVIDIA A100 (SM80) or compatible
- **CUDA**: 12.4 or later
- **PyTorch**: with CUDA support
- **Compute Capability**: >= 8.0

## Running Tests on GPU

### Option 1: Run All Steps Automatically

```bash
cd examples/whole_model/miniGPT/test_scripts
./run_all.sh
```

This will execute all validation steps in sequence with prompts before each step.

### Option 2: Run Steps Individually

```bash
cd examples/whole_model/miniGPT/test_scripts

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
cd examples/whole_model/miniGPT

# Run correctness tests only
python3 test.py --correctness-only --tolerance 1e-3

# Run benchmark only
python3 test.py --benchmark-only --warmup 5 --trials 20

# Run both correctness and benchmark
python3 test.py --warmup 5 --trials 20
```

## GPU Node Instructions

### 1. Transfer Files to GPU Node

Copy the entire `agent_work/whole_model` directory and the pattern directories to the GPU node:

```bash
# On local machine
cd <FACT_ROOT>
tar -czf whole_model.tar.gz agent_work/whole_model agent_work/pattern_table

# Transfer to GPU node
scp whole_model.tar.gz user@gpu-node:/path/to/destination/

# On GPU node
tar -xzf whole_model.tar.gz
cd agent_work/whole_model/test_scripts
```

### 2. Set Up Environment on GPU Node

The test scripts source **`test_scripts/config.sh`**, which sets **`REPO_ROOT`**, **`PYTHONPATH`** (for `kernelbench_cutlass_module/src`), and **`CUTLASS_ROOT`** (default: `$REPO_ROOT/cutlass` when that tree exists). You still need the full repo checkout next to `agent_work` (same layout as on your laptop): `cutlass/`, `kernelbench_cutlass_module/`, `KernelBench/`, and `agent_work/`.

Override defaults when needed, for example:

```bash
export CUTLASS_ROOT=/path/to/cutlass
```

Then load modules and verify PyTorch:

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

After running tests, the output will show:
- Pattern availability status
- Configurations used
- Correctness test results (PASS/FAIL)
- Performance metrics (baseline time, optimized time, speedup)

Save the output for documentation:

```bash
# Run tests and save output
./run_all.sh 2>&1 | tee results_$(date +%Y%m%d_%H%M%S).log
```

## Expected Results

### Correctness
- All test sizes should PASS with tolerance ≤ 1e-3
- Test sizes: (2,64,768), (4,128,768), (16,256,768), (128,512,768)

### Performance
- **FMHA speedup**: ~3.62x (71.31ms → 19.71ms)
- **MLP speedup**: ~2.85x (13.20ms → 4.63ms)
- **Overall speedup**: ~3.47x (84.51ms → 24.34ms)

## Troubleshooting

### Extension Compilation Fails

If extensions fail to compile:
1. Ensure CUDA toolkit is properly installed
2. Check that compute capability matches your GPU
3. Verify PyTorch CUDA version compatibility
4. Check that CUTLASS library paths are correct

### Correctness Tests Fail

If correctness tests fail:
1. Verify autotune configurations match the problem size
2. Check tolerance setting (FP16 precision requires ~1e-3 tolerance)
3. Ensure both extensions are compiled successfully
4. Check for numerical precision issues

### Poor Performance

If performance is worse than expected:
1. Verify GPU is actually being used (check CUDA_VISIBLE_DEVICES)
2. Ensure both extensions are loaded (check pattern status)
3. Check that correct autotune configurations are used
4. Verify no other processes are using GPU resources

## Notes

- **No modifications to p1 and p2 directories**: This composed model loads extensions directly from the pattern directories without modifying them
- **Fallback to PyTorch**: If CUTLASS extensions fail to load, the model automatically falls back to PyTorch implementations
- **Lazy loading**: Extensions are compiled on-demand when ModelNew is first imported
- **Configuration hardcoding**: Best configurations from autotuning are hardcoded in ModelNew.py

## Contact

For issues or questions about this composed model, refer to the individual pattern documentation:
- FMHA: `pattern_table/fmha/fp32/sm80/fused_multi_head_attention/`
- MLP GEMM Fusion: `pattern_table/gemm/tf32/sm80/mlp_gemm_fusion_gelu/`