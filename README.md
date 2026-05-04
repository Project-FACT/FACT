# FACT: Compositional Kernel Synthesis with a Three-Stage Agentic Workflow


FACT is a framework for automated GPU kernel synthesis using LLM agents and CUTLASS templates. It discovers optimization opportunities in PyTorch models, synthesizes CUTLASS kernels with auto-tuning, and composes them into end-to-end optimized models.

## Three-Stage Agentic Workflow

FACT operates through three stages, each guided by a dedicated prompt:

| Stage | Description | Output |
|-------|-------------|--------|
| **Stage 1: Pattern Discovery** | Analyzes PyTorch computation graphs and proposes optimization patterns. Classifies each pattern as REUSE, ADAPT, or NEW by querying the pattern registry. | Proposed patterns indexed by (r, τ, α) |
| **Stage 2: Pattern Realization** | Synthesizes CUTLASS kernels for each prioritized pattern. Addresses the three-level CUTLASS hierarchy (Tile, Kernel, Grid). Verifies correctness, benchmarks, and auto-tunes. | Validated kernels added to pattern registry |
| **Stage 3: Pattern Composition** | Composes realized patterns into an optimized whole model. Loads all extensions, validates end-to-end correctness, and reports speedup. | Deployable optimized PyTorch model |

Prompt files are available in [`prompts/`](prompts/):
- [`stage1_pattern_discovery.md`](prompts/stage1_pattern_discovery.md)
- [`stage2_pattern_realization.md`](prompts/stage2_pattern_realization.md)
- [`stage3_pattern_composition.md`](prompts/stage3_pattern_composition.md)

## Pattern Registry

The pattern registry is a dynamic catalog of reusable, architecture-specific CUTLASS kernel implementations indexed by the tuple **T(r, τ, α)**:

- **r** (Rule): Optimization rule — GEMM, FMHA, Batched GEMM, Epilogue Fusion
- **τ** (Data Type): FP16, FP32, TF32, BF16, FP8
- **α** (Architecture): SM80 (Ampere/A100), SM90 (Hopper/H100)

### Directory Structure

```
pattern_table/
└── <rule>/                  # e.g., gemm, fmha
    └── <data_type>/         # e.g., fp16, tf32, fp32
        └── <architecture>/  # e.g., sm80, sm90
            └── <pattern_name>/
                ├── Model.py            # Baseline PyTorch implementation
                ├── ModelNew.py         # CUTLASS-optimized implementation
                ├── cutlass_kernels/    # CUTLASS kernel sources (.cu, .h, main.cpp)
                └── pattern_info.json   # Pattern metadata and performance results
```


## Running Kernels

Patterns with `Model.py` and `ModelNew.py` can be validated and benchmarked using a unified test script structure. Each pattern follows the same 4-step workflow.

### Prerequisites

- CUDA 11.8+ with compatible GPU (A100 for SM80, H100 for SM90)
- PyTorch 2.0+
- CUTLASS 3.x — set `CUTLASS_ROOT` environment variable
- Python 3.8+

### Test Script Structure

Each pattern's test script (`test.py`) follows the same interface:

```bash
# Full validation (compile + correctness + benchmark)
python test.py

# Compile only
python test.py --compile-only

# Correctness only (requires compiled extension)
python test.py --correctness-only

# Benchmark only (requires compiled extension)
python test.py --benchmark-only
```

### How Compilation Works

`ModelNew.py` uses `torch.utils.cpp_extension.load()` to JIT-compile the CUTLASS kernel as a PyTorch extension:

```python
# In ModelNew.py — CUTLASS kernels are compiled on first use
fmha_ext = torch.utils.cpp_extension.load(
    name="fmha_ext",
    sources=[
        "cutlass_kernels/main.cpp",
        "cutlass_kernels/fmha_launch.cu",
    ],
    extra_include_paths=[cutlass_include],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_80"],
    verbose=True,
)
```

Set the `CUTLASS_ROOT` environment variable before running:

```bash
export CUTLASS_ROOT=/path/to/cutlass
```

### Validation Workflow

The test script runs four steps in sequence:

1. **Environment Check** — Verifies CUDA availability, GPU architecture, and `CUTLASS_ROOT`
2. **Compilation** — JIT-compiles the CUTLASS extension via `torch.utils.cpp_extension.load()`
3. **Correctness** — Compares `ModelNew` output against `Model` baseline across multiple input sizes with configurable tolerance (1e-2 for FP16, 1e-3 for FP32)
4. **Benchmark** — Measures latency and computes speedup, throughput (GFLOPS), and peak utilization percentage

### Pattern-Specific Notes

**FMHA Llama 3 (GQA):** The FMHA kernel receives standard MHA inputs. Grouped-Query Attention (32 Q heads, 8 KV heads) is handled by `repeat_interleave` on the KV tensors before the kernel call. The benchmark reports both full model speedup (entire LlamaDecoderLayer) and FMHA-only speedup (isolated attention computation).

**SwiGLU MLP Fusion:** Uses a dual-kernel approach — kernel 1 fuses gate_proj + up_proj with SiLU epilogue activation, kernel 2 performs down_proj. The benchmark reports both full model speedup and MLP-only speedup.

**Stream-K GEMM:** Uses `ThreadblockSwizzleStreamK` for load-balanced scheduling. Best suited for large K dimensions where M and N are relatively small. FP16 inputs with FP32 accumulator prevents overflow at large K values.

**FMHA (FP32):** Auto-tuning sweeps over tile configurations (queries_per_block, keys_per_block). Best config: Q64_K128_Aligned.

**Batched GEMM:** Auto-tuning found optimal tile shape 128x256x32 with 3 pipeline stages across 30 configurations.

## Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| CUTLASS | 3.x | GPU kernel templates |
| PyTorch | 2.0+ | Model framework, JIT compilation |
| CUDA | 11.8+ | GPU compilation and runtime |
| Python | 3.8+ | Scripting |

CUTLASS is referenced as a dependency (not included as a submodule). Install it separately:

```bash
git clone https://github.com/NVIDIA/cutlass.git --recursive
export CUTLASS_ROOT=/path/to/cutlass
```
