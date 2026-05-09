# FACT: Compositional Kernel Synthesis with a Three-Stage Agentic Workflow

FACT is a framework for automated GPU kernel synthesis using LLM agents and CUTLASS templates. It discovers optimization opportunities in PyTorch models, synthesizes CUTLASS kernels with auto-tuning, and composes them into end-to-end optimized models.

## Repository Structure

```
FACT/
├── prompts/                          # LLM prompts for the three-stage workflow
│   ├── stage1_pattern_discovery.md
│   ├── stage2_pattern_realization.md
│   └── stage3_pattern_composition.md
├── pattern_table/                    # Dynamic pattern registry (Stage 2 output)
│   ├── fmha/                         # Fused Multi-Head Attention patterns
│   │   ├── fp32/sm80/
│   │   └── fp16/sm80/
│   └── gemm/                         # GEMM patterns (square, batched, stream-K, fusion)
│       ├── tf32/sm80/
│       ├── fp16/sm80/
│       └── fp16/sm90/
├── kernelbench_cutlass_module/       # Evaluation template and utilities
│   ├── src/kernelbench/              # Core: eval, timing, CUTLASS build support
│   ├── work/                         # Working example (square GEMM)
│   └── sbatch/                       # SLURM submission scripts
└── examples/                         # Complete Stage 3 output examples
    └── whole_model/
        ├── miniGPT/                  # MiniGPT block: FMHA + MLP GEMM fusion
        └── llama3_8b/               # Llama 3 8B block: FMHA-GQA + SwiGLU fusion
```

## Three-Stage Agentic Workflow

FACT operates through three stages, each guided by a dedicated prompt:

| Stage | Description | Output |
|-------|-------------|--------|
| **Stage 1: Pattern Discovery** | Analyzes PyTorch computation graphs and proposes optimization patterns. Classifies each pattern as REUSE, ADAPT, or NEW by querying the pattern registry. | Proposed patterns indexed by (r, τ, α) |
| **Stage 2: Pattern Realization** | Synthesizes CUTLASS kernels for each prioritized pattern. Addresses the three-level CUTLASS hierarchy (Tile, Kernel, Grid). Verifies correctness, benchmarks, and auto-tunes. | Validated kernels added to pattern registry |
| **Stage 3: Pattern Composition** | Composes realized patterns into an optimized whole model. Loads all extensions, verifies end-to-end correctness, runs per-pattern ablation benchmarking, and reports speedup. | Deployable optimized PyTorch model |

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

### Available Patterns

| Pattern | Rule | Dtype | Arch | Description |
|---------|------|-------|------|-------------|
| `square_matrix_multiplication` | GEMM | TF32 | SM80 | Square GEMM with multistage pipeline |
| `batched_gemm` | GEMM | TF32 | SM80 | Batched matrix multiply |
| `mlp_gemm_fusion_gelu` | GEMM Fusion | TF32 | SM80 | Two-layer MLP with GELU epilogue fusion |
| `streamk_gemm_large_k` | GEMM | FP16 | SM80 | Stream-K scheduling for large K dimension |
| `fused_multi_head_attention` | FMHA | FP32 | SM80 | Fused multi-head attention (MiniGPT) |
| `fmha_llama3_gqa` | FMHA | FP16 | SM80 | FMHA with Grouped-Query Attention (Llama 3) |
| `swiglu_mlp_fusion` | GEMM Fusion | FP16 | SM80 | SwiGLU MLP with SiLU epilogue |
| `square_gemm_warp_specialized` | GEMM | FP16 | SM90 | Hopper warp-specialized GEMM |
| `grouped_gemm_tile_cluster` | GEMM | FP16 | SM90 | Hopper grouped GEMM with tile clusters |

## Example Compositions (Stage 3 Output)

Two complete composition examples are provided in [`examples/whole_model/`](examples/whole_model/):

### MiniGPT Block

FMHA + MLP GEMM Fusion with GELU, targeting the `44_MiniGPTBlock` problem.

```
examples/whole_model/miniGPT/
├── ModelNew.py              # Full composition (FMHA + MLP)
├── ModelNew_fmha_only.py    # Ablation: FMHA only
├── ModelNew_mlp_only.py     # Ablation: MLP only
├── ablation_benchmark.py    # Ablation runner
├── test.py                  # Validation script
└── test_scripts/            # Four-step validation pipeline
```

**Approach:** Separate-file ablation (Approach A in Stage 3 prompt).

### Llama 3 8B Block

FMHA-GQA + SwiGLU MLP Fusion, targeting the `51_Llama3_8B_Block` problem.

```
examples/whole_model/llama3_8b/
├── ModelNew.py              # Composition with enable_fmha/enable_swiglu switches
├── test.py                  # Multi-mode validation + ablation
└── test_scripts/            # Four-step validation pipeline
```

**Approach:** Constructor-switch ablation (Approach B in Stage 3 prompt).

## Evaluation Template

[`kernelbench_cutlass_module/`](kernelbench_cutlass_module/) provides the shared infrastructure used across all stages:

- **`src/kernelbench/`** — evaluation utilities (correctness checking, timing, CUTLASS build support)
- **`work/`** — a working example (square GEMM) demonstrating the expected file structure
- **`sbatch/`** — SLURM scripts for cluster-based validation

## Running Kernels

### Prerequisites

- CUDA 11.8+ with compatible GPU (A100 for SM80, H100 for SM90)
- PyTorch 2.0+
- CUTLASS 3.x — set `CUTLASS_ROOT` environment variable
- Python 3.8+

```bash
git clone https://github.com/NVIDIA/cutlass.git --recursive
export CUTLASS_ROOT=/path/to/cutlass
```

### Test Script Structure

Each pattern's `test.py` follows the same interface:

```bash
python test.py                    # Full validation (compile + correctness + benchmark)
python test.py --compile-only     # Compile only
python test.py --correctness-only # Correctness only
python test.py --benchmark-only   # Benchmark only
```

### Validation Workflow

The four-step pipeline (used by both individual patterns and composed models):

1. **Environment Check** — Verifies CUDA, GPU architecture, and `CUTLASS_ROOT`
2. **Compilation** — JIT-compiles the CUTLASS extension via `torch.utils.cpp_extension.load()`
3. **Correctness** — Compares output against PyTorch baseline across multiple input sizes
4. **Benchmark** — Measures latency, throughput (GFLOPS), and speedup

### How Compilation Works

`ModelNew.py` uses `torch.utils.cpp_extension.load()` for JIT compilation:

```python
ext = torch.utils.cpp_extension.load(
    name="cutlass_ext",
    sources=["cutlass_kernels/main.cpp", "cutlass_kernels/kernel.cu"],
    extra_include_paths=[cutlass_include],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_80"],
    verbose=True,
)
```

## Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| CUTLASS | 3.x | GPU kernel templates |
| PyTorch | 2.0+ | Model framework, JIT compilation |
| CUDA | 11.8+ | GPU compilation and runtime |
| Python | 3.8+ | Scripting |

CUTLASS is referenced as a dependency (not included). Install separately:
```bash
git clone https://github.com/NVIDIA/cutlass.git --recursive
export CUTLASS_ROOT=/path/to/cutlass
```
