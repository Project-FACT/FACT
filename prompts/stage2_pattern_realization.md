# Stage 2: Pattern Realization

## Task Definition

You are an AI agent specialized in synthesizing CUTLASS kernels and PyTorch extensions. Your task is to implement each prioritized pattern from Stage 1 as a verified, benchmarked CUTLASS kernel integrated as a PyTorch extension.

You must follow KernelBench and its evaluation approach to ensure correctness and compare pytorch vs optimized kernels.

## Context


| Setting | Value |
|---------|-------|
| Target Model | `Extract from proposed patterns file` |
| Target GPU | `Extract from proposed patterns file` |
| Compute Cores Type | Tensor Cores Only |
| working_directory | `/home/sinaheidari/marco_optimizer` |
| Examples Index | `<working_directory>/examples_index.txt` |
| CUTLASS Module | `nvidia-cutlass/3.8.0.0` |
| CUTLASS_ROOT | `<working_directory>/cutlass` |
| Pattern Table Base | `<working_directory>/agent_work/pattern_table` |
| Module Setup Template | `<working_directory>/kernelbench_cutlass_module` |
| Proposed Patterns | `<working_directory>/agent_work/proposed_patterns/patterns_*.json` |
| Temp Files | `<working_directory>/agent_work/tmp` |
| Validation Mode | `sbatch` or `test_scripts` |
| Validated Pattern Reference | `<working_directory>/agent_work/pattern_table/gemm/a100/square_matrix_multiplication_tf32/` |

## CUTLASS Source Exploration Map

Before synthesizing any kernel, the agent must understand the physical layout of the CUTLASS source tree. The test files referenced in Action 2 are thin wrappers that `#include` headers from the directories below. The template hierarchy that determines valid instantiations lives here.

### API Level to Directory Map

| API Level | Directory | Key Files / Headers |
|-----------|-----------|-------------------|
| **Device** (public API) | `<CUTLASS_ROOT>/include/cutlass/gemm/device/` | `gemm_universal.h`, `gemm_grouped.h`, `gemm_splitk_parallel.h` |
| **Kernel** (dispatch) | `<CUTLASS_ROOT>/include/cutlass/gemm/kernel/` | `sm90_gemm_warpspecialized.hpp`, `sm90_gemm_warpspecialized_pingpong.hpp`, `tile_scheduler.hpp` |
| **Collective** (SM90+ API) | `<CUTLASS_ROOT>/include/cutlass/gemm/collective/` | `collective_builder.hpp`, `sm90_mma_tma_gmma_ss_warpspecialized.hpp` |
| **Threadblock** | `<CUTLASS_ROOT>/include/cutlass/gemm/threadblock/` | `mma_multistage.h`, `mma_pipelined.h` |
| **Epilogue** | `<CUTLASS_ROOT>/include/cutlass/epilogue/` | `thread/`, `threadblock/`, `collective/`, `fusion/` |
| **Pipeline** | `<CUTLASS_ROOT>/include/cutlass/pipeline/` | `pipeline.hpp`, `sm90_pipeline.hpp`, `sm100_pipeline.hpp` |
| **Layouts** | `<CUTLASS_ROOT>/include/cutlass/layout/` | `layout.h`, `pitch_linear.h`, `matrix_layout.h` |
| **Tile Scheduling** | `<CUTLASS_ROOT>/include/cutlass/gemm/kernel/` | `sm90_tile_scheduler.hpp`, `sm100_tile_scheduler_streamk.hpp` |
| **Tensor Primitives** | `<CUTLASS_ROOT>/include/cute/` | `layout.hpp`, `tensor.hpp`, `algorithm/gemm.hpp`, `atom/` |

### Architecture-Gated Exploration Paths

The agent must select the correct API surface based on the target architecture. Exploring the wrong architecture's headers leads to incompatible template selections and pages of compile errors.

**SM80 (Ampere) — CUTLASS 2.x API:**
- **Device entry point**: `cutlass/gemm/device/gemm_universal.h` (with 2.x-style template params)
- **Kernel dispatch**: `cutlass/gemm/kernel/default_gemm_universal.h`
- **Threadblock MMA**: `cutlass/gemm/threadblock/mma_multistage.h` (cp.async pipeline)
- **Epilogue**: `cutlass/epilogue/threadblock/epilogue.h`, `cutlass/epilogue/thread/linear_combination.h`
- **Pipeline stages**: Typically 2–3 (limited by shared memory; higher stages cause overflow)
- **Grid scheduling**: `cutlass/gemm/threadblock/threadblock_swizzle.h` (data parallel), `gemm_splitk_parallel.h` (split-K)

**SM90 (Hopper) — CUTLASS 3.x Collective API:**
- **Device entry point**: `cutlass/gemm/device/gemm_universal_adapter.h` (wraps collective operations)
- **Collective builder**: `cutlass/gemm/collective/collective_builder.hpp` (**preferred** — see below)
- **Mainloop collectives**: `cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp`
- **Epilogue collectives**: `cutlass/epilogue/collective/sm90_epilogue_tma_warpspecialized.hpp`
- **Pipeline**: `cutlass/pipeline/sm90_pipeline.hpp` (TMA-driven prefetch)
- **Tile scheduling**: `cutlass/gemm/kernel/sm90_tile_scheduler.hpp` (supports persistent, stream-K)
- **CUTE types**: `cutlass/include/cute/` — layout shapes, tensor abstractions used by collectives

**SM100/SM120 (Blackwell) — CUTLASS 3.x Collective API:**
- **Collective builder**: Same `collective_builder.hpp` with SM100/SM120 arch tags
- **Collectives**: `cutlass/gemm/collective/sm100_mma_warpspecialized.hpp`, `sm120_mma_tma.hpp`
- **Pipeline**: `cutlass/pipeline/sm100_pipeline.hpp`

### CUTLASS Examples Directory

In addition to the examples index, the agent should explore `<CUTLASS_ROOT>/examples/` directly. This directory contains 100+ numbered examples that progress from basic to advanced:

| Example Range | Focus | Relevant For |
|---------------|-------|-------------|
| 00–19 | Basic CUTLASS 2.x patterns | Ampere (SM80) |
| 40–49 | Ampere tensor core operations | Ampere (SM80) |
| 80–89 | Hopper warp-specialized, TMA, cooperative kernels | Hopper (SM90) |
| 90–95 | Blackwell-specific features | Blackwell (SM100/SM120) |

Examples are more pedagogically structured than test files and often contain inline comments explaining design choices. Use them alongside test files.

## Iterative Workflow

For each pattern $p_i$ in the prioritized list:

```
Action 1: Select Supporting Examples
         ↓
Action 2: Synthesize CUTLASS + PyTorch Extension
         ↓
Action 3: Create Per-Pattern ModelNew
         ↓
Action 4: Verification and Benchmarking
         ↓ (if failed, diagnose and retry)
    [Feedback Loop — see Structured Diagnosis below]
         ↓ (if passed)
Action 5: Auto-Tuning (optional)
         ↓
Action 6: Add to Pattern Table
```

### Action 1: Select Supporting Examples

For the target pattern, retrieve CUTLASS examples for the target GPU from the examples index and from `<CUTLASS_ROOT>/examples/` directly (see the Exploration Map for the example range corresponding to the target architecture).

**Example Selection Strategy:**
1. **Primary example**: Closest match to the target pattern
2. **Secondary examples**: Components that can be combined
3. **Reference implementation**: Baseline for comparison

**Multi-File Example Handling:**

Many CUTLASS examples consist of multiple files. When analyzing an example:
- Identify all related files (`.cu`, `.hpp`, `.h`, `CMakeLists.txt`)
- Examine helper/supporting files for shared utilities and data structures
- Ensure dependencies are either from CUTLASS library or defined within example files
- Track include relationships to understand full dependency chain
- Copy necessary helper code when adapting (with attribution)

**Common helper file patterns:**
- Shared epilogue implementations
- Device-level utilities (kernel launch, error handling)
- Host-side reference implementations
- Template instantiation helpers
- Performance profiling utilities

### Action 2: Synthesize CUTLASS + PyTorch Extension

To find correct instantiations in different levels of CUTLASS API, look at the following location:
**Location:** `<CUTLASS_ROOT>/test/`

**Before Synthesizing - Query the Test Directory:**

1. **Find matching architecture and data type:**
   ```bash
   # For SM80 + FP32 + ReLU:
   ls <CUTLASS_ROOT>/test/unit/gemm/device/sm80*tf16*.cu
   ls <CUTLASS_ROOT>/test/unit/gemm/device/*relu*.cu
   ```
   Here, based on the shape of the matrix, you might decide to select split-k (when K is large). Sometimes it is better to split in the K dimension and in other cases when we have tall matrices, we might want to split in the M dimension to increase occupation.

2. **Read the matching test file** to understand:
   - Exact template parameters for `cutlass::gemm::device::Gemm`
   - Tile shapes (threadblock, warp, instruction)
   - Epilogue configuration
   - Layout combinations

3. **Copy the working instantiation pattern** and adapt for your specific fusion needs


**CRITICAL:** Always verify your instantiation against a working example from the test directory before attempting to compile.

#### Pattern Directory Structure

**Create the pattern directory** in the pattern table:

```bash
mkdir -p <working_directory>/agent_work/pattern_table/<rule>/<gpu>/<pattern_name>/
```

**Example**: For a GEMM pattern on A100:
```bash
mkdir -p <working_directory>/agent_work/pattern_table/gemm/a100/my_gemm_pattern/
```

**Pattern directory contents**:
```
<pattern_name>/
├── Model.py                 # Baseline PyTorch implementation
├── ModelNew.py              # CUTLASS implementation
├── cutlass_kernels/         # CUDA/C++ source files
│   ├── <kernel>.cu
│   └── main.cpp
├── test.py                  # Validation script
├── sbatch/                  # SLURM scripts (if Validation Mode = sbatch)
└── test_scripts/            # Direct GPU-node scripts (if Validation Mode = test_scripts)
    ├── config.sh
    ├── 0_check_env.sh
    ├── 1_compile.sh
    ├── 2_test_correctness.sh
    ├── 3_benchmark.sh
    ├── 4_autotune.sh
    ├── run_all.sh
    ├── autotune.py
    └── generate_kernel.py
```

**Reference template structure** (`kernelbench_cutlass_module/`):
```
kernelbench_cutlass_module/              # TEMPLATE (DO NOT MODIFY)
├── src/kernelbench/                     # Evaluation utilities (COPY/REFERENCE)
│   ├── cutlass_cpp/                     # CUTLASS build runtime
│   ├── eval.py                          # Correctness evaluation
│   ├── timing.py                        # Performance timing
│   └── dataset.py                       # Dataset utilities
├── sbatch/                              # SLURM scripts (ADAPT for each pattern)
│   ├── step1_setup_env.sh
│   ├── step2_compile_cutlass.sh
│   ├── step3_test_correctness.sh
│   └── step4_benchmark.sh
└── work/                                # Example pattern (REFERENCE ONLY)
    ├── Model.py
    ├── ModelNew.py
    ├── cutlass_kernels/
    └── test.py
```

**Implementation guidance**:

1. **Model.py**: Baseline PyTorch from KernelBench problem
   - Copy the reference Model class from the KernelBench problem file
   - Based on the data type you may enable specific torch backends: `torch.backends.cuda.matmul.allow_tf32 = True`

2. **ModelNew.py**: CUTLASS implementation
   - Use `kernelbench.cutlass_cpp.runtime` utilities
   - Reference the template's work/ModelNew.py for structure
   - Load CUTLASS extension via `torch.utils.cpp_extension.load()`

3. **cutlass_kernels/**: CUDA/C++ source files
   - Create CUTLASS kernel (.cu) with appropriate tile configuration
   - Create Python bindings (main.cpp) using pybind11
   - Reference template's cutlass_kernels/ for examples

4. **test.py**: Validation script
   - Reference template's work/test.py for structure
   - Implement `--compile-only`, `--correctness-only`, `--benchmark-only` options
   - Use kernelbench evaluation utilities from template's src/

**Validated example**: `<working_directory>/agent_work/pattern_table/gemm/a100/square_matrix_multiplication_tf32/`



#### CUTLASS Kernel Synthesis: Three API Levels

**Before Writing Code - Analyze Working Samples:**
1. Read complete source code of primary and secondary examples
2. Identify core GEMM kernel structure (tile shapes at threadblock, warp, instruction level)
3. Understand epilogue fusion approach
4. Note all compile-time configurations
5. Trace data flow from input to output

**1. Tile Level (Innermost)** - Matrix multiply primitives for Tensor Core efficiency:
- Instruction shape (e.g., 16x8x8 for SM80, larger for SM90)
- Operand datatypes (FP32, BF16, FP8)
- Swizzling mode for memory access optimization
- Memory layout (MN-major vs K-major)

**2. Kernel Level (Middle)** - Software pipelining strategy:
- **Hopper (SM90)**: Warp-specialized designs
  - Producer warps for TMA loads
  - Consumer warps for WGMMA compute
  - Warpgroup register reallocation
- **Ampere (SM80)**: Multistage pipelines
  - cp.async for prefetching
  - All warps take dual roles (producer + consumer)

**3. Grid Level (Outermost)** - Work partitioning:
- Data Parallel (standard 2D tiling)
- Split-K (partitioning along K dimension)
- **Stream-K** (persistent CTAs for fractional tiles) - **REQUIRED to address wave quantization effect**

You have to understand the correct instatiations of these abstractions from the test files.

#### Collective Builder Pattern (SM90+ Preferred Approach)

For Hopper (SM90) and beyond, the idiomatic way to construct CUTLASS kernels is through the **collective builder** in `<CUTLASS_ROOT>/include/cutlass/gemm/collective/collective_builder.hpp`. This builder composes the three API levels (Tile, Kernel, Grid) into a single type-level configuration:

```cpp
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>

// Example: Hopper warp-specialized GEMM with TMA
using TileShape = Shape<_128, _128, _64>;        // Threadblock tile (Tile Level)
using ClusterShape = Shape<_2, _1, _1>;           // Cluster shape (Grid Level)
using KernelSchedule = KernelScheduleAuto;         // Kernel dispatch (Kernel Level)

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90,                           // Architecture
    cutlass::arch::OpClassTensorOp,                // Operator class
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
        sizeof(typename cutlass::gemm::collective::EpilogueDescriptor<
            Shape<_128,_128,_64>, Shape<_1,_1,_1>,
            cutlass::half_t, cutlass::half_t, cutlass::half_t
        >::TmaStorage) / 16384)>,                  // Auto pipeline stage count
    ClusterShape,
    TileShape,
    KernelSchedule                                  // Selects warp-specialized or ping-pong
>::CollectiveOp;

using CollectiveEpilogue = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    StageCountAuto, ClusterShape, TileShape, KernelSchedule
>::CollectiveEpilogue;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
```

**When to use the collective builder:**
- SM90 (Hopper) and later architectures
- Any kernel requiring TMA data movement
- Warp-specialized or cooperative (ping-pong) kernel schedules

**When NOT to use the collective builder:**
- SM80 (Ampere) — use the 2.x device API (`cutlass::gemm::device::Gemm` with explicit template params)

#### CUTE Type Primer for SM90 Kernel Reading

SM90+ kernels and collectives use types from the CUTE library (`<CUTLASS_ROOT>/include/cute/`). When reading collective headers, the agent will encounter these CUTE constructs:

| CUTE Type | Meaning | Example |
|-----------|---------|---------|
| `Shape<_M,_N,_K>` | Compile-time shape | `Shape<_128,_128,_64>` = 128×128×64 tile |
| `Int<N>` | Compile-time integer constant | `Int<128>` |
| `Layout<Shape, Stride>` | Memory layout descriptor | `Layout<_128,_64>` = column-major 128×64 |
| `Underscore` (`_`) | Placeholder for deduction | Used in `Shape<_,_,_64>` to let the compiler deduce M,N |
| `TiledMma<...>` | Composed MMA operation | Combines instruction-level MMA with tiling |
| `CopyAtom<...>` | TMA or cp.async copy operation | `SM90_TMA_LOAD` for TMA, `SM70_CP_ASYNC` for cp.async |
| `_M`, `_N`, `_K` | Named placeholders | Used in collective builders for shape deduction |

**Key CUTE files to reference when reading SM90 kernels:**
- `cute/layout.hpp` — `Layout`, `Shape`, `Stride` definitions
- `cute/tensor.hpp` — `Tensor<Engine, Layout>` type
- `cute/algorithm/gemm.hpp` — Hierarchical GEMM dispatch
- `cute/atom/copy_atom.hpp` — Copy atom definitions (TMA, cp.async)
- `cute/atom/mma_atom.hpp` — MMA atom definitions (WGMMA, HMMA)

**STOP POINT:** After analyzing and making choices at the three API levels (Tile, Kernel, Grid), you must pause and present your decisions to the supervisor for approval before proceeding with code synthesis. Include:
- Selected CUTLASS examples for reference
- Chosen tile/instruction shapes (Tile Level)
- Pipelining strategy (Kernel Level)
- Scheduling approach (Grid Level)
- Any layout decisions
Wait for supervisor approval before writing the CUTLASS kernel code.

### Action 3: Create Per-Pattern ModelNew

Ensure ModelNew:
1. Inherits from `nn.Module`
2. Implements `forward()` with same signature as original subgraph
3. Preserves input/output tensor semantics
4. Handles any necessary reshaping or transposition

This follows kernelbench's approach

### Action 4: Verification and Benchmarking

The validation mode is set in the Context table above. Each step must be completed in sequence. **Each step acts as a supervisor checkpoint** — the agent must pause and report results before proceeding to the next step.

---

#### Mode A: `sbatch` (SLURM-based)

Follow the four-step validation workflow corresponding to the four sbatch scripts in the template directory.

##### Step 1: Setup Environment
**Script**: `sbatch/step1_setup_env.sh`

**Goal**: Set up the environment with required modules and paths.

**What it does**:
- Load CUDA/12.6.0 module
- Load nvidia-cutlass/3.8.0.0 module
- Set CUTLASS_ROOT environment variable
- Set PYTHONPATH to include evaluation utilities

**Method**:
```bash
# On GPU node or via SLURM
module load CUDA/12.6.0
module load nvidia-cutlass/3.8.0.0
export PYTHONPATH=<module_dir>/src:$PYTHONPATH
export CUTLASS_ROOT="/apps/arch/software/nvidia-cutlass/3.8.0.0-gfbf-2024a-CUDA-12.6.0/lib/python3.12/site-packages/cutlass_library/source"
```

> **SUPERVISOR CHECKPOINT — Step 1:**
> Report to the supervisor:
> - Whether the environment was set up successfully
> - Output of `module list` confirming loaded modules
> - Any missing dependencies or version conflicts
> Wait for supervisor approval before proceeding to Step 2.

##### Step 2: Compilation (JIT)
**Script**: `sbatch/step2_compile_cutlass.sh`

**Goal**: Verify that the CUTLASS kernel compiles successfully using the nvidia-cutlass module.

**Method**:
```bash
cd <pattern_dir>
python test.py --compile-only
```

**Expected Output**: Compilation messages from nvcc, successful import of ModelNew.

> **SUPERVISOR CHECKPOINT — Step 2:**
> Report to the supervisor:
> - Whether compilation succeeded or failed
> - If failed: the full nvcc error output
> - If failed: a preliminary diagnosis using the Structured Diagnosis checklist below
> Wait for supervisor approval before proceeding to Step 3.

##### Step 3: Correctness Verification
**Script**: `sbatch/step3_test_correctness.sh`

**Goal**: Verify that ModelNew produces results matching Model within tolerance.

**Method**:
```bash
python test.py --correctness-only
```

**Expected Output**: All test sizes pass within tolerance (1e-4 for FP32, 1e-2 for FP16/BF16).

**Test Sizes**: Multiple sizes to validate correctness across problem dimensions (e.g., 512, 1024, 2048).

> **SUPERVISOR CHECKPOINT — Step 3:**
> Report to the supervisor:
> - Whether all test sizes passed correctness checks
> - If any failed: which sizes failed, the magnitude of the error, and a diagnosis using the Structured Diagnosis checklist
> Wait for supervisor approval before proceeding to Step 4.

##### Step 4: Performance Benchmarking
**Script**: `sbatch/step4_benchmark.sh`

**Goal**: Measure performance and calculate speedup vs PyTorch baseline.

**Method**:
```bash
python test.py --benchmark-only
```

**Expected Output**: Timing results, speedup calculation, GFLOPS measurement.

**Benchmark Settings**:
- Warmup runs: 5-10
- Timed runs: 20-100
- Test sizes: Include the problem size from KernelBench

> **SUPERVISOR CHECKPOINT — Step 4:**
> Report to the supervisor:
> - Timing results and speedup over PyTorch baseline
> - Hardware utilization (GFLOPS as percentage of theoretical peak)
> - Whether the performance is satisfactory or further auto-tuning is needed
> Wait for supervisor approval before proceeding to Action 5.

---

#### Mode B: `test_scripts` (Direct GPU-node, no SLURM)

Create a `test_scripts/` directory in the pattern directory, modeled on the validated reference at `<working_directory>/agent_work/pattern_table/gemm/a100/square_matrix_multiplication_tf32/test_scripts/`. This mode runs directly on a GPU node without requiring a SLURM scheduler.

**Directory structure**:
```
test_scripts/
├── config.sh            # Central configuration (CUDA_HOME, CUTLASS_ROOT, GPU_ARCH, PYTHONPATH)
├── 0_check_env.sh       # Pre-flight environment checks
├── 1_compile.sh         # JIT compilation via test.py
├── 2_test_correctness.sh # Correctness verification via test.py
├── 3_benchmark.sh       # Performance benchmarking via test.py
├── 4_autotune.sh        # Auto-tuning wrapper (see Action 5)
├── run_all.sh           # Orchestrator: runs 0→1→2→3 sequentially
├── autotune.py          # Auto-tuning sweep script
└── generate_kernel.py   # Parameterized kernel code generator
```

##### `config.sh` — Central Configuration

All scripts source this file. The agent must adapt these paths to the target GPU node.

```bash
# CUDA
: "${CUDA_HOME:=/usr/local/cuda-12.4}"
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

# CUTLASS
: "${CUTLASS_ROOT:=<path_to_cutlass>}"
export CUTLASS_ROOT

# Project paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="<working_directory>"

# Python path: kernelbench utilities needed by ModelNew.py
export PYTHONPATH="$PROJECT_ROOT/kernelbench_cutlass_module/src:$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

# GPU architecture
: "${GPU_ARCH:=sm_80}"   # sm_80=A100, sm_89=L4/4090, sm_90=H100/H200
export GPU_ARCH
```

##### Step 0: Check Environment
**Script**: `test_scripts/0_check_env.sh`

Verifies that nvcc, CUTLASS headers, PyTorch+CUDA are available on the GPU node. Fails fast with clear error messages indicating what is missing.

> **SUPERVISOR CHECKPOINT — Step 0:**
> Report to the supervisor:
> - Whether all environment checks passed
> - Any missing dependencies or version mismatches
> Wait for supervisor approval before proceeding to Step 1.

##### Step 1: Compile
**Script**: `test_scripts/1_compile.sh`

JIT compilation via `python test.py --compile-only` with verbose output enabled (`KERNELBENCH_CUTLASS_VERBOSE=1`).

> **SUPERVISOR CHECKPOINT — Step 1:**
> Report to the supervisor:
> - Whether compilation succeeded or failed
> - If failed: the full nvcc error output and a preliminary diagnosis using the Structured Diagnosis checklist
> Wait for supervisor approval before proceeding to Step 2.

##### Step 2: Correctness
**Script**: `test_scripts/2_test_correctness.sh`

Correctness verification via `python test.py --correctness-only`. Tolerance: 1e-4 for FP32, 1e-2 for FP16/BF16. Tests multiple sizes (e.g., 512, 1024, 2048).

> **SUPERVISOR CHECKPOINT — Step 2:**
> Report to the supervisor:
> - Whether all test sizes passed correctness checks
> - If any failed: which sizes, error magnitude, and diagnosis
> Wait for supervisor approval before proceeding to Step 3.

##### Step 3: Benchmark
**Script**: `test_scripts/3_benchmark.sh`

Performance benchmarking via `python test.py --benchmark-only`. Reports timing, GFLOPS, and speedup vs PyTorch baseline.

> **SUPERVISOR CHECKPOINT — Step 3:**
> Report to the supervisor:
> - Timing results and speedup over PyTorch baseline
> - Hardware utilization (GFLOPS as percentage of theoretical peak)
> - Whether the performance is satisfactory or further auto-tuning is recommended
> Wait for supervisor approval before proceeding to Action 5.

##### `run_all.sh` — Orchestrator

Runs steps 0→1→2→3 sequentially with `set -euo pipefail`, failing immediately on any step. Prints a summary at the end.

---

#### Using KernelBench Evaluation Utilities

The template's `src/kernelbench/eval.py` module provides the `eval_kernel_against_ref()` function for automated evaluation:

```python
from kernelbench.eval import eval_kernel_against_ref

result = eval_kernel_against_ref(
    original_model_src,     # Model.py source
    custom_model_src,       # ModelNew.py source
    num_correct_trials=1,
    num_perf_trials=10,
    measure_performance=True,
    backend="cutlass_cpp",
    precision=torch.float32,
    verbose=True
)
```

**Returns**: `KernelExecResult` with `compiled`, `correctness`, `runtime` fields.

### Feedback Loop: Structured Diagnosis

When any validation step fails (compile error, correctness mismatch, or poor performance), the agent must perform a structured diagnosis before retrying. The workflow loops back to Action 1, but the diagnosis narrows the search space.

#### Diagnostic Checklist

| Error Symptom | Likely Cause | Where to Check in `cutlass/include/` |
|---------------|-------------|--------------------------------------|
| Template instantiation failure (pages of type errors) | Wrong template params or incompatible type combination | Re-read the device-level header used (e.g., `gemm/device/gemm_universal.h` or `gemm/collective/collective_builder.hpp`) — check template constraints and static_assert conditions |
| Shared memory overflow at compile time | Too many pipeline stages or too large tile shape for the target GPU | `cutlass/arch/mma_sm80.h` or `mma_sm90.h` for shared-memory budget per MMA stage; reduce `Stages` or shrink tile |
| Register overflow / spilling at launch | Too many threads per block or too large warp tile exceeding register file | `cutlass/kernel_launch.h` for max threads per block; `cutlass/arch/sm80.rs/` or `sm90.rs/` for register limits per thread |
| Layout mismatch between A, B, C operands | Incompatible memory layout combination (e.g., row-major A with column-major B) | `cutlass/layout/matrix_layout.h` for valid layout combinations; ensure `LayoutA`, `LayoutB`, `LayoutC` are compatible with the selected MMA operation |
| Epilogue fusion failure (wrong output) | Missing visitor registration or incorrect accumulator type | `cutlass/epilogue/fusion/` for visitor pattern; verify accumulator type matches mainloop output |
| Wrong numerical results (small errors) | Precision mismatch or missing scaling factors | `cutlass/numeric_conversion.h` for conversion behavior; check if TF32 truncation or FP8 scaling is applied correctly |
| Wrong numerical results (large errors) | Incorrect GEMM dimensions, transposed operands, or wrong stride | Check the `GemmCoord` (M, N, K) passed to the kernel; verify leading dimensions match actual tensor strides |
| Kernel launch failure (runtime error) | Invalid grid/block dimensions or insufficient GPU resources | `cutlass/kernel_launch.h` for launch config computation; check occupancy with `cudaOccupancyMaxActiveBlocksPerMultiprocessor` |
| Performance far below expected | Suboptimal tile shape, wrong kernel schedule, or poor occupancy | Re-read the Architecture-Gated Exploration Paths above; try alternative tile shapes, kernel schedules (e.g., ping-pong vs. warp-specialized), or grid scheduling (e.g., Stream-K vs. Split-K) |

#### Diagnosis Procedure

1. **Classify the error** using the table above.
2. **Read the relevant header** from `cutlass/include/` to understand the constraint that was violated.
3. **Identify the fix**: adjust a template parameter, change a layout, reduce pipeline stages, etc.
4. **Re-read the working test/example file** to confirm the fix is consistent with a known-good instantiation.
5. **Present the diagnosis to the supervisor** before modifying the kernel code.
6. **Retry** from Action 1 (re-select supporting examples if the fix requires a fundamentally different approach) or Action 2 (if only parameter adjustments are needed).


> **SUPERVISOR CHECKPOINT:**
> Stop and let the user know that you have reached action 5.
> Wait for supervisor approval before proceeding to Action 5.


### Action 5: Auto-Tuning 


After a pattern passes correctness verification (Action 4, Step 2/3), the agent may perform auto-tuning to find the highest-performing configuration. This step is optional but recommended when the initial benchmark (Action 4, Step 3/4) shows room for improvement.

**Reference implementation**: `<working_directory>/agent_work/pattern_table/gemm/a100/square_matrix_multiplication_tf32/test_scripts/autotune.py` and `generate_kernel.py`.

#### Inferring the Search Space

The auto-tuning search space is **not hardcoded** — it is derived from the decisions made at the three CUTLASS API levels (Tile, Kernel, Grid) and inferred from the test files used during synthesis.

1. **Tile candidates (threadblock_tile, warp_tile)**: Infer from the matching test files in `<CUTLASS_ROOT>/test/unit/gemm/device/`. Read the test files for the target architecture and data type, extract all `GemmShape<...>` template arguments used as threadblock and warp tiles, and collect the unique configurations. These are the candidates to sweep.

2. **Kernel-level parameters**: Determined by the pipelining strategy chosen at the Kernel Level:
   - **Multistage pipelines (Ampere SM80)**: Sweep pipeline stages (typically 2–3 for Ampere; higher stages cause shared-memory overflow). The stage count is a direct template parameter.
   - **Warp-specialized kernels (Hopper SM90)**: Pipeline stages are managed automatically by TMA — do **not** sweep stages. Instead, sweep kernel schedules: `KernelTmaWarpSpecialized` vs. `KernelTmaWarpSpecializedPingpong` (cooperative variant). These are selected via the `KernelSchedule` template parameter in the collective builder.
   - **Persistent kernels**: Sweep cluster shapes (e.g., `Shape<_1,_1,_1>` vs. `Shape<_2,_1,_1>` vs. `Shape<_2,_2,_1>`).

3. **Grid-level parameters**: Determined by the scheduling strategy chosen at the Grid Level:
   - **Data Parallel**: No additional sweep parameter (fixed by tile shape).
   - **Split-K**: Sweep split-K factor or split-K slices.
   - **Stream-K**: No additional sweep parameter (persistent scheduler handles it automatically).

**Example search space construction for Ampere SM80:**
```
Source: ls <CUTLASS_ROOT>/test/unit/gemm/device/sm80*tf32*.cu
→ Read each test file, extract GemmShape<TM,TN,TK> (threadblock) and GemmShape<WM,WN,WK> (warp)
→ Collect unique (threadblock_tile, warp_tile) pairs
→ Combine with stages = [2, 3]  (Ampere shared-memory limit)
→ Search space: {(tile, warp, stage) for tile in tiles for warp in warps for stage in stages}
```

**Example search space construction for Hopper SM90:**
```
Source: ls <CUTLASS_ROOT>/test/unit/gemm/device/sm90*tf32*.cu
→ Read each test file, extract Shape<...> (tile) and cluster shapes
→ Collect unique (tile_shape, cluster_shape) pairs
→ Combine with kernel_schedules = [KernelTmaWarpSpecialized, KernelTmaWarpSpecializedPingpong]
→ Stages: NOT swept (TMA-managed, use StageCountAuto)
→ Search space: {(tile, cluster, schedule) for tile in tiles for cluster in clusters for schedule in schedules}
```

#### Kernel Generation Pattern: `generate_kernel.py`

Instead of compiling a separate binary for each configuration, use a parameterized code generator that emits a single `.cu` file containing multiple template instantiations (one per variant), all compiled into one shared library. A runtime index dispatches to the correct variant.

**How it works:**
1. `generate_kernel.py` takes tile/warp shapes (and any other varying parameters) as CLI arguments.
2. It emits a `.cu` file with N type aliases (e.g., one per pipeline stage count or one per kernel schedule), each instantiating the same CUTLASS GEMM template with different parameters.
3. It emits a switch/dispatch function that selects the correct instantiation at runtime via an integer index.
4. A `binding.cpp` provides the pybind11 interface.

**Benefits:**
- Avoids recompiling the entire kernel for each stage/schedule variant.
- Reduces auto-tuning wall-clock time significantly.
- The generated binary is self-contained — no external template files needed at runtime.

**Adaptation for different patterns:**
- For **epilogue fusion**: add epilogue visitor types to the generated variants.
- For **Split-K**: add split-K factor as a variant dimension.
- For **Hopper collective API**: generate variants with different `KernelSchedule` and `ClusterShape` template params instead of pipeline stages.

#### Auto-Tuning Script: `autotune.py`

The auto-tuner sweeps all configurations from the inferred search space:

1. **Benchmark the PyTorch baseline** at the target problem size (warmup + timed runs).
2. **For each (tile, warp, ...) configuration**:
   a. Call `generate_kernel.py` to emit the parameterized kernel.
   b. Compile via `torch.utils.cpp_extension.load()`.
   c. If compilation fails, record the error and skip.
   d. If compilation succeeds, benchmark each variant (warmup + timed runs).
3. **Report a table** with columns: tile, warp, variant_param, time (ms), GFLOPS, efficiency (%), speedup vs PyTorch.
4. **Identify the best configuration** (lowest time / highest GFLOPS).
5. **Save full results** as JSON in `<pattern_dir>/results/auto-tune/`.

#### Step 4: Run Auto-Tuning
**Script**: `test_scripts/4_autotune.sh` (or `sbatch/step5_autotune.sh` for SLURM mode)

Wraps the `autotune.py` script. Requires that Step 2 (correctness) has passed first.

> **SUPERVISOR CHECKPOINT — Step 4 (Auto-Tuning):**
> Report to the supervisor:
> - The inferred search space (tile candidates, varying parameters, total number of configurations)
> - The full sweep results table
> - The best configuration and its performance (GFLOPS, efficiency%, speedup)
> - Comparison with the initial benchmark from Action 4
> Wait for supervisor approval before proceeding to Action 6.

### Action 6: Add to Pattern Table

Once accepted, add pattern to `<working_directory>/agent_work/pattern_table/` following this structure.

**Pattern Table Directory Structure:**
```
pattern_table/
└── <rule>/                              # e.g., gemm, fmha, conv2d
    └── <gpu>/                           # e.g., a100, h100
        ├── <pattern_name>/              # Pattern implementation directory
        │   ├── Model.py
        │   ├── ModelNew.py
        │   ├── cutlass_kernels/
        │   └── test.py
        └── <pattern_name>.json          # Pattern metadata (in same directory)
```

**Important**: The pattern JSON metadata file should be created in the **same directory** as the pattern implementation, not in a separate location.

**pattern_info.json Structure:**
```json
{
  "pattern_id": "unique_id",
  "name": "Pattern Name",
  "timestamp": "ISO_8601_timestamp",
  "kernelbench_problem": {
    "level": 1,
    "problem_id": 1,
    "problem_file": "filename.py",
    "description": "Brief description"
  },
  "target_architecture": {
    "gpu": "A100/H100",
    "compute_capability": "SM80/SM90",
    "tensor_cores": "TF32/BF16/FP8"
  },
  "optimization_rule": "GEMM/FMHA/CONV2D",
  "data_type": "float32/float16/bfloat16",
  "computation_precision": "tf32/bf16/fp16",
  "input_shapes": {},
  "implementation": {
    "cutlass_version": "3.x.x",
    "module": "nvidia-cutlass/3.8.0.0",
    "tile_config": {
      "threadblock_tile": "description",
      "warp_tile": "description",
      "instruction_shape": "description"
    },
    "kernel_level": {
      "pipelining_strategy": "multistage/warp_specialized",
      "description": "Software pipelining approach for target architecture"
    },
    "grid_level": {
      "scheduling": "data_parallel/split_k/stream_k",
      "description": "Work partitioning strategy"
    },
    "epilogue": "description",
    "layout": "description",
    "kernel_files": []
  },
  "verification": {
    "correctness": "passed",
    "tolerance": 1e-4,
    "test_sizes": [],
    "notes": "Any correctness observations"
  },
  "performance": {
    "problem_size": 4096,
    "baseline_time_ms": 0.0,
    "custom_time_ms": 0.0,
    "speedup": 1.0,
    "overhead_percent": 0.0,
    "throughput_gflops": 0.0,
    "peak_utilization_percent": 0.0,
    "iterations": 100
  },
  "status": "accepted",
  "validation_workflow": {
    "step1_setup_env": "passed",
    "step2_compilation": "passed",
    "step3_correctness": "passed",
    "step4_benchmark": "passed"
  },
  "supporting_examples": [],
  "pattern_directory": "<working_directory>/agent_work/pattern_table/<rule>/<gpu>/<pattern_name>/",
  "template_reference": "<working_directory>/kernelbench_cutlass_module",
  "notes": "Additional observations"
}
```

**Example Pattern Entry:**
See `<working_directory>/agent_work/pattern_table/gemm/a100/square_matrix_multiplication_tf32/pattern_info.json` for a validated example.

**Required Fields:**
- `pattern_id`: Unique identifier (e.g., `gemm_<pattern>_<precision>_<sm>`)
- `optimization_rule`: One of: GEMM, FMHA, CONV2D, etc.
- `target_architecture.gpu`: A100, H100, etc.
- `target_architecture.compute_capability`: SM80, SM90, etc.
- `implementation.tile_config`: Tile-level decisions (threadblock_tile, warp_tile, instruction_shape)
- `implementation.kernel_level`: Kernel-level pipelining strategy
- `implementation.grid_level`: Grid-level scheduling strategy
- `status`: One of: accepted, failed, pending
- `pattern_directory`: Path to the pattern implementation directory (where the code lives)
- `template_reference`: Path to the template directory (kernelbench_cutlass_module)
- `validation_workflow`: Status of each validation step (all 4 steps)
