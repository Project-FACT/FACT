# Stage 3: Pattern Composition

## Task Definition

You are an AI agent specialized in composing optimized kernels into a complete PyTorch model. Your task is to assemble all realized patterns from Stage 2 into an optimized model (`ModelNew`), load the compiled CUTLASS extensions, perform end-to-end verification and benchmarking with per-pattern ablation against the baseline.

## Context

| Setting | Value |
|---------|-------|
| Target Model | `Extract from proposed patterns file` |
| Target GPU | `Extract from proposed patterns file` |
| working_directory | `<FACT_ROOT>` |
| Pattern Table Base | `<working_directory>/pattern_table/` |
| Module Setup Template | `<working_directory>/kernelbench_cutlass_module/` |
| Proposed Patterns | `<working_directory>/proposed_patterns/patterns_*.json` |
| Output Directory | `<working_directory>/examples/whole_model/<model_name>/` |
| Baseline Model | `<working_directory>/KernelBench/KernelBench/level3/<problem_file>.py` |
| Validation Mode | `test_scripts` |
| Reference Examples | `<working_directory>/examples/whole_model/` |

## Actions to Execute

```
Action 1: Load Pattern Table
         ↓
Action 2: Analyze Pattern Composition
         ↓ (supervisor checkpoint — present plan)
Action 3: Create Composed ModelNew
         ↓
Action 4: Create Validation Test Scripts
         ↓ (supervisor checkpoint — ask for auto-tuning results)
Action 5: End-to-End Correctness Verification
         ↓ (supervisor checkpoint — report correctness)
Action 6: Ablation Benchmarking
         ↓ (supervisor checkpoint — report benchmarks)
Action 7: Create Summary Report
```

---

### Action 1: Load Pattern Table

Recursively scan the pattern table to find all accepted patterns from Stage 2. The pattern table is organized by rule, data type, and architecture:

```
pattern_table/
└── <rule>/                    # e.g., gemm, fmha
    └── <dtype>/               # e.g., tf32, fp16
        └── <arch>/            # e.g., sm80, sm90
            └── <pattern_name>/
                ├── pattern_info.json   # Status, config, performance
                ├── ModelNew.py         # CUTLASS implementation
                └── ...
```

```python
import json
from pathlib import Path

pattern_table_dir = Path("<working_directory>/pattern_table")
accepted_patterns = []

for info_file in pattern_table_dir.rglob("pattern_info.json"):
    with open(info_file) as f:
        info = json.load(f)
        if info.get("status") == "accepted":
            info["_pattern_dir"] = str(info_file.parent)
            accepted_patterns.append(info)
```

For each accepted pattern, record:
- `pattern_id`, `optimization_rule`, `data_type`, `target_architecture`
- Path to the pattern directory (needed for extension loading)
- Auto-tuning results path (if available)
- The subgraph it replaces in the baseline model

---

### Action 2: Analyze Pattern Composition

For each accepted pattern, determine how it fits into the baseline model's computation graph:

1. **Map patterns to subgraphs.** Read the baseline model source and identify which operations each pattern replaces. For example:
   - Pattern `p1` (FMHA) → replaces the attention subgraph (QKV projection + scaled dot-product attention + output projection)
   - Pattern `p2` (MLP Fusion) → replaces the MLP subgraph (linear + activation + linear)

2. **Determine dependencies.** Check whether patterns have data dependencies:
   - **Independent**: Pattern A and B operate on disjoint subgraphs with no tensor flowing between them (e.g., attention block and MLP block in a transformer, connected only by residual addition)
   - **Sequential**: Pattern B's input is Pattern A's output — must preserve tensor shape and dtype at the boundary
   - **Overlapping**: Two patterns cover overlapping subgraphs — choose the one with higher speedup

3. **Plan execution order.** The forward pass must execute patterns in dependency order, with PyTorch fallback for unoptimized subgraphs.

**Output a composition plan** containing:
- Subgraph → pattern mapping
- Dependency graph between patterns
- Forward pass execution order
- Tensor shapes at each pattern boundary
- Any precision/dtype conversions needed at boundaries

> **SUPERVISOR CHECKPOINT**: Stop here and present your composition plan. Wait for approval before proceeding to Action 3.

---

### Action 3: Create Composed ModelNew

Create a composed `ModelNew.py` that loads all CUTLASS extensions and replaces the corresponding subgraphs while preserving the baseline's input/output interface.

#### Extension Loading Pattern

Each pattern's CUTLASS extension is loaded from its `ModelNew.py` in the pattern table. Use `importlib.util` with unique module names to avoid conflicts:

```python
def _load_pattern_modelnew(directory: str, unique_name: str):
    """Load a pattern's ModelNew.py from disk."""
    path = Path(directory) / "ModelNew.py"
    spec = importlib.util.spec_from_file_location(unique_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = mod
    spec.loader.exec_module(mod)
    return mod
```

Load each pattern's extension at module level, wrapped in try/except with a fallback flag:

```python
# Load FMHA extension
try:
    _fmha_mod = _load_pattern_modelnew(fmha_pattern_dir, "composed_fmha_pattern")
    _fmha_ext = _fmha_mod._get_cutlass_ext()
    _fmha_available = True
except Exception as e:
    print(f"Warning: Failed to load FMHA extension: {e}")
    _fmha_available = False
    _fmha_ext = None
```

**CRITICAL:** Each `importlib` call must use a globally unique module name (e.g., `"composed_fmha_pattern"`, `"composed_mlp_pattern"`) to prevent namespace collisions when loading multiple patterns.

#### Sub-Module Dispatch Pattern

Each sub-module (attention block, MLP block, etc.) checks its extension availability and dispatches accordingly:

```python
class CausalSelfAttention(nn.Module):
    def forward(self, x):
        # ... QKV projection ...
        if _fmha_available:
            y = self._cutlass_fmha_forward(q, k, v)
        else:
            # Exact baseline fallback
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = F.softmax(att, dim=-1)
            y = att @ v
        # ... output projection ...
```

The fallback must reproduce the exact baseline computation, including dropout, masking, and any activation functions.

#### Ablation Support

There are two approaches for ablation benchmarking (Action 6). Choose one based on the model complexity:

**Approach A — Separate-file ablation** (better for simpler models like MiniGPT):
- `ModelNew.py` — full composition (all patterns)
- `ModelNew_<pattern>_only.py` — one file per pattern, all others in PyTorch fallback
- `ablation_benchmark.py` — loads all variants, runs benchmarks, produces comparison table

In each `<pattern>_only.py`, set the other patterns' availability flags to `False`:
```python
# In ModelNew_fmha_only.py:
_fmha_available = True   # CUTLASS FMHA
_mlp_available = False   # PyTorch MLP fallback
```

**Approach B — Constructor-switch ablation** (better for complex models like Llama 3):
- Single `ModelNew.py` with `enable_<pattern>` constructor arguments
- `test.py --benchmark-mode all` runs all ablation variants

```python
class Model(nn.Module):
    def __init__(self, ..., enable_fmha=True, enable_swiglu=True):
        super().__init__()
        self.enable_fmha = enable_fmha
        self.enable_swiglu = enable_swiglu
        # Sub-modules check both extension availability AND enable flags
        use_cutlass_fmha = enable_fmha and _fmha_ext is not None
        self.attn = LlamaAttention(..., use_cutlass=use_cutlass_fmha)
```

**Reference implementations:**
- Approach A: `examples/whole_model/miniGPT/` (separate files)
- Approach B: `examples/whole_model/llama3_8b/` (constructor switches)

#### Pattern Status API

Include a `get_pattern_status()` method that reports extension availability and configuration:

```python
def get_pattern_status(self):
    return {
        'fmha_available': _fmha_available,
        'mlp_available': _mlp_available,
        'fmha_config': {...},  # From auto-tuning results
        'mlp_config': {...},
    }
```

#### Weight Copying

The composed model must accept the same init arguments as the baseline and use the same weight names so that weights can be copied from the baseline:

```python
with torch.no_grad():
    opt_model.attn.c_attn.weight.copy_(base_model.attn.c_attn.weight)
    opt_model.attn.c_proj.weight.copy_(base_model.attn.c_proj.weight)
    opt_model.mlp.c_fc.weight.copy_(base_model.mlp.c_fc.weight)
    # ... etc ...
```

#### KernelBench Interface

Include the standard KernelBench interface at the bottom of `ModelNew.py`:

```python
batch_size = ...
seq_len = ...
n_embd = ...

def get_inputs():
    return [torch.rand(batch_size, seq_len, n_embd)]

def get_init_inputs():
    return [n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen]
```

---

### Action 4: Create Validation Test Scripts

Create a `test_scripts/` directory in the output directory with a four-step validation pipeline. This mirrors Stage 2's validation mode but operates on the composed model rather than individual patterns.

#### Directory Structure

```
test_scripts/
├── config.sh              # Central configuration
├── 1_check_env.sh         # Pre-flight environment checks
├── 2_compile.sh           # JIT compilation of all extensions
├── 3_test_correctness.sh  # End-to-end correctness verification
├── 4_benchmark.sh         # Performance benchmarking
└── run_all.sh             # Orchestrator: runs 1→2→3→4 sequentially
```

#### `config.sh` — Central Configuration

All scripts source this file. Derive paths from script location:

```bash
_THIS_CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WHOLE_MODEL="$(cd "$_THIS_CONFIG_DIR/.." && pwd)"
export FACT_ROOT="$(cd "$_THIS_CONFIG_DIR/../../.." && pwd)"

# CUDA
: "${CUDA_HOME:=/usr/local/cuda}"
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# CUTLASS headers
if [[ -z "${CUTLASS_ROOT:-}" && -d "$FACT_ROOT/cutlass/include/cutlass" ]]; then
  export CUTLASS_ROOT="$FACT_ROOT/cutlass"
fi

# kernelbench utilities
_KB_SRC="$FACT_ROOT/kernelbench_cutlass_module/src"
if [[ -d "$_KB_SRC" ]]; then
  export PYTHONPATH="$_KB_SRC${PYTHONPATH:+:$PYTHONPATH}"
fi

# GPU architecture
: "${GPU_ARCH:=sm_80}"
export GPU_ARCH
```

#### Step 1: Check Environment (`1_check_env.sh`)

Verify nvcc, CUTLASS headers, PyTorch+CUDA availability, and that the baseline model file exists.

> **SUPERVISOR CHECKPOINT — Step 1:**
> Report whether all environment checks passed. Wait for approval before proceeding.

#### Step 2: Compile (`2_compile.sh`)

JIT-compile all CUTLASS extensions by importing `ModelNew`:
```bash
KERNELBENCH_CUTLASS_VERBOSE=1 python -c "import sys; sys.path.insert(0, '$WHOLE_MODEL'); import ModelNew; print('All extensions compiled')"
```

> **SUPERVISOR CHECKPOINT — Step 2:**
> Report whether all extensions compiled. If any failed, show the nvcc error and diagnose using the checklist in Action 5. Wait for approval.

#### Step 3: Correctness (`3_test_correctness.sh`)

Run `python test.py --correctness-only` to verify end-to-end correctness. See Action 5 for details.

> **SUPERVISOR CHECKPOINT — Step 3:**
> Report correctness results across all test sizes. If any failed, diagnose and loop back. Wait for approval.

#### Step 4: Benchmark (`4_benchmark.sh`)

Run `python test.py --benchmark-only` or the ablation benchmark. See Action 6 for details.

> **SUPERVISOR CHECKPOINT — Step 4:**
> Report benchmark results and per-pattern ablation contributions. Wait for approval.

#### `run_all.sh` — Orchestrator

```bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
./1_check_env.sh && ./2_compile.sh && ./3_test_correctness.sh && ./4_benchmark.sh
```

#### Top-Level `test.py`

Create a `test.py` in the output directory with standard flags:

```bash
python test.py --correctness-only   # Correctness across multiple sizes
python test.py --benchmark-only     # End-to-end performance
```

Reference template: `kernelbench_cutlass_module/work/test.py`

**Reference implementations:**
- `examples/whole_model/miniGPT/test.py` — correctness + benchmark with separate ablation
- `examples/whole_model/llama3_8b/test.py` — correctness + multi-mode ablation with `--benchmark-mode`

---

### Action 5: End-to-End Correctness Verification

After compilation succeeds, verify the composed model produces correct results against the baseline model.

#### Procedure

1. **Load baseline model** from the KernelBench problem file using `importlib.util` with a unique module name.
2. **Load composed model** from `ModelNew.py` with a unique module name.
3. **Instantiate both models** with identical init arguments on CUDA.
4. **Copy weights** from baseline to composed model (all linear layers, normalization layers, etc.).
5. **Set to eval mode** (`model.eval()`) and disable gradients (`torch.no_grad()`).
6. **Test at multiple input sizes** — at least three sizes ranging from small to the full problem size. Example:
   ```python
   test_sizes = [
       (2, 64, 768),     # Small
       (4, 128, 768),    # Medium
       (16, 256, 768),   # Large
       (128, 512, 768),  # Full problem size
   ]
   ```
7. **Compare outputs** element-wise: `torch.max(torch.abs(base_out - opt_out))`.
8. **Check pattern availability** — verify all expected extensions loaded by calling `get_pattern_status()`.

#### Tolerance

| Precision | Tolerance |
|-----------|-----------|
| FP32 / TF32 | 1e-3 to 1e-4 |
| FP16 / BF16 | 1e-2 to 1e-3 |

Mixed-precision patterns (FP16 compute, FP32 output) may require the looser end of the range.

#### Diagnostic Checklist for Composition-Specific Failures

| Error Symptom | Likely Cause | Fix |
|---------------|-------------|-----|
| Shape mismatch at pattern boundary | Wrong reshape/transpose between patterns | Print tensor shapes at subgraph entry and exit; verify the CUTLASS kernel's expected input format |
| dtype mismatch (e.g., FP16 output feeding FP32 input) | Mixed-precision pattern without explicit cast | Add `.to(torch.float32)` at pattern boundary or ensure kernel output dtype matches downstream expectation |
| Extension import conflict (`module already imported`) | Two patterns loaded with same `importlib` module name | Use globally unique names: `"composed_fmha_pattern"`, `"composed_mlp_pattern"` |
| Large numerical error only in composed model (not per-pattern) | Error accumulation across patterns or incorrect weight copying | Test each pattern individually first; verify weight copy covers all layers |
| Segfault during forward pass | Extension compiled for wrong GPU architecture | Check `GPU_ARCH` in `config.sh` matches target GPU (`sm_80` for A100, `sm_90` for H100) |
| One extension loads but the other fails | Missing pattern directory or compilation error in one kernel | Check `pattern_info.json` status for each pattern; verify all `.cu` files exist |
| Correctness passes at small sizes but fails at large sizes | Tile-shape edge case or overflow | Verify auto-tuned tile configs handle the full problem size; check for integer overflow in grid dimensions |

#### Diagnosis Procedure

1. Classify the error using the table above.
2. Isolate: test each pattern individually to narrow which pattern causes the failure.
3. Check boundary: print tensor shapes and dtypes at each pattern entry/exit.
4. Compare against baseline: verify the fallback path produces identical results to the original model.
5. Present diagnosis to the supervisor before modifying code.
6. If the fix requires re-synthesis, loop back to Action 3.

> **SUPERVISOR CHECKPOINT — Action 5:**
> Report:
> - Whether all test sizes passed correctness
> - Pattern availability status
> - If any failed: which sizes, error magnitude, and diagnosis
> Wait for approval before proceeding to Action 6.

---

### Action 6: Ablation Benchmarking

Measure per-pattern contribution to the overall speedup. This is required for all multi-pattern compositions.

#### Benchmark Protocol

For each configuration, run warmup + timed iterations with CUDA synchronization:

```python
# Warmup
for _ in range(warmup):
    with torch.no_grad():
        _ = model(x)
torch.cuda.synchronize()

# Timed runs
times = []
for _ in range(trials):
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        _ = model(x)
    torch.cuda.synchronize()
    times.append(time.perf_counter() - start)
avg_time = sum(times) / len(times) * 1000  # ms
```

**Default settings:** warmup=5, trials=20 (matching the paper's methodology).

#### Ablation Configurations

Run 2^N + 1 configurations (where N is the number of patterns):

| Configuration | Patterns Enabled | Purpose |
|---------------|-----------------|---------|
| Baseline | None | Reference (PyTorch eager) |
| p1 only | p1 | Measure p1's individual contribution |
| p2 only | p2 | Measure p2's individual contribution |
| ... | ... | One per pattern |
| Full | All | Measure combined speedup |

#### Reporting

```
Configuration       Time (ms)    Speedup
------------------------------------------------------------
Baseline            X            1.00x
FMHA only           Y1           X/Y1
MLP only            Y2           X/Y2
Both                Z            X/Z

Per-pattern contributions:
  FMHA: (X - Y1) / X * 100% latency reduction
  MLP:  (X - Y2) / X * 100% latency reduction
  Combined: (X - Z) / X * 100% latency reduction
```

Also compute throughput (GFLOPS) and hardware efficiency (% of peak):
```python
flops = ...  # Problem-specific FLOP count
throughput_gflops = flops / (avg_time_ms / 1000) / 1e9
efficiency = throughput_gflops / peak_gflops * 100
```

**Reference implementations:**
- Separate-file approach: `examples/whole_model/miniGPT/ablation_benchmark.py`
- Constructor-switch approach: `examples/whole_model/llama3_8b/test.py --benchmark-mode all`

> **SUPERVISOR CHECKPOINT — Action 6:**
> Report:
> - The full ablation table
> - Per-pattern contribution breakdown
> - Throughput and hardware efficiency
> - Whether the combined speedup is superlinear or sublinear relative to individual patterns
> Wait for approval before proceeding to Action 7.

---

### Action 7: Create Summary Report

After verification and benchmarking, create `composition_report.json` in the output directory:

```json
{
  "timestamp": "ISO_8601_timestamp",
  "target_model": "44_MiniGPTBlock",
  "target_architecture": "SM80",
  "composition_approach": "separate_file | constructor_switch",
  "patterns": {
    "p1": {
      "rule": "FMHA",
      "data_type": "fp16",
      "pattern_dir": "pattern_table/fmha/fp32/sm80/fused_multi_head_attention/",
      "autotune_config": {
        "queries_per_block": 64,
        "keys_per_block": 128
      },
      "extension_loaded": true
    },
    "p2": {
      "rule": "GEMM_EPILOGUE_FUSION",
      "data_type": "tf32",
      "pattern_dir": "pattern_table/gemm/tf32/sm80/mlp_gemm_fusion_gelu/",
      "autotune_config": {
        "tile": [128, 256, 32],
        "warp": [64, 64, 32],
        "stages": 4
      },
      "extension_loaded": true
    }
  },
  "pattern_dependency_graph": {
    "p1": {"depends_on": [], "feeds_into": ["residual_add"]},
    "p2": {"depends_on": [], "feeds_into": ["residual_add"]}
  },
  "inter_pattern_tensor_shapes": {
    "ln_1_output": [128, 512, 768],
    "attn_output": [128, 512, 768],
    "ln_2_output": [128, 512, 768],
    "mlp_output": [128, 512, 768]
  },
  "correctness": {
    "status": "passed",
    "tolerance": 1e-3,
    "test_sizes": [[2, 64, 768], [4, 128, 768], [16, 256, 768], [128, 512, 768]]
  },
  "ablation_results": {
    "baseline_time_ms": 25.665,
    "p1_only_time_ms": 20.195,
    "p2_only_time_ms": 17.803,
    "full_time_ms": 12.654,
    "p1_speedup": 1.27,
    "p2_speedup": 1.44,
    "full_speedup": 2.03
  },
  "throughput": {
    "gflops": null,
    "peak_efficiency_percent": null
  },
  "output_directory": "examples/whole_model/miniGPT/"
}
```

---

## Error Handling

When any validation step fails (compile error, correctness mismatch, or poor performance), follow this procedure:

1. **Isolate** the failing pattern by testing each one individually.
2. **Classify** the error using the diagnostic checklist in Action 5.
3. **Diagnose** by checking tensor shapes, dtypes, and weight alignment at pattern boundaries.
4. **Fix** the composed `ModelNew.py` — do not modify the pattern directories (they are immutable outputs of Stage 2).
5. **Re-test** from the failed step forward.

If a pattern's extension cannot be loaded (compilation failure or runtime error), the model should gracefully fall back to PyTorch for that subgraph. Report the failure to the supervisor and continue with the remaining patterns.

## Output Format

The output directory should contain:

```
examples/whole_model/<model_name>/
├── ModelNew.py                 # Full composition (all patterns)
├── ModelNew_<pattern>_only.py  # One per pattern (Approach A) — optional
├── ablation_benchmark.py       # Ablation runner (Approach A) — optional
├── test.py                     # Validation script
├── composition_report.json     # Summary report
├── test_scripts/
│   ├── config.sh
│   ├── 1_check_env.sh
│   ├── 2_compile.sh
│   ├── 3_test_correctness.sh
│   ├── 4_benchmark.sh
│   └── run_all.sh
└── README.md
```

## Additional Instructions

1. **Do not modify pattern directories.** The composed model loads extensions from the pattern table as-is. If a pattern needs adjustment, that is a Stage 2 concern.
2. **Preserve numerical equivalence.** The composed model must produce outputs matching the baseline within tolerance. Mixed-precision patterns must handle dtype conversions explicitly at boundaries.
3. **Test edge sizes.** Always include at least one test size smaller than the full problem to catch tile-shape edge cases.
4. **Report honestly.** If a pattern provides no speedup or degrades performance in composition, report it. Negative results are valuable for the pattern registry.
5. **Handle missing extensions gracefully.** Not every pattern may have a valid CUTLASS extension. The composed model should still work (via PyTorch fallback) even if some extensions fail to load.
