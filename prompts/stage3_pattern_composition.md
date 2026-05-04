# Stage 3: Pattern Composition

## Task Definition

You are an AI agent specialized in composing optimized kernels into a complete PyTorch model. Your task is to assemble all realized patterns from Stage 2 into an optimized model (ModelNew), load the compiled extensions, perform end-to-end verification and benchmarking against the baseline.

## Context

- **Target Model**: KernelBench Level 3 problem - `get from user`
- **Target GPU Architecture**: Ampere (SM80)
- **Pattern Table**: `/Users/raycatcher/Desktop/cutlass_agent/agent_work/pattern_table`
- **Output Directory**: `/Users/raycatcher/Desktop/cutlass_agent/agent_work/whole_model`

## Actions to Execute

### Action 1: Load Pattern Table

Read all accepted patterns from Stage 2:

```python
import json
from pathlib import Path

pattern_table_dir = Path("/home/sinaheidari/marco_optimizer/agent_work/pattern_table")
accepted_patterns = []

for pattern_dir in pattern_table_dir.iterdir():
    if not pattern_dir.is_dir():
        continue
    info_file = pattern_dir / "pattern_info.json"
    if info_file.exists():
        with open(info_file) as f:
            info = json.load(f)
            if info.get("status") == "accepted":
                accepted_patterns.append(info)
```

### Action 2: Analyze Pattern Composition

For each accepted pattern:
1. Identify which subgraph(s) it replaces in the baseline model
2. Determine dependencies between patterns (e.g., pattern A's output is pattern B's input)
3. Plan the order of execution in the forward pass

**Composition Strategy:**
- **Independent patterns**: Can be applied in parallel
- **Sequential patterns**: Must be applied in dependency order
- **Overlapping patterns**: Choose the best performing one

**Stop Point**: Stop here and give me a your plan

### Action 3: Create Composed ModelNew

Create a new PyTorch module that:
1. Loads all CUTLASS extensions from the pattern table
2. Replaces optimized subgraphs with CUTLASS kernel calls
3. Preserves the original model's input/output interface
4. Falls back to PyTorch implementation for non-optimized parts

#### Create ModelNew 

You could see ModelNew templates in the selected pattern directories. But your ModelNew must load all extensions related to selected patterns.


### Action 4: Compile and Load Extensions

You must replicate the four-step test scripts we have under the test_scripts directory under each pattern directory. 
**Stop Point**: Here, stop and ask for autotuning results so that you could select the best found parameters for the problems sizes.
After receiving the path for auto-tuning results, generate the four-step validation workflow (you do not have to include auto-tuning scripts).


### Action 7: Create Summary Report

After verification and benchmarking, create a comprehensive summary:

```json
{
  "timestamp": "2026-04-13T06:30:00",
  "target_model": "44_MiniGPTBlock.py",
  "target_architecture": "SM90",
  "stages": {
    "stage1": {
      "patterns_proposed": 3,
      "patterns_prioritized": ["p1", "p2"]
    },
    "stage2": {
      "patterns_accepted": 2,
      "patterns_failed": 1,
      "accepted_patterns": ["p1", "p2"]
    },
    "stage3": {
      "correctness": "passed",
      "speedup": 2.15,
      "baseline_time_ms": 12.5,
      "optimized_time_ms": 5.8,
      "patterns_used": ["p1_fmha", "p2_mlp"]
    }
  },
  "overall_speedup": 2.15,
  "notes": [
    "FMHA pattern provides 2.5x speedup for attention block",
    "MLP pattern provides 1.8x speedup for MLP block",
    "Combined effect: 2.15x overall speedup"
  ]
}
```
# Stop and let me run the validation test scripts.