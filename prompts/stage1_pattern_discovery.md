# Stage 1: Pattern Discovery

## Task Definition

You are an AI agent specialized in discovering optimization opportunities in PyTorch computation graphs. Your task is to analyze a baseline PyTorch model and identify subgraphs that can be optimized using CUTLASS kernels.

## Context

- **Target Model**: KernelBench problem - "The user will provide this."
- **Target GPU Architecture**: "The user will provide this."
- **working_directory**: `/home/sinaheidari/marco_optimizer/`
- **examples_index**: `<working_directory>/examples_index.txt`
- **CUTLASS_ROOT**: `<working_directory>/cutlass`
- **pattern_table**: `<working_directory>/agent_work/pattern_table/`

## Actions to Execute

### Action 1: Read Instruction Template

You are searching for the following CUTLASS patterns:
- **GEMM patterns**: Basic matrix multiplication, grouped GEMM, batched GEMM
- **Attention mechanisms**: Fused Multi-Head Attention (FMHA)
- **Epilogue fusion**: Operations fused with GEMM such as bias addition, activation functions (ReLU, GELU), softmax, layer normalization
- **Convolution patterns**: 2D convolution forward/backward (if applicable)

**Target Architecture Considerations:**
- **Hopper (SM90)**: Warp-specialized designs, TMA data movement, FP8 support, grouped operations
- **Ampere (SM80)**: Multi-stage pipelines with cp.async, TF32 tensor core math, sparse tensor operations

### Action 2: Analyze Computation Graph

1. Load the target model from problem directory.
2. **Use PyTorch tools** to programmatically extract the computation graph:
   - Instantiate the model with sample inputs
   - Use `torch.jit.trace()` or `torch.fx.Tracer()` to capture the computation graph
   - Use `torch.fx.GraphModule` to inspect nodes and their properties
3. **Programmatically identify** using the traced graph:
   - Input tensor shapes and data types (via `tensor.shape`, `tensor.dtype`)
   - Sequence of operations in the forward pass (via `graph.nodes`)
   - Dependencies between operations (via `node.args`, `node.users`)
   - Potential fusion points (by analyzing node adjacency and compatibility)

**Key Operations to Look For:**
- Linear layers (matrix multiplication)
- Layer normalization
- Attention mechanisms (QKV projection, softmax, output projection)
- Activation functions (GELU, ReLU, softmax)
- Element-wise operations that can be fused

### Action 3: Query Examples Index

For each candidate subgraph identified in Action 2, retrieve relevant CUTLASS examples from:
- `<working_directory>/examples_index`
- OR directly from `<CUTLASS_ROOT>/examples/` directory

**Retrieval Criteria:**
- Optimization rule similarity
- Data type compatibility
- Input shape similarity
- GPU architecture match

### Action 3.5: Check Pattern Table for Reuse

If the pattern_table directory exists and contains patterns:

1. Search for entries matching the optimization_rule (e.g., "FMHA", "GEMM", "GEMM with epilogue fusion")

2. For each match, check compatibility:
   - Same target architecture (e.g., SM80 for Ampere)?
   - Compatible data type (exact match or convertible)?
   - Similar input shapes (within 2× range)?

3. Classify each pattern:
   - **REUSE**: Exact match - use existing kernel as-is
   - **ADAPT**: Compatible but needs modification - modify existing kernel code
   - **NEW**: No compatible pattern found - synthesize from CUTLASS examples

4. Document reuse decision with:
   - Strategy (REUSE/ADAPT/NEW)
   - Existing pattern path (if REUSE or ADAPT)
   - Adaptations needed (if ADAPT)

If pattern_table is empty or no matches found, all patterns are NEW.

### Action 4: Propose Patterns

Based on your analysis, propose a set of optimization patterns $\mathcal{P}_{\text{proposed}}$.

**Data type variants.**
Each optimization rule can be instantiated with multiple data types, and each $(r, \tau, \alpha, \sigma)$ combination is a distinct entry in the pattern table with its own performance profile. For each candidate pattern, consider all data types that the target architecture supports:
- **Ampere (SM80)**: FP32 (via TF32 tensor cores), FP16, BF16
- **Hopper (SM90)**: FP32 (via TF32 tensor cores), FP16, BF16, FP8 (E4M3, E5M2)

The model's native data type (e.g., FP32) is always a candidate, but lower-precision alternatives may yield higher throughput at the cost of reduced numerical precision. When proposing a lower-precision variant, note the expected precision trade-off and whether the workload is tolerant of it.

For each pattern variant, specify:
1. **Subgraph**: The specific operations to be replaced (e.g., "QKV projection + attention + output projection")
2. **Optimization Rule**: The CUTLASS pattern type (e.g., FlashAttention, FMHA, GEMM with epilogue fusion)
3. **Data Type ($\tau$)**: The computation precision (e.g., tf32, fp16, bf16, fp8)
4. **Reuse Info**: 
   - **Strategy**: REUSE, ADAPT, or NEW
   - **Existing Pattern Path**: Path to reusable pattern (if REUSE or ADAPT)
   - **Adaptations**: List of modifications needed (if ADAPT)
5. **Supporting Examples**: Paths to relevant CUTLASS examples for this rule + data type + architecture combination
6. **Priority**: Expected performance impact (high/medium/low)
7. **Rationale**: A concise explanation of why this pattern (rule + data type combination) is worth exploring for the target workload and architecture. Consider:
   - Why this data type is appropriate for the target architecture and problem (e.g., "FP16 doubles throughput vs TF32 on SM80 while maintaining sufficient precision for inference")
   - Why this optimization rule matches the workload characteristics (e.g., "Large K dimension relative to M/N motivates Stream-K scheduling")
   - Any expected challenges or trade-offs (e.g., "FP8 may introduce quantization error for attention scores")

**Example**: For a GEMM pattern on Ampere, the agent should propose separate variants:
- `p1`: GEMM with TF32 tensor cores, strategy=ADAPT from existing square GEMM pattern
- `p1b`: GEMM with FP16 tensor cores, strategy=NEW if no FP16 pattern exists
- `p1c`: GEMM with FP8 tensor cores, strategy=NEW (requires Hopper)

### Action 5: Prioritize Patterns

Order your proposed patterns by:
1. Expected performance gain
2. Implementation complexity
3. Quality of supporting examples

## Output Format

Create a JSON file with timestamp: `<working_directory>/agent_work/proposed_patterns/patterns_YYYYMMDD_HHMMSS.json`

**JSON Structure:**
```json
{
  "timestamp": "2026-04-13T04:30:00",
  "target_model": "44_MiniGPTBlock.py",
  "target_architecture": "SM90",
  "proposed_patterns": [
    {
      "pattern_id": "p1",
      "name": "Fused Multi-Head Attention",
      "subgraph_description": "QKV projection + scaled dot-product attention + output projection",
      "operations": ["c_attn", "attention computation", "c_proj"],
      "optimization_rule": "FMHA",
      "data_type": "fp16",
      "reuse_info": {
        "strategy": "REUSE | ADAPT | NEW",
        "existing_pattern_path": "path/to/pattern (if REUSE or ADAPT)",
        "adaptations": ["list of modifications (if ADAPT)"]
      },
      "supporting_examples": [
        "<CUTLASS_ROOT>/examples/88_hopper_fmha"
      ],
      "priority": "high",
      "input_shapes": {
        "x": "[batch_size, seq_len, n_embd]",
        "n_head": 8,
        "n_embd": 768
      },
      "expected_challenges": [],
      "rationale": "FP16 halves memory bandwidth vs FP32 and is natively supported by Hopper tensor cores. Attention scores are sensitive to precision, but FP16 is sufficient for inference workloads where QKV values are bounded after layer normalization."
    }
  ],
  "prioritized_patterns": ["p1", "p2", ...]
}
```

## Additional Instructions

1. **Be Specific**: Clearly identify which operations in the model correspond to which CUTLASS patterns
2. **Consider Dependencies**: Note any dependencies between patterns (e.g., one pattern must be implemented before another)
3. **Architecture Specificity**: Tailor your suggestions to the target GPU architecture
4. **Feasibility Assessment**: If a pattern seems infeasible or low-benefit, explain why

## Error Handling

If you cannot find exact matches in the examples index:
- Look for combinations of examples that can be composed
- Consider adapting examples from similar operations
- Note any gaps in the examples index that would need to be filled

If you need to create temporary files, add them to:
<working_directory>/agent_work/tmp