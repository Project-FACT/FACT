#!/bin/bash
# Step 2: Compile Extensions
# Compiles both FMHA and MLP CUTLASS extensions

set -e  # Exit on error

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$_SCRIPT_DIR/config.sh"

echo "=========================================="
echo "Step 2: Compile Extensions"
echo "=========================================="
echo ""

# Set verbosity
export KERNELBENCH_CUTLASS_VERBOSE=1

echo "Note: Extensions will be compiled on-demand when ModelNew.py is imported"
echo "      This happens automatically during model initialization."
echo ""

# Test compilation by importing ModelNew
echo "Testing extension compilation by importing ModelNew..."
export FACT_ROOT WHOLE_MODEL REPO_ROOT
echo "  REPO_ROOT=$REPO_ROOT"
echo "  FACT_ROOT=$FACT_ROOT"
echo "  WHOLE_MODEL=$WHOLE_MODEL"
if [ ! -f "$WHOLE_MODEL/ModelNew.py" ]; then
    echo "✗ ERROR: Composed ModelNew.py not found at $WHOLE_MODEL/ModelNew.py"
    exit 1
fi

python3 << 'EOF'
import importlib.util
import os
import sys

agent = os.environ["FACT_ROOT"]
whole = os.environ["WHOLE_MODEL"]
fmha_pattern_dir = os.path.join(agent, "pattern_table/fmha/fp32/sm80/fused_multi_head_attention")
mlp_pattern_dir = os.path.join(agent, "pattern_table/gemm/tf32/sm80/mlp_gemm_fusion_gelu")
sys.path.insert(0, fmha_pattern_dir)
sys.path.insert(0, mlp_pattern_dir)
sys.path.insert(0, whole)

print("Importing ModelNew (this will trigger compilation)...")
try:
    composed_path = os.path.join(whole, "ModelNew.py")
    spec = importlib.util.spec_from_file_location("whole_model_compose_compile", composed_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {composed_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["whole_model_compose_compile"] = mod
    spec.loader.exec_module(mod)
    Model = mod.Model
    get_init_inputs = mod.get_init_inputs
    print("✓ ModelNew imported successfully")
    
    # Try to initialize model to ensure extensions are loaded
    init_args = get_init_inputs()
    model = Model(*init_args).cuda().eval()
    
    status = model.get_pattern_status()
    print(f"\nExtension Status:")
    print(f"  FMHA available: {status['fmha_available']}")
    print(f"  MLP available: {status['mlp_available']}")
    
    if status['fmha_available'] and status['mlp_available']:
        print("\n✓ Step 2 COMPLETED: Both extensions compiled successfully")
    elif status['fmha_available']:
        print("\n⚠️  WARNING: Only FMHA extension compiled (MLP failed)")
    elif status['mlp_available']:
        print("\n⚠️  WARNING: Only MLP extension compiled (FMHA failed)")
    else:
        print("\n✗ ERROR: Neither extension compiled successfully")
        sys.exit(1)
        
    print(f"\nUsing configurations:")
    print(f"  FMHA: queries_per_block={status['fmha_config']['queries_per_block']}, "
          f"keys_per_block={status['fmha_config']['keys_per_block']}, "
          f"aligned={status['fmha_config']['aligned']}")
    print(f"  MLP: tile={status['mlp_config']['tile']}, "
          f"warp={status['mlp_config']['warp']}, "
          f"stages={status['mlp_config']['stages']}")
    
except Exception as e:
    print(f"✗ ERROR: Failed to compile extensions: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

if [ $? -ne 0 ]; then
    echo "✗ Compilation failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "Extensions are compiled and ready"
echo "=========================================="
echo ""
echo "Next step: Run './3_test_correctness.sh' to verify correctness"