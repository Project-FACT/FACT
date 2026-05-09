#!/bin/bash
# Step 1: Check Environment
# Verifies that all required dependencies and GPU resources are available

set -e  # Exit on error

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$_SCRIPT_DIR/config.sh"

echo "=========================================="
echo "Step 1: Environment Check"
echo "=========================================="
echo ""

# Check if we're on a GPU node
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "⚠️  WARNING: CUDA_VISIBLE_DEVICES not set"
    echo "   Trying to detect GPU..."
fi

# Check for CUDA
if ! command -v nvcc &> /dev/null; then
    echo "✗ ERROR: nvcc (CUDA compiler) not found"
    echo "   Please ensure CUDA is installed and in PATH"
    exit 1
else
    echo "✓ nvcc found: $(nvcc --version | head -n 1)"
fi

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "✗ ERROR: python3 not found"
    exit 1
else
    echo "✓ python3 found: $(python3 --version)"
fi

# Check for PyTorch with CUDA support
echo ""
echo "Checking PyTorch installation..."
python3 << 'EOF'
import sys
try:
    import torch
    print(f"✓ PyTorch version: {torch.__version__}")
    
    if not torch.cuda.is_available():
        print("✗ ERROR: PyTorch CUDA not available")
        sys.exit(1)
    
    print(f"✓ CUDA available: Yes")
    print(f"✓ CUDA version: {torch.version.cuda}")
    print(f"✓ GPU count: {torch.cuda.device_count()}")
    print(f"✓ Current GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ Compute capability: {torch.cuda.get_device_capability(0)}")
    
    # Check compute capability is at least 8.0 (for A100)
    major, minor = torch.cuda.get_device_capability(0)
    if major < 8:
        print(f"⚠️  WARNING: Compute capability {major}.{minor} < 8.0")
        print("   This kernel targets SM80 (A100) and may not work on this GPU")
    
except ImportError as e:
    print(f"✗ ERROR: Failed to import PyTorch: {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ ERROR: PyTorch check failed: {e}")
    sys.exit(1)
EOF

if [ $? -ne 0 ]; then
    echo "✗ PyTorch check failed"
    exit 1
fi

echo ""
echo "Repo / build paths (from test_scripts/config.sh):"
echo "  REPO_ROOT=$REPO_ROOT"
echo "  FACT_ROOT=$FACT_ROOT"
echo "  WHOLE_MODEL=$WHOLE_MODEL"
echo "  CUTLASS_ROOT=${CUTLASS_ROOT:-<unset>}"

echo ""
echo "Checking pattern directories..."
fmha_dir="$FACT_ROOT/pattern_table/fmha/fp32/sm80/fused_multi_head_attention"
mlp_dir="$FACT_ROOT/pattern_table/gemm/tf32/sm80/mlp_gemm_fusion_gelu"

if [ ! -d "$fmha_dir" ]; then
    echo "✗ ERROR: FMHA pattern directory not found: $fmha_dir"
    exit 1
else
    echo "✓ FMHA pattern directory found"
fi

if [ ! -d "$mlp_dir" ]; then
    echo "✗ ERROR: MLP pattern directory not found: $mlp_dir"
    exit 1
else
    echo "✓ MLP pattern directory found"
fi

echo ""
echo "Checking required files..."
required_files=(
    "$fmha_dir/Model.py"
    "$fmha_dir/ModelNew.py"
    "$fmha_dir/cutlass_kernels/fmha_launch.cu"
    "$fmha_dir/cutlass_kernels/main.cpp"
    "$mlp_dir/Model.py"
    "$mlp_dir/ModelNew.py"
    "$mlp_dir/cutlass_kernels/mlp_gemm_fusion.cu"
    "$mlp_dir/cutlass_kernels/main.cpp"
)

for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "✗ ERROR: Required file not found: $file"
        exit 1
    fi
done
echo "✓ All required files found"

echo ""
echo "Checking kernelbench (PYTHONPATH)..."
python3 << 'EOF'
import os
import sys

try:
    import kernelbench

    print("✓ kernelbench import OK:", kernelbench.__file__)
except ImportError as e:
    print("✗ ERROR: cannot import kernelbench:", e)
    print("  REPO_ROOT=", os.environ.get("REPO_ROOT"))
    sys.exit(1)
EOF

echo ""
echo "Checking CUTLASS headers..."
if [[ -z "${CUTLASS_ROOT:-}" ]]; then
    echo "✗ ERROR: CUTLASS_ROOT is not set and no default cutlass/ tree found under REPO_ROOT"
    exit 1
fi
if [[ ! -d "$CUTLASS_ROOT/include/cutlass" ]]; then
    echo "✗ ERROR: CUTLASS_ROOT=$CUTLASS_ROOT is missing include/cutlass/"
    exit 1
fi
echo "✓ CUTLASS_ROOT=$CUTLASS_ROOT"

echo ""
echo "=========================================="
echo "✓ Step 1 COMPLETED: Environment checks passed"
echo "=========================================="
echo ""
echo "Next step: Run './2_compile.sh' to compile extensions"