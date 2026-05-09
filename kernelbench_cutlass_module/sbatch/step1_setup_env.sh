#!/bin/bash
#SBATCH --job-name=kb_setup_env
#SBATCH --account=pearl
#SBATCH --partition=a100_normal_q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:10:00
#SBATCH --output=results/step1_setup_env_%j.out
#SBATCH --error=results/step1_setup_env_%j.err

# Step 1: Setup Environment for KernelBench CUTLASS Test
# This script sets up the directory structure and copies necessary files

set -e  # Exit on error

echo "=================================================="
echo "Step 1: Setting up KernelBench CUTLASS Environment"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo ""

# Set paths
WORK_DIR="<FACT_ROOT>/kernelbench_cutlass_module"
KERNELBENCH_SRC="<FACT_ROOT>/KernelBench/src/kernelbench"
CUTLASS_ROOT="<CUTLASS_ROOT>"

cd $WORK_DIR

echo "Working directory: $WORK_DIR"
echo ""

# Create directory structure
echo "Creating directory structure..."
mkdir -p src/kernelbench/cutlass_cpp
mkdir -p work
mkdir -p results
mkdir -p sbatch
echo "✓ Directories created"
echo ""

# Copy KernelBench utilities
echo "Copying KernelBench utilities..."
cp $KERNELBENCH_SRC/cutlass_cpp/runtime.py src/kernelbench/cutlass_cpp/
cp $KERNELBENCH_SRC/cutlass_cpp/__init__.py src/kernelbench/cutlass_cpp/
# Use minimal __init__.py to avoid heavy dependencies (utils, openai, litellm, etc.)
cat > src/kernelbench/__init__.py << 'EOF'
"""
Minimal kernelbench package for CUTLASS testing.
This is a stripped-down version that only includes CUTLASS runtime support.
"""

# Version info
__version__ = "0.1.0-minimal"

# No imports needed - submodules will be imported directly
__all__ = []
EOF
echo "✓ KernelBench utilities copied (using minimal __init__.py)"
echo ""

# Verify CUTLASS installation
echo "Verifying CUTLASS installation..."
if [ -d "$CUTLASS_ROOT/include/cutlass" ]; then
    echo "✓ CUTLASS_ROOT: $CUTLASS_ROOT"
    echo "✓ CUTLASS headers found"
else
    echo "❌ CUTLASS headers not found at $CUTLASS_ROOT/include/cutlass"
    exit 1
fi
echo ""

# Check Python and PyTorch
echo "Checking Python environment..."
python --version
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
echo ""

# Check CUDA module loading
echo "Checking CUDA module..."
module load CUDA/12.6.0
which nvcc
nvcc --version
echo ""

# Verify test files exist
echo "Checking test files..."
if [ -f "work/Model.py" ]; then
    echo "✓ work/Model.py exists"
else
    echo "❌ work/Model.py not found"
fi

if [ -f "work/ModelNew.py" ]; then
    echo "✓ work/ModelNew.py exists"
else
    echo "❌ work/ModelNew.py not found"
fi

if [ -f "work/test.py" ]; then
    echo "✓ work/test.py exists"
else
    echo "❌ work/test.py not found"
fi
echo ""

echo "=================================================="
echo "✓ Step 1 Complete: Environment setup successful"
echo "=================================================="
echo ""
echo "Next step: Submit step2_compile_cutlass.sh to compile the CUTLASS extension"
echo "Command: sbatch sbatch/step2_compile_cutlass.sh"
