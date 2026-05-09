#!/bin/bash
#SBATCH --job-name=kb_compile
#SBATCH --account=pearl
#SBATCH --partition=a100_normal_q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:15:00
#SBATCH --output=results/step2_compile_%j.out
#SBATCH --error=results/step2_compile_%j.err

# Step 2: Compile CUTLASS Extension
# This script tests JIT compilation of the CUTLASS GEMM kernel

set -e  # Exit on error

echo "=================================================="
echo "Step 2: Compiling CUTLASS Extension"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo ""

# Set paths
WORK_DIR="<FACT_ROOT>/kernelbench_cutlass_module"

cd $WORK_DIR

# Load CUDA module (needed for nvcc compiler)
module load CUDA/12.6.0
# Load nvidia-cutlass module (provides CUTLASS headers and libraries)
module load nvidia-cutlass/3.8.0.0

# Add src to Python path
export PYTHONPATH="$WORK_DIR/src:$PYTHONPATH"

# Set CUTLASS_ROOT to point to module's CUTLASS headers
# Headers are in: lib/python3.12/site-packages/cutlass_library/source/include/cutlass
export CUTLASS_ROOT="/apps/arch/software/nvidia-cutlass/3.8.0.0-gfbf-2024a-CUDA-12.6.0/lib/python3.12/site-packages/cutlass_library/source"

echo "Working directory: $WORK_DIR"
echo "Modules loaded: CUDA/12.6.0, nvidia-cutlass/3.8.0.0"
echo "CUTLASS_ROOT: $CUTLASS_ROOT"
echo ""

# Check environment
echo "Checking environment..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import torch; print(f'Device: {torch.cuda.get_device_name(0)}')"
echo ""

# Set verbose for compilation
export KERNELBENCH_CUTLASS_VERBOSE=1

echo "=================================================="
echo "Compiling CUTLASS Extension (JIT via PyTorch)..."
echo "=================================================="
echo ""

# Compile test - this will trigger JIT compilation
python work/test.py --compile-only

echo ""
echo "=================================================="
echo "✓ Step 2 Complete: CUTLASS extension compiled"
echo "=================================================="
echo ""
echo "Next step: Submit step3_test_correctness.sh to verify correctness"
echo "Command: sbatch sbatch/step3_test_correctness.sh"
