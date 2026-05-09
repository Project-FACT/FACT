#!/bin/bash
#SBATCH --job-name=kb_ncu_profile
#SBATCH --account=pearl
#SBATCH --partition=a100_normal_q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:15:00
#SBATCH --output=results/step5_ncu_%j.out
#SBATCH --error=results/step5_ncu_%j.err

# Step 5: NCU Profiling
# Builds on step4 - adds NCU profiling metrics

set -e  # Exit on error

echo "=================================================="
echo "Step 5: NCU Profiling"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo ""

# Set paths (same as step4)
WORK_DIR="<FACT_ROOT>/kernelbench_cutlass_module"

cd $WORK_DIR

# Load modules
module load CUDA/12.6.0
module load nvidia-cutlass/3.8.0.0

# Set up environment
export PYTHONPATH="$WORK_DIR/src:$PYTHONPATH"
export CUTLASS_ROOT="/apps/arch/software/nvidia-cutlass/3.8.0.0-gfbf-2024a-CUDA-12.6.0/lib/python3.12/site-packages/cutlass_library/source"

echo "Setup:"
echo "  Working dir: $WORK_DIR"
echo "  Modules: CUDA/12.6.0, nvidia-cutlass/3.8.0.0"
echo "  CUTLASS_ROOT: $CUTLASS_ROOT"
echo ""

# Create profiling results directory
mkdir -p results/ncu_reports

echo "=================================================="
echo "NCU Profiling Configuration"
echo "=================================================="
echo "Matrix size: 4096x4096"
echo "Iterations: 100 (for good profiling data)"
echo ""

# Test with PyTorch first (baseline)
echo "=================================================="
echo "1. Profiling PyTorch (cuBLAS TF32)"
echo "=================================================="

ncu --set full \
    --target-processes all \
    -o results/ncu_reports/pytorch_ncu \
    python work/test.py --compile-only

if [ -f "results/ncu_reports/pytorch_ncu.ncu-rep" ]; then
    echo "✓ PyTorch NCU profiling complete"
    ncu --export results/ncu_reports/pytorch_ncu.csv \
        results/ncu_reports/pytorch_ncu.ncu-rep
    echo "  Report: results/ncu_reports/pytorch_ncu.ncu-rep"
    echo "  CSV:    results/ncu_reports/pytorch_ncu.csv"
else
    echo "❌ PyTorch NCU profiling failed"
fi
echo ""

# Profile CUTLASS
echo "=================================================="
echo "2. Profiling CUTLASS (TF32 Tensor Cores)"
echo "=================================================="

ncu --set full \
    --target-processes all \
    -o results/ncu_reports/cutlass_ncu \
    python work/test.py --compile-only

if [ -f "results/ncu_reports/cutlass_ncu.ncu-rep" ]; then
    echo "✓ CUTLASS NCU profiling complete"
    ncu --export results/ncu_reports/cutlass_ncu.csv \
        results/ncu_reports/cutlass_ncu.ncu-rep
    echo "  Report: results/ncu_reports/cutlass_ncu.ncu-rep"
    echo "  CSV:    results/ncu_reports/cutlass_ncu.csv"
else
    echo "❌ CUTLASS NCU profiling failed"
fi
echo ""

echo "=================================================="
echo "Quick Summary"
echo "=================================================="

if [ -f "results/ncu_reports/pytorch_ncu.ncu-rep" ]; then
    echo ""
    echo "PyTorch NCU Summary:"
    ncu --print summary results/ncu_reports/pytorch_ncu.ncu-rep | head -30
fi

if [ -f "results/ncu_reports/cutlass_ncu.ncu-rep" ]; then
    echo ""
    echo "CUTLASS NCU Summary:"
    ncu --print summary results/ncu_reports/cutlass_ncu.ncu-rep | head -30
fi

echo ""
echo "=================================================="
echo "✓ Step 5 Complete: NCU Profiling"
echo "=================================================="
echo ""
echo "NCU Results:"
echo "  results/ncu_reports/pytorch_ncu.ncu-rep"
echo "  results/ncu_reports/pytorch_ncu.csv"
echo "  results/ncu_reports/cutlass_ncu.ncu-rep"
echo "  results/ncu_reports/cutlass_ncu.csv"
