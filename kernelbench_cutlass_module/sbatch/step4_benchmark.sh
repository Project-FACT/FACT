#!/bin/bash
#SBATCH --job-name=kb_bench
#SBATCH --account=pearl
#SBATCH --partition=a100_normal_q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:30:00
#SBATCH --output=results/step4_benchmark_%j.out
#SBATCH --error=results/step4_benchmark_%j.err

# Step 4: Benchmark Performance
# This script benchmarks ModelNew (CUTLASS) vs Model (PyTorch) performance

set -e  # Exit on error

echo "=================================================="
echo "Step 4: Performance Benchmark"
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

echo "=================================================="
echo "Running Performance Benchmarks"
echo "=================================================="
echo ""

# Run benchmark
python work/test.py --benchmark-only

echo ""
echo "=================================================="
echo "✓ Step 4 Complete: Benchmark complete"
echo "=================================================="
echo ""
echo "All tests finished! Check results/ directory for output files."
