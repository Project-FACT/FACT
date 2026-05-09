#!/bin/bash
# Run all validation steps for the composed model
# This script executes all 4 test steps in sequence

set -e  # Exit on error

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$_SCRIPT_DIR/config.sh"

echo "=========================================="
echo "Composed Model Validation Pipeline"
echo "Pattern Composition: p1 (FMHA) + p2 (MLP GEMM Fusion)"
echo "=========================================="
echo ""
echo "This pipeline will:"
echo "  1. Check environment and dependencies"
echo "  2. Compile CUTLASS extensions"
echo "  3. Test correctness against baseline"
echo "  4. Benchmark performance"
echo ""
echo "Press Enter to continue or Ctrl+C to cancel..."
read

# Step 1: Environment Check
echo ""
echo "Starting Step 1: Environment Check..."
echo "----------------------------------------"
./1_check_env.sh

# Step 2: Compile Extensions
echo ""
echo "Starting Step 2: Compile Extensions..."
echo "----------------------------------------"
./2_compile.sh

# Step 3: Correctness Testing
echo ""
echo "Starting Step 3: Correctness Testing..."
echo "----------------------------------------"
./3_test_correctness.sh

# Step 4: Benchmark
echo ""
echo "Starting Step 4: Performance Benchmarking..."
echo "----------------------------------------"
./4_benchmark.sh

# All steps completed successfully
echo ""
echo "=========================================="
echo "✓ ALL VALIDATION STEPS COMPLETED SUCCESSFULLY"
echo "=========================================="
echo ""
echo "Summary:"
echo "  ✓ Environment checked"
echo "  ✓ Extensions compiled"
echo "  ✓ Correctness verified"
echo "  ✓ Performance benchmarked"
echo ""
echo "Check the output above for detailed results including:"
echo "  - Pattern availability status"
echo "  - Configurations used"
echo "  - Correctness test results"
echo "  - Performance speedup"
echo ""
echo "Expected overall speedup: ~3.47x"
echo "  - FMHA contribution: ~3.62x"
echo "  - MLP contribution: ~2.85x"