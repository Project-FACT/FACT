#!/bin/bash
# Run all validation steps for the composed model
# This script executes all 4 test steps in sequence

set -e  # Exit on error

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$_SCRIPT_DIR/config.sh"

echo "=========================================="
echo "Composed Model Validation Pipeline"
echo "Pattern Composition: FMHA + SwiGLU MLP"
echo "Target: Llama 3 8B Single Block"
echo "=========================================="
echo ""
echo "This pipeline will:"
echo "  1. Check environment and dependencies"
echo "  2. Compile CUTLASS extensions"
echo "  3. Test correctness against baseline"
echo "  4. Benchmark performance (multi-mode)"
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
echo "Starting Step 4: Multi-Mode Benchmarking..."
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
echo "  ✓ Performance benchmarked (multi-mode)"
echo ""
echo "Check the output above for detailed results including:"
echo "  - Pattern availability status"
echo "  - Configurations used"
echo "  - Correctness test results"
echo "  - Per-pattern and end-to-end speedup"
echo ""
echo "Expected results:"
echo "  - FMHA speedup: ~5.29x (attention only)"
echo "  - SwiGLU speedup: ~8.42x (MLP only)"
echo "  - Combined end-to-end speedup: ~2.98x"
