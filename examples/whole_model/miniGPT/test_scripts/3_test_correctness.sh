#!/bin/bash
# Step 3: Test Correctness
# Verifies that the optimized model produces correct outputs

set -e  # Exit on error

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$_SCRIPT_DIR/config.sh"

echo "=========================================="
echo "Step 3: Correctness Testing"
echo "=========================================="
echo ""

cd "$WHOLE_MODEL"

echo "Running correctness tests..."
echo "Note: This tests multiple input sizes to ensure robustness"
echo ""

python3 test.py --correctness-only --tolerance 1e-3

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Step 3 COMPLETED: All correctness tests passed"
    echo "=========================================="
    echo ""
    echo "Next step: Run './4_benchmark.sh' to measure performance"
else
    echo ""
    echo "=========================================="
    echo "✗ Step 3 FAILED: Correctness tests failed"
    echo "=========================================="
    exit 1
fi