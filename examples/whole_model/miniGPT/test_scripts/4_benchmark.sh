#!/bin/bash
# Step 4: Benchmark Performance
# Measures the performance improvement of the optimized model

set -e  # Exit on error

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$_SCRIPT_DIR/config.sh"

echo "=========================================="
echo "Step 4: Performance Benchmarking"
echo "=========================================="
echo ""

cd "$WHOLE_MODEL"

echo "Running performance benchmark..."
echo "Note: This tests the full problem size (batch=128, seq_len=512, emb=768)"
echo "      Warmup: 5 iterations, Timed: 20 iterations"
echo ""

python3 test.py --benchmark-only --warmup 5 --trials 20

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Step 4 COMPLETED: Benchmark completed"
    echo "=========================================="
    echo ""
    echo "Expected Performance:"
    echo "  - FMHA speedup: ~3.62x"
    echo "  - MLP speedup: ~2.85x"
    echo "  - Overall expected speedup: ~3.47x"
    echo ""
    echo "See above output for actual measured speedup"
else
    echo ""
    echo "=========================================="
    echo "✗ Step 4 FAILED: Benchmark failed"
    echo "=========================================="
    exit 1
fi