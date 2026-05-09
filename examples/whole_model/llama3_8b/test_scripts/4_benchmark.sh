#!/bin/bash
# Step 4: Multi-Mode Performance Benchmarking
# Measures performance for all 4 configurations

set -e  # Exit on error

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$_SCRIPT_DIR/config.sh"

echo "=========================================="
echo "Step 4: Multi-Mode Performance Benchmark"
echo "=========================================="
echo ""

cd "$WHOLE_MODEL"

echo "Running multi-mode performance benchmark..."
echo "This will test 4 configurations:"
echo "  1. Baseline (PyTorch only)"
echo "  2. FMHA-only"
echo "  3. SwiGLU-only"
echo "  4. Full (FMHA + SwiGLU)"
echo ""
echo "Problem size: batch=16, seq_len=2048, emb=4096"
echo "Warmup: 5 iterations, Timed: 20 iterations"
echo ""

python3 test.py --benchmark-mode all --warmup 5 --trials 20

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Step 4 COMPLETED: Benchmark completed"
    echo "=========================================="
    echo ""
    echo "Expected Performance:"
    echo "  - FMHA speedup: ~5.29x (attention only)"
    echo "  - SwiGLU speedup: ~8.42x (MLP only)"
    echo "  - Overall expected speedup: ~2.98x (end-to-end)"
    echo ""
    echo "See above output for actual measured speedup"
else
    echo ""
    echo "=========================================="
    echo "✗ Step 4 FAILED: Benchmark failed"
    echo "=========================================="
    exit 1
fi
