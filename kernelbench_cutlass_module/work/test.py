#!/usr/bin/env python3
"""
Test script for KernelBench CUTLASS setup - Square Matrix Multiplication (Problem 1, Level 1)

This script tests:
1. Environment setup (CUTLASS_ROOT, PYTHONPATH)
2. Module imports
3. Correctness of ModelNew vs Model
4. Performance comparison

Usage:
    python test.py [--compile-only] [--correctness-only] [--benchmark-only]
"""
import argparse
import os
import sys
import time
from typing import List

import torch

# Add src to Python path (for kernelbench modules)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
# Add current directory to path (for ModelNew/Model imports)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set up environment - use nvidia-cutlass module for CUTLASS_ROOT
if 'CUTLASS_ROOT' not in os.environ:
    os.environ['CUTLASS_ROOT'] = '/apps/arch/software/nvidia-cutlass/3.8.0.0-gfbf-2024a-CUDA-12.6.0/lib/python3.12/site-packages/cutlass_library/source'


def check_environment():
    """Check that required environment is set up correctly"""
    print("=" * 60)
    print("Step 0: Environment Check")
    print("=" * 60)

    # Check CUTLASS_ROOT
    cutlass_root = os.environ.get('CUTLASS_ROOT')
    if not cutlass_root:
        print("❌ FAIL: CUTLASS_ROOT not set")
        return False

    cutlass_include = os.path.join(cutlass_root, 'include', 'cutlass')
    if not os.path.isdir(cutlass_include):
        print(f"❌ FAIL: CUTLASS_ROOT={cutlass_root} does not contain include/cutlass/")
        return False

    print(f"✓ CUTLASS_ROOT: {cutlass_root}")
    print(f"✓ CUTLASS headers found: {cutlass_include}")

    # Check CUDA
    if not torch.cuda.is_available():
        print("❌ FAIL: CUDA not available")
        return False

    device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(device)
    compute_capability = torch.cuda.get_device_capability(device)
    print(f"✓ CUDA available: {device_name}")
    print(f"✓ Compute capability: {compute_capability[0]}.{compute_capability[1]}")

    print()
    return True


def test_compile():
    """Test that ModelNew can be compiled"""
    print("=" * 60)
    print("Step 1: Compile CUTLASS Extension")
    print("=" * 60)

    try:
        from ModelNew import ModelNew
        print("✓ ModelNew imported successfully")

        # Create a small test to trigger compilation
        model = ModelNew().cuda()
        print("✓ ModelNew instantiated on GPU")

        # Small test compilation
        A = torch.rand(64, 64, dtype=torch.float32, device='cuda')
        B = torch.rand(64, 64, dtype=torch.float32, device='cuda')
        C = model(A, B)
        print(f"✓ Test compilation successful: output shape {C.shape}")

        print()
        return True, model

    except Exception as e:
        print(f"❌ FAIL: Compilation failed: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False, None


def test_correctness(model_new):
    """Test correctness of ModelNew vs Model"""
    print("=" * 60)
    print("Step 2: Correctness Test")
    print("=" * 60)

    try:
        from Model import Model

        model_baseline = Model().cuda()

        # Test with different sizes
        test_sizes = [512, 1024, 2048]

        for size in test_sizes:
            print(f"\nTesting size {size}x{size}:")

            # Create test data (use same seed for reproducibility)
            torch.manual_seed(42)
            A = torch.rand(size, size, dtype=torch.float32, device='cuda')
            B = torch.rand(size, size, dtype=torch.float32, device='cuda')

            # Run baseline
            with torch.no_grad():
                C_baseline = model_baseline(A, B)

            # Run CUTLASS
            with torch.no_grad():
                C_cutlass = model_new(A, B)

            # Check correctness
            max_diff = torch.max(torch.abs(C_baseline - C_cutlass)).item()
            mean_diff = torch.mean(torch.abs(C_baseline - C_cutlass)).item()
            rel_diff = max_diff / (torch.max(torch.abs(C_baseline)).item() + 1e-6)

            print(f"  Max difference: {max_diff:.6e}")
            print(f"  Mean difference: {mean_diff:.6e}")
            print(f"  Relative difference: {rel_diff:.6e}")

            # Use fp32 tolerance from KernelBench
            tolerance = 1e-4
            if max_diff < tolerance:
                print(f"  ✓ PASS (tolerance: {tolerance:.1e})")
            else:
                print(f"  ❌ FAIL (tolerance: {tolerance:.1e})")
                return False

        print("\n✓ All correctness tests passed")
        print()
        return True

    except Exception as e:
        print(f"❌ FAIL: Correctness test failed: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False


def test_benchmark(model_new):
    """Benchmark performance of ModelNew vs Model"""
    print("=" * 60)
    print("Step 3: Performance Benchmark")
    print("=" * 60)

    try:
        from Model import Model

        model_baseline = Model().cuda()

        # Benchmark sizes (including the problem size N=4096)
        test_sizes = [1024, 2048, 4096]
        num_warmup = 5
        num_trials = 20

        print(f"Benchmark settings: {num_warmup} warmup runs, {num_trials} timed runs")
        print()

        results = []

        for size in test_sizes:
            print(f"Benchmarking size {size}x{size}:")

            # Create test data
            torch.manual_seed(42)
            A = torch.rand(size, size, dtype=torch.float32, device='cuda')
            B = torch.rand(size, size, dtype=torch.float32, device='cuda')

            # Warmup
            for _ in range(num_warmup):
                with torch.no_grad():
                    _ = model_baseline(A, B)
                    _ = model_new(A, B)
            torch.cuda.synchronize()

            # Benchmark baseline
            times_baseline = []
            for _ in range(num_trials):
                torch.cuda.synchronize()
                start = time.perf_counter()
                with torch.no_grad():
                    _ = model_baseline(A, B)
                torch.cuda.synchronize()
                times_baseline.append(time.perf_counter() - start)

            # Benchmark CUTLASS
            times_cutlass = []
            for _ in range(num_trials):
                torch.cuda.synchronize()
                start = time.perf_counter()
                with torch.no_grad():
                    _ = model_new(A, B)
                torch.cuda.synchronize()
                times_cutlass.append(time.perf_counter() - start)

            # Compute statistics
            mean_baseline = sum(times_baseline) / len(times_baseline)
            mean_cutlass = sum(times_cutlass) / len(times_cutlass)
            speedup = mean_baseline / mean_cutlass

            print(f"  Baseline (PyTorch): {mean_baseline*1000:.3f} ms")
            print(f"  CUTLASS:            {mean_cutlass*1000:.3f} ms")
            print(f"  Speedup:            {speedup:.3f}x")

            # Compute GFLOPS
            gflops = (2 * size ** 3) / (mean_cutlass * 1e9)
            print(f"  CUTLASS throughput: {gflops:.2f} GFLOPS")
            print()

            results.append({
                'size': size,
                'baseline_ms': mean_baseline * 1000,
                'cutlass_ms': mean_cutlass * 1000,
                'speedup': speedup,
                'gflops': gflops
            })

        print("✓ Benchmark complete")
        print()
        return True, results

    except Exception as e:
        print(f"❌ FAIL: Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False, None


def main():
    parser = argparse.ArgumentParser(description='Test KernelBench CUTLASS setup')
    parser.add_argument('--compile-only', action='store_true',
                        help='Only test compilation')
    parser.add_argument('--correctness-only', action='store_true',
                        help='Only test correctness (requires compiled extension)')
    parser.add_argument('--benchmark-only', action='store_true',
                        help='Only run benchmark (requires compiled extension)')
    args = parser.parse_args()

    # Step 0: Environment check
    if not check_environment():
        sys.exit(1)

    # Step 1: Compile
    success, model_new = test_compile()
    if not success:
        sys.exit(1)

    if args.compile_only:
        print("✓ Compilation test only - exiting")
        sys.exit(0)

    # Step 2: Correctness
    if not args.benchmark_only:
        if not test_correctness(model_new):
            sys.exit(1)

    # Step 3: Benchmark
    if not args.correctness_only:
        success, results = test_benchmark(model_new)
        if not success:
            sys.exit(1)

    print("=" * 60)
    print("✓ ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
