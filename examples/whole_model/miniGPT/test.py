#!/usr/bin/env python3
"""
Test script for composed MiniGPTBlock model with p1 (FMHA) and p2 (MLP GEMM Fusion) patterns.
Compares baseline PyTorch model against optimized ModelNew.
"""

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path


def _ensure_cutlass_agent_runtime_paths() -> None:
    """Match ModelNew.py: kernelbench on sys.path and CUTLASS_ROOT for JIT."""
    # FACT/examples/whole_model/miniGPT/<this_file> -> FACT root = parent^3
    repo_root = Path(__file__).resolve().parent.parent.parent
    kb_src = repo_root / "kernelbench_cutlass_module" / "src"
    if kb_src.is_dir():
        s = str(kb_src)
        if s not in sys.path:
            sys.path.insert(0, s)
    if not os.environ.get("CUTLASS_ROOT", "").strip():
        cutlass = repo_root / "cutlass"
        if (cutlass / "include" / "cutlass").is_dir():
            os.environ["CUTLASS_ROOT"] = str(cutlass)


_ensure_cutlass_agent_runtime_paths()

import torch
import torch.nn as nn

# FACT/examples/whole_model/miniGPT/test.py -> FACT root = parent^3
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent.parent

_baseline_path = _REPO_ROOT / "KernelBench" / "KernelBench" / "level3" / "44_MiniGPTBlock.py"
_baseline_spec = importlib.util.spec_from_file_location(
    "whole_model_kernelbench_minigpt_baseline", _baseline_path
)
if _baseline_spec is None or _baseline_spec.loader is None:
    raise ImportError(f"Cannot load baseline from {_baseline_path}")
_baseline_mod = importlib.util.module_from_spec(_baseline_spec)
sys.modules["whole_model_kernelbench_minigpt_baseline"] = _baseline_mod
_baseline_spec.loader.exec_module(_baseline_mod)
BaseModel = _baseline_mod.Model
get_inputs = _baseline_mod.get_inputs
get_init_inputs = _baseline_mod.get_init_inputs

# Load composed whole_model/ModelNew.py explicitly (pattern dirs also contain ModelNew.py).
_composed_path = _THIS_DIR / "ModelNew.py"
_composed_spec = importlib.util.spec_from_file_location(
    "whole_model_composed_modelnew", _composed_path
)
if _composed_spec is None or _composed_spec.loader is None:
    raise ImportError(f"Cannot load composed model from {_composed_path}")
_composed_mod = importlib.util.module_from_spec(_composed_spec)
sys.modules["whole_model_composed_modelnew"] = _composed_mod
_composed_spec.loader.exec_module(_composed_mod)
OptimizedModel = _composed_mod.Model

# Set environment variable for verbose output
os.environ['KERNELBENCH_CUTLASS_VERBOSE'] = '1'

def set_seed(seed=42):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def create_models():
    """Create baseline and optimized models."""
    init_args = get_init_inputs()
    base_model = BaseModel(*init_args).cuda().eval()
    opt_model = OptimizedModel(*init_args).cuda().eval()
    
    # Copy weights from base to optimized model
    with torch.no_grad():
        # Copy attention weights
        opt_model.attn.c_attn.weight.copy_(base_model.attn.c_attn.weight)
        opt_model.attn.c_attn.bias.copy_(base_model.attn.c_attn.bias)
        opt_model.attn.c_proj.weight.copy_(base_model.attn.c_proj.weight)
        opt_model.attn.c_proj.bias.copy_(base_model.attn.c_proj.bias)
        
        # Copy MLP weights
        opt_model.mlp.c_fc.weight.copy_(base_model.mlp.c_fc.weight)
        opt_model.mlp.c_fc.bias.copy_(base_model.mlp.c_fc.bias)
        opt_model.mlp.c_proj.weight.copy_(base_model.mlp.c_proj.weight)
        opt_model.mlp.c_proj.bias.copy_(base_model.mlp.c_proj.bias)
        
        # Copy LayerNorm weights
        opt_model.ln_1.weight.copy_(base_model.ln_1.weight)
        opt_model.ln_1.bias.copy_(base_model.ln_1.bias)
        opt_model.ln_2.weight.copy_(base_model.ln_2.weight)
        opt_model.ln_2.bias.copy_(base_model.ln_2.bias)
    
    return base_model, opt_model

def test_correctness(base_model, opt_model, test_sizes, tolerance=1e-3):
    """Test correctness across different input sizes."""
    print(f"\n{'='*60}")
    print("CORRECTNESS TEST")
    print(f"{'='*60}")
    print(f"Tolerance: {tolerance} (FP16 precision ≈1e-3)")
    
    # Print pattern status
    status = opt_model.get_pattern_status()
    print(f"\nPattern Status:")
    print(f"  FMHA available: {status['fmha_available']}")
    print(f"  MLP available: {status['mlp_available']}")
    print(f"  FMHA config: queries_per_block={status['fmha_config']['queries_per_block']}, "
          f"keys_per_block={status['fmha_config']['keys_per_block']}, "
          f"aligned={status['fmha_config']['aligned']}")
    print(f"  MLP config: tile={status['mlp_config']['tile']}, "
          f"warp={status['mlp_config']['warp']}, "
          f"stages={status['mlp_config']['stages']}")
    
    all_passed = True
    
    for i, (bsz, seq, emb) in enumerate(test_sizes):
        print(f"\n--- Test {i+1}: batch={bsz}, seq_len={seq}, emb={emb} ---")
        
        # Create input
        x = torch.randn(bsz, seq, emb, device='cuda')
        
        # Forward pass
        with torch.no_grad():
            base_out = base_model(x)
            opt_out = opt_model(x)
        
        # Compare outputs
        max_diff = torch.max(torch.abs(base_out - opt_out)).item()
        rel_diff = max_diff / (torch.max(torch.abs(base_out)).item() + 1e-8)
        
        passed = max_diff < tolerance
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  Max diff: {max_diff:.2e}, Relative diff: {rel_diff:.2e} {status}")
        
        # Show sample values for direct comparison
        print(f"  Sample value comparison (first 3 elements of first batch):")
        for idx in range(min(3, seq * emb)):
            base_val = base_out[0, 0, idx].item()
            opt_val = opt_out[0, 0, idx].item()
            diff = abs(base_val - opt_val)
            print(f"    [{idx}] Baseline: {base_val:.6f}, Optimized: {opt_val:.6f}, diff: {diff:.2e}")
        
        if not passed:
            all_passed = False
            print(f"    Baseline stats: mean={base_out.mean():.4e}, std={base_out.std():.4e}")
            print(f"    Optimized stats: mean={opt_out.mean():.4e}, std={opt_out.std():.4e}")
    
    print(f"\n{'='*60}")
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print(f"{'='*60}\n")
    
    return all_passed

def benchmark(base_model, opt_model, input_size, warmup=5, trials=20):
    """Benchmark both models."""
    import sys
    print(f"\n{'='*60}", flush=True)
    print("PERFORMANCE BENCHMARK", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Input size: {input_size}", flush=True)
    print(f"Warmup: {warmup} iterations, Timed: {trials} iterations", flush=True)
    sys.stdout.flush()
    
    bsz, seq, emb = input_size
    x = torch.randn(bsz, seq, emb, device='cuda')
    
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = base_model(x)
            _ = opt_model(x)
    torch.cuda.synchronize()
    
    # Benchmark baseline
    times_base = []
    for _ in range(trials):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = base_model(x)
        torch.cuda.synchronize()
        times_base.append(time.perf_counter() - start)
    
    # Benchmark optimized
    times_opt = []
    for _ in range(trials):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = opt_model(x)
        torch.cuda.synchronize()
        times_opt.append(time.perf_counter() - start)
    
    # Statistics
    base_time = sum(times_base) / len(times_base) * 1000  # ms
    opt_time = sum(times_opt) / len(times_opt) * 1000  # ms
    speedup = base_time / opt_time
    
    print(f"\nResults:")
    print(f"  Baseline: {base_time:.3f} ms")
    print(f"  Optimized: {opt_time:.3f} ms")
    print(f"  Speedup: {speedup:.2f}x")
    
    # Calculate FLOPS (simplified)
    flops = (3 + 1) * bsz * seq * emb * emb + 2 * bsz * 8 * seq * seq * (emb // 8)
    gflops = flops / 1e9
    throughput = gflops / (opt_time / 1000)
    
    print(f"  Throughput: {throughput:.1f} GFLOPS")
    
    # A100 theoretical peak
    theoretical_peak = 19.5e3  # GFLOPS for FP32 on A100
    efficiency = (throughput / theoretical_peak) * 100
    print(f"  Efficiency: {efficiency:.1f}% of theoretical peak (FP32)")
    
    print(f"{'='*60}\n")
    
    return base_time, opt_time, speedup

def main():
    """Main test function."""
    print("=== COMPOSED MODEL TEST STARTED ===", flush=True)
    sys.stdout.flush()
    
    parser = argparse.ArgumentParser(description="Composed Model Pattern Test")
    parser.add_argument("--correctness-only", action="store_true", help="Only test correctness")
    parser.add_argument("--benchmark-only", action="store_true", help="Only run benchmark")
    parser.add_argument("--tolerance", type=float, default=1e-3, help="Correctness tolerance")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--trials", type=int, default=20, help="Benchmark trials")
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(42)
    print("✓ Seed set", flush=True)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available")
        return 1
    
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Compute Capability: {torch.cuda.get_device_capability(0)}", flush=True)
    
    # Create models
    print("\nLoading models...")
    base_model, opt_model = create_models()
    print("✓ Models loaded and weights synchronized")
    
    # Print pattern status
    status = opt_model.get_pattern_status()
    print(f"\nPattern Status:")
    print(f"  FMHA available: {status['fmha_available']}")
    print(f"  MLP available: {status['mlp_available']}")
    
    if not status['fmha_available'] and not status['mlp_available']:
        print("\nWARNING: Neither FMHA nor MLP extensions are available!")
        print("This may indicate compilation issues.")
    
    # Correctness test
    if not args.benchmark_only:
        test_sizes = [
            (2, 64, 768),    # Small
            (4, 128, 768),   # Medium
            (16, 256, 768),  # Large
        ]
        
        # Add full size if not correctness-only
        if not args.correctness_only:
            test_sizes.append((128, 512, 768))  # Full problem size
        
        passed = test_correctness(base_model, opt_model, test_sizes, args.tolerance)
        if not passed and args.correctness_only:
            return 1
    
    # Benchmark
    if not args.correctness_only:
        try:
            benchmark(base_model, opt_model, (128, 512, 768), args.warmup, args.trials)
        except Exception as e:
            print(f"ERROR during benchmark: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())