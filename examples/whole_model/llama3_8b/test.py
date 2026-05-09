#!/usr/bin/env python3
"""
Test script for composed Llama 3 8B model with FMHA and SwiGLU MLP patterns.

Benchmark modes:
  - baseline: PyTorch only (no patterns)
  - fmha_only: FMHA pattern enabled
  - swiglu_only: SwiGLU pattern enabled
  - full: Both patterns enabled
  - all: Run all modes
"""
import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path


def _ensure_cutlass_agent_runtime_paths() -> None:
    """Match ModelNew.py: kernelbench on sys.path and CUTLASS_ROOT for JIT."""
    # This file is at: FACT/examples/whole_model/llama3_8b/test.py
    _THIS_DIR = Path(__file__).resolve().parent
    _REPO_ROOT = _THIS_DIR.parent.parent.parent

    kb_src = _REPO_ROOT / "kernelbench_cutlass_module" / "src"
    if kb_src.is_dir():
        s = str(kb_src)
        if s not in sys.path:
            sys.path.insert(0, s)

    if not os.environ.get("CUTLASS_ROOT", "").strip():
        cutlass = _REPO_ROOT / "cutlass"
        if (cutlass / "include" / "cutlass").is_dir():
            os.environ["CUTLASS_ROOT"] = str(cutlass)


_ensure_cutlass_agent_runtime_paths()

import torch
import torch.nn as nn

# Path setup
# FACT/examples/whole_model/llama3_8b/test.py -> FACT root = parent^3
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent.parent

# Load baseline model from target
_baseline_path = _REPO_ROOT / "KernelBench" / "KernelBench" / "level3" / "51_Llama3_8B_Block.py"
_baseline_spec = importlib.util.spec_from_file_location(
    "llama3_baseline_kernelbench", _baseline_path
)
if _baseline_spec is None or _baseline_spec.loader is None:
    raise ImportError(f"Cannot load baseline from {_baseline_path}")
_baseline_mod = importlib.util.module_from_spec(_baseline_spec)
sys.modules["llama3_baseline_kernelbench"] = _baseline_mod
_baseline_spec.loader.exec_module(_baseline_mod)
BaseModel = _baseline_mod.Model
get_inputs = _baseline_mod.get_inputs
get_init_inputs = _baseline_mod.get_init_inputs

# Load composed ModelNew.py
_composed_path = _THIS_DIR / "ModelNew.py"
_composed_spec = importlib.util.spec_from_file_location(
    "llama3_composed_modelnew", _composed_path
)
if _composed_spec is None or _composed_spec.loader is None:
    raise ImportError(f"Cannot load composed model from {_composed_path}")
_composed_mod = importlib.util.module_from_spec(_composed_spec)
sys.modules["llama3_composed_modelnew"] = _composed_mod
_composed_spec.loader.exec_module(_composed_mod)
OptimizedModel = _composed_mod.Model

# Set verbose output
os.environ['KERNELBENCH_CUTLASS_VERBOSE'] = '1'


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_model(enable_fmha=False, enable_swiglu=False):
    """Create model with specified pattern configuration."""
    init_args = get_init_inputs()
    model = OptimizedModel(*init_args, enable_fmha=enable_fmha, enable_swiglu=enable_swiglu)
    return model.cuda().eval()


def copy_weights_from_baseline(target_model):
    """Copy weights from baseline to target model."""
    init_args = get_init_inputs()
    base_model = BaseModel(*init_args).cuda().eval()

    with torch.no_grad():
        # Copy RMSNorm weights
        target_model.decoder_layer.input_layernorm.weight.copy_(
            base_model.decoder_layer.input_layernorm.weight)
        target_model.decoder_layer.post_attention_layernorm.weight.copy_(
            base_model.decoder_layer.post_attention_layernorm.weight)

        # Copy attention weights
        target_model.decoder_layer.self_attn.q_proj.weight.copy_(
            base_model.decoder_layer.self_attn.q_proj.weight)
        target_model.decoder_layer.self_attn.k_proj.weight.copy_(
            base_model.decoder_layer.self_attn.k_proj.weight)
        target_model.decoder_layer.self_attn.v_proj.weight.copy_(
            base_model.decoder_layer.self_attn.v_proj.weight)
        target_model.decoder_layer.self_attn.o_proj.weight.copy_(
            base_model.decoder_layer.self_attn.o_proj.weight)

        # Copy MLP weights
        target_model.decoder_layer.mlp.gate_proj.weight.copy_(
            base_model.decoder_layer.mlp.gate_proj.weight)
        target_model.decoder_layer.mlp.up_proj.weight.copy_(
            base_model.decoder_layer.mlp.up_proj.weight)
        target_model.decoder_layer.mlp.down_proj.weight.copy_(
            base_model.decoder_layer.mlp.down_proj.weight)

    return base_model


def test_correctness(base_model, opt_model, test_sizes, tolerance=1e-2):
    """Test correctness across different input sizes."""
    print(f"\n{'='*60}")
    print("CORRECTNESS TEST")
    print(f"{'='*60}")
    print(f"Tolerance: {tolerance}")

    status = opt_model.get_pattern_status()
    print(f"\nPattern Status:")
    print(f"  FMHA available: {status['fmha_available']}, enabled: {status['fmha_enabled']}")
    print(f"  SwiGLU available: {status['swiglu_available']}, enabled: {status['swiglu_enabled']}")

    all_passed = True

    for i, (bsz, seq, emb) in enumerate(test_sizes):
        print(f"\n--- Test {i+1}: batch={bsz}, seq_len={seq}, emb={emb} ---")

        x = torch.randn(bsz, seq, emb, device='cuda')

        with torch.no_grad():
            base_out = base_model(x)
            opt_out = opt_model(x)

        max_diff = torch.max(torch.abs(base_out - opt_out)).item()
        rel_diff = max_diff / (torch.max(torch.abs(base_out)).item() + 1e-8)

        passed = max_diff < tolerance
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  Max diff: {max_diff:.2e}, Relative diff: {rel_diff:.2e} {status}")

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


def benchmark_model(model, input_size, warmup=5, trials=20, mode_name=""):
    """Benchmark a single model."""
    print(f"\n--- Benchmarking: {mode_name} ---", flush=True)

    bsz, seq, emb = input_size
    x = torch.randn(bsz, seq, emb, device='cuda')

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(x)
    torch.cuda.synchronize()

    # Timed runs
    times = []
    for _ in range(trials):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(x)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    avg_time = sum(times) / len(times) * 1000  # ms
    return avg_time


def run_multi_mode_benchmark(input_size, warmup=5, trials=20):
    """Run all 4 benchmark configurations."""
    print(f"\n{'='*60}")
    print("PERFORMANCE BENCHMARK - MULTI-MODE")
    print(f"{'='*60}")
    print(f"Input size: {input_size}")
    print(f"Warmup: {warmup} iterations, Timed: {trials} iterations\n")

    results = {}

    # Mode 1: Baseline (PyTorch only)
    print("Mode 1: Baseline (PyTorch only)...", flush=True)
    model_baseline = create_model(enable_fmha=False, enable_swiglu=False)
    _ = copy_weights_from_baseline(model_baseline)
    results['baseline'] = benchmark_model(model_baseline, input_size, warmup, trials, "Baseline")

    # Mode 2: FMHA-only
    print("Mode 2: FMHA-only...", flush=True)
    model_fmha = create_model(enable_fmha=True, enable_swiglu=False)
    _ = copy_weights_from_baseline(model_fmha)
    results['fmha_only'] = benchmark_model(model_fmha, input_size, warmup, trials, "FMHA-only")

    # Mode 3: SwiGLU-only
    print("Mode 3: SwiGLU-only...", flush=True)
    model_swiglu = create_model(enable_fmha=False, enable_swiglu=True)
    _ = copy_weights_from_baseline(model_swiglu)
    results['swiglu_only'] = benchmark_model(model_swiglu, input_size, warmup, trials, "SwiGLU-only")

    # Mode 4: Full (both patterns)
    print("Mode 4: Full (FMHA + SwiGLU)...", flush=True)
    model_full = create_model(enable_fmha=True, enable_swiglu=True)
    _ = copy_weights_from_baseline(model_full)
    results['full'] = benchmark_model(model_full, input_size, warmup, trials, "Full")

    # Print summary table
    print(f"\n{'='*60}")
    print("BENCHMARK RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Configuration':<20} {'Time (ms)':<12} {'Speedup':<10}")
    print("-" * 60)

    baseline_time = results['baseline']
    configs = [
        ('Baseline (PyTorch)', 'baseline', 1.0),
        ('FMHA-only', 'fmha_only', baseline_time / results['fmha_only']),
        ('SwiGLU-only', 'swiglu_only', baseline_time / results['swiglu_only']),
        ('Full (both)', 'full', baseline_time / results['full']),
    ]

    for name, key, speedup in configs:
        print(f"{name:<20} {results[key]:>10.3f}   {speedup:>6.2f}x")

    print(f"{'='*60}\n")

    # Calculate FLOPS (simplified estimate)
    bsz, seq, emb = input_size
    # Attention FLOPS: 2 * B * S^2 * H * d (for Q@K and attn@V)
    # MLP FLOPS: 2 * B * S * d * h + 2 * B * S * h * d (where h = intermediate_dim)
    n_heads = 32
    head_dim = 128
    intermediate_dim = 14336

    attn_flops = 2 * bsz * seq * seq * n_heads * head_dim
    mlp_flops = 2 * bsz * seq * emb * intermediate_dim * 2
    total_flops = attn_flops + mlp_flops
    gflops = total_flops / 1e9

    full_throughput = gflops / (results['full'] / 1000)
    print(f"Estimated throughput (full): {full_throughput:.1f} GFLOPS")

    return results


def main():
    """Main test function."""
    print("=== COMPOSED LLAMA 3 8B MODEL TEST ===", flush=True)

    parser = argparse.ArgumentParser(description="Composed Llama 3 8B Pattern Test")
    parser.add_argument("--correctness-only", action="store_true", help="Only test correctness")
    parser.add_argument("--benchmark-mode", type=str,
                        choices=["baseline", "fmha_only", "swiglu_only", "full", "all"],
                        default="all", help="Benchmark mode")
    parser.add_argument("--tolerance", type=float, default=1e-2, help="Correctness tolerance")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--trials", type=int, default=20, help="Benchmark trials")

    args = parser.parse_args()

    # Set seed
    set_seed(42)
    print("✓ Seed set", flush=True)

    # Check CUDA
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available")
        return 1

    print(f"CUDA Device: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Compute Capability: {torch.cuda.get_device_capability(0)}", flush=True)

    # Correctness test
    if not args.benchmark_mode or args.benchmark_mode == "all":
        test_sizes = [
            (2, 128, 4096),
            (4, 512, 4096),
            (8, 1024, 4096),
            (16, 2048, 4096),
        ]

        # Test with both patterns enabled for correctness
        opt_model = create_model(enable_fmha=True, enable_swiglu=True)
        base_model = copy_weights_from_baseline(opt_model)

        passed = test_correctness(base_model, opt_model, test_sizes, args.tolerance)
        if args.correctness_only:
            return 0 if passed else 1

    # Benchmark
    if not args.correctness_only:
        try:
            if args.benchmark_mode == "all":
                run_multi_mode_benchmark((16, 2048, 4096), args.warmup, args.trials)
            else:
                # Single mode benchmark
                mode_map = {
                    "baseline": (False, False),
                    "fmha_only": (True, False),
                    "swiglu_only": (False, True),
                    "full": (True, True),
                }
                enable_fmha, enable_swiglu = mode_map[args.benchmark_mode]

                model = create_model(enable_fmha, enable_swiglu)
                _ = copy_weights_from_baseline(model)

                time_ms = benchmark_model(
                    model, (16, 2048, 4096), args.warmup, args.trials,
                    f"Mode: {args.benchmark_mode}"
                )
                print(f"\nResult: {time_ms:.3f} ms")

        except Exception as e:
            print(f"ERROR during benchmark: {e}")
            import traceback
            traceback.print_exc()
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
