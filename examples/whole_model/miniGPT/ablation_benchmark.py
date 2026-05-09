#!/usr/bin/env python3
"""
Ablation benchmark for MiniGPT - tests all three variants:
1. Baseline (PyTorch)
2. FMHA only (CUTLASS FMHA + PyTorch MLP)
3. MLP only (PyTorch FMHA + CUTLASS MLP)
4. Both (CUTLASS FMHA + CUTLASS MLP)
"""

import importlib.util
import sys
import time
import torch
from pathlib import Path

# Add paths - pattern modules for extensions
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "kernelbench_cutlass_module" / "src"))

# Import whole_model modules explicitly to avoid confusion with pattern modules
import importlib.util

def load_whole_model_module(filename, module_name):
    """Load a ModelNew.py from the whole_model directory."""
    model_path = _THIS_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, model_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod

# Load baseline
_baseline_path = _REPO_ROOT / "KernelBench" / "KernelBench" / "level3" / "44_MiniGPTBlock.py"
_baseline_spec = importlib.util.spec_from_file_location("baseline", _baseline_path)
_baseline_mod = importlib.util.module_from_spec(_baseline_spec)
_baseline_spec.loader.exec_module(_baseline_mod)
BaseModel = _baseline_mod.Model


def create_model(model_class, init_args):
    """Create a model instance."""
    return model_class(*init_args).cuda().eval()


def copy_weights(base_model, target_model):
    """Copy weights from baseline to target model."""
    with torch.no_grad():
        # LayerNorm weights
        target_model.ln_1.weight.copy_(base_model.ln_1.weight)
        target_model.ln_1.bias.copy_(base_model.ln_1.bias)
        target_model.ln_2.weight.copy_(base_model.ln_2.weight)
        target_model.ln_2.bias.copy_(base_model.ln_2.bias)

        # Attention weights
        target_model.attn.c_attn.weight.copy_(base_model.attn.c_attn.weight)
        target_model.attn.c_attn.bias.copy_(base_model.attn.c_attn.bias)
        target_model.attn.c_proj.weight.copy_(base_model.attn.c_proj.weight)
        target_model.attn.c_proj.bias.copy_(base_model.attn.c_proj.bias)

        # MLP weights - both baseline and new models have c_fc and c_proj attributes
        target_model.mlp.c_fc.weight.copy_(base_model.mlp.c_fc.weight)
        target_model.mlp.c_fc.bias.copy_(base_model.mlp.c_fc.bias)
        target_model.mlp.c_proj.weight.copy_(base_model.mlp.c_proj.weight)
        target_model.mlp.c_proj.bias.copy_(base_model.mlp.c_proj.bias)


def benchmark_model(model, input_size, warmup=5, trials=20):
    """Benchmark a single model."""
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

    return sum(times) / len(times) * 1000  # ms


def main():
    print("=" * 70)
    print("" * 15 + "MINIGPT PATTERN ABLATION BENCHMARK")
    print("=" * 70)

    # Model configuration
    init_args = [768, 8, 0.0, 0.0, 1024]
    input_size = (128, 512, 768)

    print(f"\nInput size: {input_size}")
    print(f"Warmup: 5 iterations, Timed: 20 iterations")
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")

    # Create baseline model
    print("Loading models...")
    base_model = create_model(BaseModel, init_args)
    print("  ✓ Baseline loaded")

    # Import and create ablation models (explicit loading to avoid path conflicts)
    fmha_mod = load_whole_model_module("ModelNew_fmha_only.py", "minigpt_fmha_only")
    FmhaOnlyModel = fmha_mod.Model

    mlp_mod = load_whole_model_module("ModelNew_mlp_only.py", "minigpt_mlp_only")
    MlpOnlyModel = mlp_mod.Model

    both_mod = load_whole_model_module("ModelNew.py", "minigpt_both")
    BothModel = both_mod.Model

    fmha_only_model = create_model(FmhaOnlyModel, init_args)
    copy_weights(base_model, fmha_only_model)
    print("  ✓ FMHA-only model loaded")

    mlp_only_model = create_model(MlpOnlyModel, init_args)
    copy_weights(base_model, mlp_only_model)
    print("  ✓ MLP-only model loaded")

    both_model = create_model(BothModel, init_args)
    copy_weights(base_model, both_model)
    print("  ✓ Both model loaded")

    # Check pattern availability
    fmha_status = fmha_only_model.get_pattern_status()
    mlp_status = mlp_only_model.get_pattern_status()
    both_status = both_model.get_pattern_status()

    print(f"\nPattern Status:")
    print(f"  FMHA extension: {'✓ Available' if fmha_status['fmha_available'] else '✗ Not available'}")
    print(f"  MLP extension:  {'✓ Available' if mlp_status['mlp_available'] else '✗ Not available'}")

    if not fmha_status['fmha_available'] and not mlp_status['mlp_available']:
        print("\n⚠ WARNING: Neither extension is available! Results will be invalid.")
        return 1

    # Benchmark all models
    print(f"\nRunning benchmarks...")
    time_base = benchmark_model(base_model, input_size)
    time_fmha_only = benchmark_model(fmha_only_model, input_size)
    time_mlp_only = benchmark_model(mlp_only_model, input_size)
    time_both = benchmark_model(both_model, input_size)

    # Calculate speedups
    speedup_fmha_only = time_base / time_fmha_only
    speedup_mlp_only = time_base / time_mlp_only
    speedup_both = time_base / time_both

    # Calculate throughput (GFLOPS)
    flops = (3 + 1) * input_size[0] * input_size[1] * input_size[2] * input_size[2] + \
            2 * input_size[0] * 8 * input_size[1] * input_size[1] * (input_size[2] // 8)
    gflops = flops / 1e9
    throughput_base = gflops / (time_base / 1000)
    throughput_fmha_only = gflops / (time_fmha_only / 1000)
    throughput_mlp_only = gflops / (time_mlp_only / 1000)
    throughput_both = gflops / (time_both / 1000)

    # Display results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"{'Variant':<15} {'Time (ms)':<12} {'Speedup':<10} {'Throughput':<15}")
    print("-" * 70)
    print(f"{'Baseline':<15} {time_base:<12.3f} {'1.00x':<10} {throughput_base:<15.1f}")
    print(f"{'FMHA only':<15} {time_fmha_only:<12.3f} {speedup_fmha_only:<10.2f} {throughput_fmha_only:<15.1f}")
    print(f"{'MLP only':<15} {time_mlp_only:<12.3f} {speedup_mlp_only:<10.2f} {throughput_mlp_only:<15.1f}")
    print(f"{'Both':<15} {time_both:<12.3f} {speedup_both:<10.2f} {throughput_both:<15.1f}")
    print("=" * 70)

    print(f"\nIndividual Pattern Contributions:")
    print(f"  FMHA speedup:     {speedup_fmha_only:.2f}x")
    print(f"  MLP speedup:      {speedup_mlp_only:.2f}x")
    print(f"  Combined speedup: {speedup_both:.2f}x")

    # A100 theoretical peak (FP32)
    theoretical_peak = 19.5e3  # GFLOPS
    efficiency = (throughput_both / theoretical_peak) * 100
    print(f"\nPeak Efficiency: {efficiency:.1f}% of A100 theoretical peak (FP32)")

    print("=" * 70 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
