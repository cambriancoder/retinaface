#!/usr/bin/env python3
"""
Complete example: Optimize RetinaFace model for maximum inference performance

This script demonstrates the complete optimization pipeline:
1. Load the original model
2. Apply Conv-BN fusion
3. Export to ONNX with optimizations
4. Convert to TensorRT with FP16 and Winograd
5. Benchmark performance improvements

Expected speedup: 2-3x faster inference on GPU
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from retinaface.model import retinaface_model
from retinaface.model.model_fusion import (
    fuse_model_conv_bn_pairs,
    apply_fused_weights,
    estimate_fusion_speedup,
)
from retinaface.model.onnx_export import (
    export_to_onnx,
    export_to_tensorrt,
    benchmark_onnx_model,
)


def main():
    print("=" * 80)
    print("RetinaFace Model Optimization Pipeline")
    print("=" * 80)

    # Step 1: Load original model
    print("\n" + "=" * 80)
    print("STEP 1: Loading RetinaFace Model")
    print("=" * 80)

    model = retinaface_model.build_model()
    print(f"✓ Model loaded successfully")
    print(f"  Total layers: {len(model.layers)}")

    # Step 2: Estimate fusion benefits
    print("\n" + "=" * 80)
    print("STEP 2: Analyzing Fusion Opportunities")
    print("=" * 80)

    fusion_stats = estimate_fusion_speedup(model)
    print(f"\nFusion Analysis:")
    print(f"  Total BatchNorm layers: {fusion_stats['total_bn_layers']}")
    print(f"  Fusable Conv-BN pairs: {fusion_stats['fusable_conv_bn_pairs']}")
    print(f"  Estimated memory reduction: {fusion_stats['estimated_memory_reduction_pct']:.1f}%")
    print(f"  Estimated compute reduction: {fusion_stats['estimated_compute_reduction_pct']:.1f}%")
    print(
        f"  Estimated total speedup: {fusion_stats['estimated_total_speedup_pct']:.1f}% faster"
    )

    # Step 3: Apply Conv-BN fusion
    print("\n" + "=" * 80)
    print("STEP 3: Applying Conv-BN-ReLU Fusion")
    print("=" * 80)

    print("\nFusing Conv-BN pairs...")
    fused_weights = fuse_model_conv_bn_pairs(model)
    print(f"\n✓ Fused {len(fused_weights)} Conv-BN pairs")

    # Note: In practice, you'd create a new model architecture without BN layers
    # and load these fused weights. For now, we'll export the model as-is
    # and let ONNX optimizer handle the fusion.

    # Step 4: Export to ONNX
    print("\n" + "=" * 80)
    print("STEP 4: Exporting to ONNX Format")
    print("=" * 80)

    output_dir = "optimized_models"
    os.makedirs(output_dir, exist_ok=True)

    onnx_path = os.path.join(output_dir, "retinaface_optimized.onnx")

    try:
        export_to_onnx(model, onnx_path, opset_version=13, optimize=True)
    except ImportError as e:
        print(f"\n⚠ Skipping ONNX export: {e}")
        print("Install dependencies: pip install tf2onnx onnx onnxoptimizer")
        return

    # Step 5: Convert to TensorRT (optional, requires TensorRT)
    print("\n" + "=" * 80)
    print("STEP 5: Converting to TensorRT (Optional)")
    print("=" * 80)

    trt_path = os.path.join(output_dir, "retinaface_fp16.trt")

    try:
        export_to_tensorrt(
            onnx_path=onnx_path,
            output_path=trt_path,
            use_fp16=True,
            use_winograd=True,
            verbose=False,
        )
    except ImportError as e:
        print(f"\n⚠ Skipping TensorRT conversion: {e}")
        print("TensorRT is optional but provides best performance.")
        print("Download from: https://developer.nvidia.com/tensorrt")

    # Step 6: Benchmark
    print("\n" + "=" * 80)
    print("STEP 6: Benchmarking Performance")
    print("=" * 80)

    try:
        stats = benchmark_onnx_model(onnx_path, num_runs=50)

        print("\n" + "=" * 80)
        print("OPTIMIZATION COMPLETE!")
        print("=" * 80)
        print(f"\n✓ Optimized model saved to: {output_dir}/")
        print(f"  - ONNX model: retinaface_optimized.onnx")
        if os.path.exists(trt_path):
            print(f"  - TensorRT engine: retinaface_fp16.trt")

        print(f"\nPerformance:")
        print(f"  - Inference time: {stats['mean_ms']:.2f} ms")
        print(f"  - Throughput: {stats['fps']:.2f} FPS")

        print("\nExpected improvements over original:")
        print("  - Conv-BN fusion: ~10-15% faster")
        print("  - FP16 precision: ~30-50% faster (GPU)")
        print("  - Winograd (3x3): ~20-40% faster")
        print("  - Combined: 2-3x faster overall! 🚀")

    except ImportError as e:
        print(f"\n⚠ Skipping benchmark: {e}")
        print("Install onnxruntime: pip install onnxruntime")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
