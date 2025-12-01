# RetinaFace Optimization Examples

This directory contains example scripts demonstrating how to optimize the RetinaFace model for maximum inference performance.

## Quick Start

### 1. Install Dependencies

```bash
# Core dependencies for ONNX export
pip install tf2onnx onnx onnxoptimizer onnxruntime

# Optional: TensorRT (for maximum performance)
# Download from: https://developer.nvidia.com/tensorrt
```

### 2. Run Complete Optimization Pipeline

```bash
python examples/optimize_model.py
```

This will:
1. Load the RetinaFace model
2. Analyze fusion opportunities
3. Apply Conv-BN fusion
4. Export to ONNX with optimizations
5. Convert to TensorRT with FP16 + Winograd
6. Benchmark performance

## Optimization Techniques

### 1. **Conv-BatchNorm Fusion** (10-15% speedup)

Fuse BatchNorm parameters into Conv weights, eliminating 130+ BatchNorm layers.

```python
from retinaface.model import retinaface_model
from retinaface.model.model_fusion import fuse_model_conv_bn_pairs

model = retinaface_model.build_model()
fused_weights = fuse_model_conv_bn_pairs(model)
```

### 2. **ONNX Export** (5-10% speedup)

Export to ONNX format with automatic graph optimizations.

```python
from retinaface.model.onnx_export import export_to_onnx

export_to_onnx(
    model,
    "retinaface_optimized.onnx",
    optimize=True
)
```

### 3. **TensorRT with FP16** (30-50% speedup)

Convert to TensorRT engine with FP16 mixed precision (NOT quantization).

```python
from retinaface.model.onnx_export import export_to_tensorrt

export_to_tensorrt(
    onnx_path="retinaface.onnx",
    output_path="retinaface_fp16.trt",
    use_fp16=True,
    use_winograd=True
)
```

### 4. **Winograd Convolutions** (20-40% speedup)

Automatically enabled in TensorRT for 3x3 convolutions.

## Performance Expectations

| Optimization | Inference Time | Speedup |
|-------------|----------------|---------|
| Original Model | ~100ms | 1.0x (baseline) |
| + Conv-BN Fusion | ~85ms | 1.18x |
| + ONNX Optimizations | ~80ms | 1.25x |
| + FP16 Precision | ~50ms | 2.0x |
| + Winograd | **~35ms** | **2.8x** 🚀 |

**Total expected speedup: 2-3x faster inference!**

## Hardware Requirements

### For FP16 Mixed Precision:
- NVIDIA GPU with Tensor Cores
- RTX 20xx series or newer
- Tesla V100, A100
- GTX 16xx series (limited support)

### For Basic Optimizations (no GPU required):
- Conv-BN fusion: CPU-friendly
- ONNX optimizations: CPU-friendly
- Works on any hardware

## Custom Kernel Opportunities

For advanced users, additional speedup is possible with custom CUDA kernels:

### 1. Fused SSH Module Kernel
Combine the 3 parallel SSH detection convolutions into a single kernel.

**Expected gain**: 5-7% faster

### 2. Fused FPN Upsample-Crop-Add Kernel
Custom kernel for Feature Pyramid Network operations.

**Expected gain**: 3-5% faster

### 3. Custom Classification Reshape
Optimized memory access pattern for score reshaping.

**Expected gain**: 1-2% faster

**Total with custom kernels: 3-3.5x faster overall!**

## Troubleshooting

### "No module named 'tf2onnx'"
```bash
pip install tf2onnx onnx onnxoptimizer
```

### "TensorRT not found"
TensorRT is optional but provides best performance. Download from:
https://developer.nvidia.com/tensorrt

### "FP16 not supported on this GPU"
FP16 requires modern NVIDIA GPUs with Tensor Cores. The script will automatically fall back to FP32.

## Additional Resources

- [OPTIMIZATIONS.md](../OPTIMIZATIONS.md) - Detailed optimization documentation
- [model_fusion.py](../retinaface/model/model_fusion.py) - Conv-BN fusion utilities
- [onnx_export.py](../retinaface/model/onnx_export.py) - ONNX/TensorRT export utilities

## Questions?

See the main [OPTIMIZATIONS.md](../OPTIMIZATIONS.md) for more details on each optimization technique.
