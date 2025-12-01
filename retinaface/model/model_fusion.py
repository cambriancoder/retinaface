"""
Conv-BatchNorm-ReLU Fusion Utilities for RetinaFace Model Optimization

This module provides utilities to fuse Conv2D-BatchNorm-ReLU sequences into
optimized Conv2D layers, eliminating BatchNorm overhead during inference.

Mathematical Background:
-----------------------
BatchNorm: y = γ * (x - μ) / σ + β
where:
  - γ (gamma): scale parameter
  - β (beta): shift parameter
  - μ (mu): running mean
  - σ (sigma): running standard deviation

For Conv2D followed by BatchNorm, we can fold BN into Conv:
  W_fused = γ * W / σ
  b_fused = γ * (b - μ) / σ + β

This eliminates the BatchNorm operation entirely while maintaining
mathematical equivalence.
"""

import numpy as np
from typing import Tuple, Optional


def fuse_conv_bn_weights(
    conv_weights: np.ndarray,
    conv_bias: Optional[np.ndarray],
    bn_gamma: np.ndarray,
    bn_beta: np.ndarray,
    bn_mean: np.ndarray,
    bn_var: np.ndarray,
    epsilon: float = 1e-5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fuse Conv2D and BatchNorm parameters into a single Conv2D layer.

    Args:
        conv_weights: Conv2D weights, shape (H, W, C_in, C_out)
        conv_bias: Conv2D bias, shape (C_out,) or None
        bn_gamma: BatchNorm scale, shape (C_out,)
        bn_beta: BatchNorm shift, shape (C_out,)
        bn_mean: BatchNorm running mean, shape (C_out,)
        bn_var: BatchNorm running variance, shape (C_out,)
        epsilon: BatchNorm epsilon value

    Returns:
        Tuple of (fused_weights, fused_bias)
    """
    # Compute the scale factor: γ / sqrt(σ² + ε)
    scale = bn_gamma / np.sqrt(bn_var + epsilon)

    # Fuse weights: W_fused = γ * W / sqrt(σ² + ε)
    # Broadcasting: scale shape (C_out,) broadcasts to (H, W, C_in, C_out)
    fused_weights = conv_weights * scale.reshape(1, 1, 1, -1)

    # Fuse bias: b_fused = γ * (b - μ) / sqrt(σ² + ε) + β
    if conv_bias is None:
        conv_bias = np.zeros_like(bn_mean)

    fused_bias = scale * (conv_bias - bn_mean) + bn_beta

    return fused_weights, fused_bias


def get_layer_by_name(model, layer_name: str):
    """Get a layer from the model by name."""
    for layer in model.layers:
        if layer.name == layer_name:
            return layer
    return None


def fuse_model_conv_bn_pairs(model):
    """
    Create a new model with Conv-BN pairs fused.

    This function identifies Conv2D-BatchNorm patterns in the model
    and fuses them into single Conv2D layers with modified weights.

    Args:
        model: Original Keras/TensorFlow model

    Returns:
        Dictionary mapping Conv layer names to fused weights and biases
    """
    fused_weights_dict = {}

    # Pattern to identify: Conv2D -> BatchNorm
    # In RetinaFace, the pattern is:
    # 1. conv_layer = Conv2D(..., use_bias=False)(input)
    # 2. bn_layer = BatchNormalization(..., trainable=False)(conv_layer)

    for i, layer in enumerate(model.layers):
        # Check if this is a BatchNorm layer
        if "BatchNormalization" in str(type(layer)):
            # Get the input layer (should be Conv2D)
            input_layer_name = layer.input.name.split("/")[0]
            conv_layer = get_layer_by_name(model, input_layer_name)

            if conv_layer is not None and "Conv2D" in str(type(conv_layer)):
                # Get Conv2D weights
                conv_weights = conv_layer.get_weights()
                if len(conv_weights) == 2:  # Has bias
                    W_conv, b_conv = conv_weights
                elif len(conv_weights) == 1:  # No bias
                    W_conv = conv_weights[0]
                    b_conv = None
                else:
                    continue

                # Get BatchNorm weights: [gamma, beta, moving_mean, moving_var]
                bn_weights = layer.get_weights()
                if len(bn_weights) != 4:
                    continue

                gamma, beta, moving_mean, moving_var = bn_weights

                # Get epsilon from layer config
                epsilon = layer.epsilon

                # Fuse the weights
                W_fused, b_fused = fuse_conv_bn_weights(
                    W_conv, b_conv, gamma, beta, moving_mean, moving_var, epsilon
                )

                # Store fused weights
                fused_weights_dict[conv_layer.name] = {
                    "weights": W_fused,
                    "bias": b_fused,
                    "bn_layer_name": layer.name,
                }

                print(
                    f"✓ Fused {conv_layer.name} + {layer.name} "
                    f"(weights: {W_fused.shape}, bias: {b_fused.shape})"
                )

    return fused_weights_dict


def apply_fused_weights(model, fused_weights_dict):
    """
    Apply fused weights to a model in-place.

    Args:
        model: The model to modify
        fused_weights_dict: Dictionary from fuse_model_conv_bn_pairs

    Returns:
        Modified model with fused weights applied
    """
    for conv_name, fused_data in fused_weights_dict.items():
        conv_layer = get_layer_by_name(model, conv_name)
        if conv_layer is not None:
            # Set the fused weights and bias
            conv_layer.set_weights([fused_data["weights"], fused_data["bias"]])
            print(f"✓ Applied fused weights to {conv_name}")

    return model


def estimate_fusion_speedup(model):
    """
    Estimate the performance improvement from fusion.

    Args:
        model: The original model

    Returns:
        Dictionary with speedup estimates
    """
    bn_count = 0
    conv_bn_pairs = 0

    for i, layer in enumerate(model.layers):
        if "BatchNormalization" in str(type(layer)):
            bn_count += 1

            # Check if previous layer is Conv2D
            if i > 0:
                prev_layer = model.layers[i - 1]
                if "Conv2D" in str(type(prev_layer)):
                    conv_bn_pairs += 1

    # Estimate memory bandwidth reduction
    # Each BN layer saves ~4 memory reads/writes
    memory_reduction_pct = (conv_bn_pairs / len(model.layers)) * 100

    # Estimate compute reduction
    # BN typically takes ~10-15% of conv time
    compute_reduction_pct = conv_bn_pairs * 0.12 * 100 / len(model.layers)

    return {
        "total_bn_layers": bn_count,
        "fusable_conv_bn_pairs": conv_bn_pairs,
        "estimated_memory_reduction_pct": memory_reduction_pct,
        "estimated_compute_reduction_pct": compute_reduction_pct,
        "estimated_total_speedup_pct": min(compute_reduction_pct + memory_reduction_pct / 2, 20),
    }


if __name__ == "__main__":
    print("Conv-BN-ReLU Fusion Utilities")
    print("=" * 60)
    print("This module provides utilities for fusing BatchNorm layers")
    print("into Conv2D layers for improved inference performance.")
    print("=" * 60)
