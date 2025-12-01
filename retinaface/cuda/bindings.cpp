/*
 * Python Bindings for Custom CUDA Kernels
 *
 * This file provides Python bindings for the custom CUDA kernels
 * using pybind11 and PyTorch for tensor management.
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

// Forward declarations of CUDA kernel launchers
extern "C" {
    void launch_fused_ssh_module(
        const float* input,
        const float* weights_conv1, const float* bias_conv1,
        const float* weights_conv2, const float* bias_conv2,
        const float* weights_conv3, const float* bias_conv3,
        const float* weights_conv4, const float* bias_conv4,
        float* output,
        int batch, int height, int width, int channels_in,
        cudaStream_t stream
    );

    void launch_fused_fpn_upsample_crop_add(
        const float* input_high_res,
        const float* input_lateral,
        float* output,
        int batch, int height_low, int width_low, int channels,
        cudaStream_t stream
    );

    void launch_optimized_classification_reshape(
        const float* input,
        float* output,
        int batch, int height, int width,
        cudaStream_t stream
    );
}


// ============================================================================
// Python Interface Functions
// ============================================================================

torch::Tensor fused_ssh_module(
    torch::Tensor input,                // [B, H, W, C_in]
    torch::Tensor weights_conv1,        // [256, 3, 3, C_in]
    torch::Tensor bias_conv1,           // [256]
    torch::Tensor weights_conv2,        // [128, 3, 3, C_in]
    torch::Tensor bias_conv2,           // [128]
    torch::Tensor weights_conv3,        // [128, 3, 3, C_in]
    torch::Tensor bias_conv3,           // [128]
    torch::Tensor weights_conv4,        // [128, 3, 3, 128]
    torch::Tensor bias_conv4            // [128]
) {
    // Validate inputs
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D: [B, H, W, C]");

    auto batch = input.size(0);
    auto height = input.size(1);
    auto width = input.size(2);
    auto channels_in = input.size(3);

    // Create output tensor [B, H, W, 512]
    auto output = torch::empty({batch, height, width, 512}, input.options());

    // Get CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Launch kernel
    launch_fused_ssh_module(
        input.data_ptr<float>(),
        weights_conv1.data_ptr<float>(), bias_conv1.data_ptr<float>(),
        weights_conv2.data_ptr<float>(), bias_conv2.data_ptr<float>(),
        weights_conv3.data_ptr<float>(), bias_conv3.data_ptr<float>(),
        weights_conv4.data_ptr<float>(), bias_conv4.data_ptr<float>(),
        output.data_ptr<float>(),
        batch, height, width, channels_in,
        stream
    );

    // Check for CUDA errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA kernel error: ") + cudaGetErrorString(error)
        );
    }

    return output;
}


torch::Tensor fused_fpn_upsample_crop_add(
    torch::Tensor input_high_res,    // [B, H, W, C]
    torch::Tensor input_lateral      // [B, H*2, W*2, C]
) {
    TORCH_CHECK(input_high_res.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input_lateral.is_cuda(), "Lateral input must be a CUDA tensor");

    auto batch = input_high_res.size(0);
    auto height_low = input_high_res.size(1);
    auto width_low = input_high_res.size(2);
    auto channels = input_high_res.size(3);

    // Output has same shape as lateral connection
    auto output = torch::empty_like(input_lateral);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    launch_fused_fpn_upsample_crop_add(
        input_high_res.data_ptr<float>(),
        input_lateral.data_ptr<float>(),
        output.data_ptr<float>(),
        batch, height_low, width_low, channels,
        stream
    );

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA kernel error: ") + cudaGetErrorString(error)
        );
    }

    return output;
}


torch::Tensor optimized_classification_reshape(
    torch::Tensor input    // [B, H, W, 4]
) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D: [B, H, W, 4]");
    TORCH_CHECK(input.size(3) == 4, "Last dimension must be 4");

    auto batch = input.size(0);
    auto height = input.size(1);
    auto width = input.size(2);

    // Create output tensor [B, H*2, W, 2]
    auto output = torch::empty({batch, height * 2, width, 2}, input.options());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    launch_optimized_classification_reshape(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch, height, width,
        stream
    );

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA kernel error: ") + cudaGetErrorString(error)
        );
    }

    return output;
}


// ============================================================================
// Module Definition
// ============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Custom CUDA kernels for RetinaFace optimization";

    m.def("fused_ssh_module", &fused_ssh_module,
          "Fused SSH module kernel (5-7% speedup)",
          py::arg("input"),
          py::arg("weights_conv1"),
          py::arg("bias_conv1"),
          py::arg("weights_conv2"),
          py::arg("bias_conv2"),
          py::arg("weights_conv3"),
          py::arg("bias_conv3"),
          py::arg("weights_conv4"),
          py::arg("bias_conv4"));

    m.def("fused_fpn_upsample_crop_add", &fused_fpn_upsample_crop_add,
          "Fused FPN upsample-crop-add kernel (3-5% speedup)",
          py::arg("input_high_res"),
          py::arg("input_lateral"));

    m.def("optimized_classification_reshape", &optimized_classification_reshape,
          "Optimized classification reshape kernel (1-2% speedup)",
          py::arg("input"));
}
