"""
Build script for custom CUDA kernels

Usage:
    python setup.py build_ext --inplace

This will compile the CUDA kernels and create a Python module.
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Get the directory containing this file
current_dir = os.path.dirname(os.path.abspath(__file__))

# CUDA extension
cuda_extension = CUDAExtension(
    name='retinaface_cuda',
    sources=[
        os.path.join(current_dir, 'bindings.cpp'),
        os.path.join(current_dir, 'custom_kernels.cu'),
    ],
    extra_compile_args={
        'cxx': [
            '-O3',
            '-std=c++14',
        ],
        'nvcc': [
            '-O3',
            '--use_fast_math',
            '-gencode=arch=compute_60,code=sm_60',  # Pascal
            '-gencode=arch=compute_61,code=sm_61',  # Pascal
            '-gencode=arch=compute_70,code=sm_70',  # Volta
            '-gencode=arch=compute_75,code=sm_75',  # Turing
            '-gencode=arch=compute_80,code=sm_80',  # Ampere
            '-gencode=arch=compute_86,code=sm_86',  # Ampere
            '-gencode=arch=compute_89,code=sm_89',  # Ada Lovelace
            '-gencode=arch=compute_90,code=sm_90',  # Hopper
            '--ptxas-options=-v',
            '-std=c++14',
        ],
    },
)

setup(
    name='retinaface_cuda',
    version='1.0.0',
    author='RetinaFace Optimization Team',
    description='Custom CUDA kernels for RetinaFace inference optimization',
    ext_modules=[cuda_extension],
    cmdclass={'build_ext': BuildExtension},
    install_requires=[
        'torch>=1.9.0',
    ],
)
