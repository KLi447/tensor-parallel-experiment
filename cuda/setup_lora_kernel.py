from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

pytorch_include_paths = torch.utils.cpp_extension.include_paths()

setup(
    name='lora_cuda_ops',
    ext_modules=[
        CUDAExtension('lora_cuda_ops_lib', [
            'lora_kernel.cu',
        ],
        include_dirs=pytorch_include_paths)
    ],
    cmdclass={
        'build_ext': BuildExtension
    })