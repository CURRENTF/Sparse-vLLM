from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

setup(
    name='cpu_gpu_transfer',
    ext_modules=[
        CUDAExtension(
            name='cpu_gpu_transfer_cuda',
            sources=[
                'kernels/gather_copy.cu',
            ],
            include_dirs=[
                os.path.abspath('kernels')
            ],
            extra_compile_args={
                'cxx': ['-std=c++17'],
                'nvcc': ['-std=c++17', '--expt-relaxed-constexpr'],
            },
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
)
