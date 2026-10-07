#!/usr/bin/env bash
# Build PyTorch 2.4.1 (last release with Python 3.8 support) for Blackwell sm_120.
set -euo pipefail
TC="$(cd "$(dirname "$0")" && pwd)"
export CUDA_HOME="$TC/cuda-12.8" PATH="$TC/cuda-12.8/bin:$TC/build-venv/bin:$PATH"
export CUDACXX="$TC/cuda-12.8/bin/nvcc" CUDA_NVCC_EXECUTABLE="$TC/cuda-12.8/bin/nvcc" CMAKE_CUDA_COMPILER="$TC/cuda-12.8/bin/nvcc"
export TORCH_CUDA_ARCH_LIST="12.0" USE_CUDA=1 USE_CUDNN=0 USE_NCCL=0 USE_DISTRIBUTED=1 \
       USE_FLASH_ATTENTION=0 USE_MEM_EFF_ATTENTION=0 BUILD_TEST=0 USE_KINETO=0 \
       USE_ROCM=0 USE_XPU=0 USE_MPS=0 MAX_JOBS=30 PYTORCH_BUILD_VERSION=2.4.1+cu128sm120 PYTORCH_BUILD_NUMBER=1
# Source patches for sm_120 (already applied in pytorch-src, see `git -C pytorch-src diff`):
#  - cmake select_compute_arch.cmake: accept two-digit arch majors ("12.0")
#  - torch/utils/cpp_extension.py: allow 10.0/12.0 and parse two-digit majors
#  - torch/__init__.py: preload the CUDA 12.8 libnvrtc (system CUDA 12.0 copy rejects sm_120)
cd "$TC/pytorch-src"
python setup.py bdist_wheel
ls -la dist
