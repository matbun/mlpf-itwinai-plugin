#!/bin/bash
# -*- coding: utf-8 -*-

# Copyright 2025 Matteo Bunino, CERN
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Load modules
# NOTE: REFLECT THEM IN THE MAIN README! 
ml --force purge
ml Python/3.11.5-GCCcore-13.2.0 
ml CMake/3.24.3-GCCcore-11.3.0
ml mpi4py
ml OpenMPI
ml CUDA/12.3
ml GCCcore/11.3.0
ml NCCL
ml cuDNN/8.9.7.29-CUDA-12.3.0
# ml UCX-CUDA/1.15.0-GCCcore-13.2.0-CUDA-12.3.0 # this may cause problems with horovod as it pushes gcc to version 13, which is not supported by the build setup

# You should have CUDA 12.3 now

# Create and install torch env (uv installation: https://docs.astral.sh/uv/getting-started/installation/)
uv venv
uv pip install --no-cache-dir -e . --extra-index-url https://download.pytorch.org/whl/cu121 --index-strategy unsafe-best-match

uv pip install --no-cache-dir "deepspeed==0.15.*"

# Horovod variables
export LDSHARED="$CC -shared" &&
export CMAKE_CXX_STANDARD=17 

export HOROVOD_MPI_THREADS_DISABLE=1
export HOROVOD_CPU_OPERATIONS=MPI

export HOROVOD_GPU_ALLREDUCE=NCCL
export HOROVOD_NCCL_LINK=SHARED
export HOROVOD_NCCL_HOME=$EBROOTNCCL

export HOROVOD_WITH_PYTORCH=1
export HOROVOD_WITHOUT_TENSORFLOW=1
export HOROVOD_WITHOUT_MXNET=1

uv pip install --no-cache-dir --no-build-isolation 'horovod[pytorch] @ git+https://github.com/horovod/horovod'
# uv run horovodrun --check-build

# # Legacy-style installation
# python -m venv .venv
# source .venv/bin/activate
# pip install --no-cache-dir -e . --extra-index-url https://download.pytorch.org/whl/cu121
