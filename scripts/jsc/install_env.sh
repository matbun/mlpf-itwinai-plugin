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

# shellcheck disable=all

# Load modules
ml --force purge
ml Stages/2024 GCC OpenMPI CUDA/12 cuDNN MPI-settings/CUDA
ml Python CMake HDF5 PnetCDF libaio mpi4py

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

# # Legacy-style installation
# python -m venv .venv
# source .venv/bin/activate
# pip install --no-cache-dir -e . --extra-index-url https://download.pytorch.org/whl/cu121
