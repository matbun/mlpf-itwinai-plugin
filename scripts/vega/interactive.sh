#!/bin/bash

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

env(){
    # Load env modules

    ml --force purge
    ml Python/3.11.5-GCCcore-13.2.0 
    ml CMake/3.24.3-GCCcore-11.3.0
    ml mpi4py
    ml OpenMPI
    ml CUDA/12.3
    ml GCCcore/11.3.0
    ml NCCL
    ml cuDNN/8.9.7.29-CUDA-12.3.0
    ml UCX-CUDA/1.15.0-GCCcore-13.2.0-CUDA-12.3.0
    module unload OpenSSL
}

alloc(){
    # Allocate a node interactively
    
    salloc \
        --partition=gpu \
        --account=d2024d11-083-users  \
        --nodes=1 \
        --gres=gpu:4 \
        --gpus-per-node=4 \
        --time=1:59:00 \
        --cpus-per-task=24 \
        --ntasks-per-node=1
}

term(){
    # Open a terminal in the allocated node
    
    srun --jobid $1 --overlap --pty /bin/bash
}

ray(){
    # Create a dumy Ray cluster of 2 nodes

    uv run ray stop
    uv run ray start \
        --head \
        --node-ip-address=127.0.0.1 \
        --port=7639 \
        --num-cpus 1
    echo "HEAD NODE STARTED" 
    uv run ray start \
        --address=127.0.0.1:7639 \
        --num-cpus 1
    echo "WORKER NODE STARTED" 
}

run(){
    # CPU-only execution on login node
    
    RAY_CPUS=32
    RAY_GPUS=1
    
    uv run ray stop
    uv run ray start \
        --head \
        --node-ip-address=localhost \
        --port=7639 \
        --num-cpus=$RAY_CPUS \
        --num-gpus=$RAY_GPUS 
        # --block &
    echo "RAY STARTED"
    
    uv run python -u pipeline.py \
        --train \
        --ray-train \
        --config parameters/pyg-clic-itwinai.yaml \
        --data-dir /ceph/hpc/data/d2024d11-083-users/data/tensorflow_datasets/clic \
        --ntrain 50 \
        --nvalid 50 \
        --prefix foo_prefix \
        --ray-cpus $RAY_CPUS \
        --gpus $RAY_GPUS \
        --gpu-batch-multiplier 8 \
        --num-workers 8 \
        --prefetch-factor 8 \
        --experiments-dir $PWD/experiments \
        --slurm-nnodes 1 \
        --num-epochs 2
}

export ITWINAI_LOG_LEVEL=DEBUG

run_itwinai(){

    RAY_CPUS=32
    RAY_GPUS=1

    uv run ray stop

    # Make mlpf visible
    export PYTHONPATH="$PWD:$PYTHONPATH"

    uv run python -u \
        pipeline_itwinai.py \
        --train \
        --ray-train \
        --config parameters/pyg-clic-itwinai.yaml \
        --data-dir /ceph/hpc/data/d2024d11-083-users/data/tensorflow_datasets/clic \
        --ntrain 50 \
        --nvalid 50 \
        --ray-cpus $RAY_CPUS \
        --gpus $RAY_GPUS \
        --prefix foo_prefix \
        --gpu-batch-multiplier 8 \
        --num-workers 8 \
        --prefetch-factor 8 \
        --experiments-dir $PWD/experiments \
        --num-epochs 2 \
        --slurm-nnodes 1 \
        --itwinai-trainerv 4
}

run_itwinai_ray(){

    RAY_CPUS=32
    RAY_GPUS=1
    
    uv run ray stop
    uv run ray start \
        --head \
        --node-ip-address=localhost \
        --port=7639 \
        --num-cpus=$RAY_CPUS \
        --num-gpus=$RAY_GPUS 
        # --block &
    echo "RAY STARTED"

    # Make mlpf visible
    export PYTHONPATH="$PWD:$PYTHONPATH"

    # uv run python -Xfrozen_modules=off -m debugpy --listen 5678 --wait-for-client  \
    uv run python -u \
       pipeline_itwinai.py \
        --train \
        --ray-train \
        --config parameters/pyg-clic-itwinai.yaml \
        --data-dir /ceph/hpc/data/d2024d11-083-users/data/tensorflow_datasets/clic \
        --ntrain 50 \
        --nvalid 50 \
        --ray-cpus $RAY_CPUS \
        --gpus $RAY_GPUS \
        --prefix foo_prefix \
        --gpu-batch-multiplier 8 \
        --num-workers 8 \
        --prefetch-factor 8 \
        --experiments-dir $PWD/experiments \
        --num-epochs 2 \
        --slurm-nnodes 1 \
        --itwinai-trainerv 4
}