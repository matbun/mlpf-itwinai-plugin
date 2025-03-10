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

LOGS_SLURM="logs_slurm"
EXPERIMENTS="experiments_scaling"
REPLICAS=1
NODES_LIST="1 2 4 8" #"1 2 4 8 16"
BASE_TIME=75 #minutes
T=$BASE_TIME #"01:15:00"
# RUN_NAME="mlpf-pyg-ray-bl"
SCRIPT="scripts/vega/slurm.vega.sh"
BASELINE_SCRIPT="scripts/vega/training_ray.sh"

# Variables for SLURM script
export EXPERIMENTS_LOCATION=$EXPERIMENTS
export BATCH_SIZE=32 #32
export N_TRAIN=10000  #100000 #500

# NOTE: remember to check how many GPUs per node were requested in the slurm scripts!

echo "You are going to delete '$LOGS_SLURM' and '$EXPERIMENTS'."
read -p "Do you really want to delete the existing experiments and repeat the scaling test? [y/N] " choice

if [[ "$choice" == "y" || "$choice" == "Y" ]]; then
  rm -rf $LOGS_SLURM logs_torchrun logs_srun mllogs scalability-metrics plots
  mkdir $LOGS_SLURM
  rm -rf $EXPERIMENTS
else
  echo "Keeping existing logs."
fi


# Scaling test
for N in $NODES_LIST
do
    T=$((BASE_TIME / N + 10))

    for (( i=0; i < $REPLICAS; i++  )); do

        # Validation data should be just enough so that all workers receive a bit
        export N_VALID=$((500*N))

        export DIST_MODE="ddp"
        export RUN_NAME="ddp-itwinai-mlpf"
        sbatch \
        --job-name="$RUN_NAME-n$N" \
        --output="$LOGS_SLURM/job-$RUN_NAME-n$N-$i.out" \
        --error="$LOGS_SLURM/job-$RUN_NAME-n$N-$i.err" \
        --nodes=$N \
        --time=$T \
        $SCRIPT

        export DIST_MODE="deepspeed"
        export RUN_NAME="ds-itwinai-mlpf"
        sbatch \
        --job-name="$RUN_NAME-n$N" \
        --output="$LOGS_SLURM/job-$RUN_NAME-n$N-$i.out" \
        --error="$LOGS_SLURM/job-$RUN_NAME-n$N-$i.err" \
        --nodes=$N \
        --time=$T \
        $SCRIPT

        export DIST_MODE="horovod"
        export RUN_NAME="horovod-itwinai-mlpf"
        sbatch \
        --job-name="$RUN_NAME-n$N" \
        --output="$LOGS_SLURM/job-$RUN_NAME-n$N-$i.out" \
        --error="$LOGS_SLURM/job-$RUN_NAME-n$N-$i.err" \
        --nodes=$N \
        --time=$T \
        $SCRIPT

        # Baseline without itwinai
        export RUN_NAME="baseline-mlpf"
        sbatch \
        --job-name="$RUN_NAME-n$N" \
        --output="$LOGS_SLURM/job-$RUN_NAME-n$N-$i.out" \
        --error="$LOGS_SLURM/job-$RUN_NAME-n$N-$i.err" \
        --nodes=$N \
        --time=$T \
        $BASELINE_SCRIPT

    done
done



