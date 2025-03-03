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

# Python virtual environment
export PYTHON_VENV=".venv"

# Clear SLURM logs (*.out and *.err files)
rm -rf logs_slurm
mkdir logs_slurm
rm -rf logs_torchrun logs_mpirun logs_srun checkpoints mllogs


# Disable pytest ANSI coloring
export NO_COLOR=1

export DIST_MODE="ddp"
export RUN_NAME="ddp-itwinai"
sbatch  \
    --job-name="$RUN_NAME-n$N" \
    --output="logs_slurm/job-$RUN_NAME-n$N.out" \
    --error="logs_slurm/job-$RUN_NAME-n$N.err" \
    scripts/jsc/slurm.jsc.sh

export DIST_MODE="deepspeed"
export RUN_NAME="ds-itwinai"
sbatch  \
    --job-name="$RUN_NAME-n$N" \
    --output="logs_slurm/job-$RUN_NAME-n$N.out" \
    --error="logs_slurm/job-$RUN_NAME-n$N.err" \
    scripts/jsc/slurm.jsc.sh

export DIST_MODE="horovod"
export RUN_NAME="horovod-itwinai"
sbatch \
    --job-name="$RUN_NAME-n$N" \
    --output="logs_slurm/job-$RUN_NAME-n$N.out" \
    --error="logs_slurm/job-$RUN_NAME-n$N.err" \
    scripts/jsc/slurm.jsc.sh
