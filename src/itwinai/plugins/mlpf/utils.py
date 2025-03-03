# Copyright 2025 Matteo Bunino, CERN
#
# Original work Copyright 2021-2025 Joosep Pata, Eric Wulff, Farouk Mokhtar,
# Javier Duarte, Aadi Tepper, Ka Wa Ho, & Lars Sørlie
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
#
# File adapted from particleflow (https://github.com/jpata/particleflow/tree/v2.2.0)
# for the itwinai plugin (https://github.com/matbun/mlpf-itwinai-plugin)


import datetime
import logging
import platform
from pathlib import Path

# from comet_ml import OfflineExperiment, Experiment  # isort:skip


def create_experiment_dir(prefix=None, suffix=None, experiments_dir="experiments"):
    if prefix is None:
        train_dir = Path(experiments_dir) / datetime.datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )
    else:
        train_dir = Path(experiments_dir) / (
            prefix + datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        )

    if suffix is not None:
        train_dir = train_dir.with_name(train_dir.name + "." + platform.node())

    train_dir.mkdir(parents=True, exist_ok=True)

    return str(train_dir)


def create_comet_experiment(comet_exp_name, comet_offline=False, outdir=None):
    from comet_ml import OfflineExperiment, Experiment  # isort:skip

    try:
        if comet_offline:
            logging.info("Using comet-ml OfflineExperiment, saving logs locally.")
            if outdir is None:
                raise ValueError(
                    "Please specify am output directory when setting comet_offline to True"
                )

            experiment = OfflineExperiment(
                project_name=comet_exp_name,
                auto_metric_logging=True,
                auto_param_logging=True,
                auto_histogram_weight_logging=True,
                auto_histogram_gradient_logging=False,
                auto_histogram_activation_logging=False,
                offline_directory=outdir + "/cometml",
                auto_output_logging="simple",
            )
        else:
            logging.info("Using comet-ml Experiment, streaming logs to www.comet.ml.")

            experiment = Experiment(
                project_name=comet_exp_name,
                auto_metric_logging=True,
                auto_param_logging=True,
                auto_histogram_weight_logging=True,
                auto_histogram_gradient_logging=False,
                auto_histogram_activation_logging=False,
                auto_output_logging="simple",
            )
    except Exception as e:
        logging.warning("Failed to initialize comet-ml dashboard: {}".format(e))
        experiment = None
    return experiment
