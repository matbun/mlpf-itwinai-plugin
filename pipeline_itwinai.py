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

"""
PyTorch supervised training of MLPF using DistributedDataParallel or Ray Train.
Authors: Farouk Mokhtar, Joosep Pata, Eric Wulff, Matteo Bunino
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Dict

import yaml

from itwinai.loggers import LoggersCollection, MLFlowLogger, TensorBoardLogger
from itwinai.pipeline import Pipeline
from itwinai.plugins.mlpf.model.PFDataset import SHARING_STRATEGY
from itwinai.plugins.mlpf.model.trainer_itwinai import MLPFTrainer, override_config
from itwinai.plugins.mlpf.utils import create_experiment_dir

# comet needs to be imported before torch
# from comet_ml import OfflineExperiment, Experiment  # noqa: F401, isort:skip

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

parser = argparse.ArgumentParser()

# add default=None to all arparse arguments to ensure they do not override
# values loaded from the config file given by --config unless explicitly given
parser.add_argument("--config", type=str, default=None, help="yaml config")
parser.add_argument(
    "--prefix", type=str, default=None, help="prefix appended to result dir name"
)
parser.add_argument(
    "--data-dir", type=str, default=None, help="path to `tensorflow_datasets/`"
)
parser.add_argument("--gpus", type=int, default=None, help="to use CPU set to 0; else e.g., 4")
parser.add_argument(
    "--gpu-batch-multiplier",
    type=int,
    default=None,
    help="Increase batch size per GPU by this constant factor",
)
parser.add_argument(
    "--num-workers", type=int, default=None, help="number of processes to load the data"
)
parser.add_argument(
    "--prefetch-factor",
    type=int,
    default=None,
    help="number of samples to fetch & prefetch at every call",
)
parser.add_argument(
    "--resume-training",
    type=str,
    default=None,
    help="training dir containing the checkpointed training to resume",
)
parser.add_argument(
    "--load",
    type=str,
    default=None,
    help="load checkpoint and continue training from previous epoch",
)
parser.add_argument(
    "--relaxed-load",
    action="store_true",
    default=None,
    help="load parameters from the checkpoint model with the same name as the existing model, "
    "ignoring any missing parameters",
)

parser.add_argument("--train", action="store_true", default=None, help="initiates a training")
parser.add_argument("--test", action="store_true", default=None, help="tests the model")
parser.add_argument("--num-epochs", type=int, default=None, help="number of training epochs")
parser.add_argument(
    "--patience", type=int, default=None, help="patience before early stopping"
)
parser.add_argument("--lr", type=float, default=None, help="learning rate")
parser.add_argument(
    "--conv-type",
    type=str,
    default=None,
    help="which graph layer to use",
    choices=["attention", "gnn_lsh", "mamba"],
)
parser.add_argument(
    "--num-convs",
    type=int,
    default=None,
    help="number of cross-particle convolution (GNN, attention, Mamba) layers",
)
parser.add_argument(
    "--make-plots",
    action="store_true",
    default=None,
    help="make plots of the test predictions",
)
parser.add_argument(
    "--ntrain",
    type=int,
    default=None,
    help="training samples to use, if None use entire dataset",
)
parser.add_argument(
    "--ntest",
    type=int,
    default=None,
    help="training samples to use, if None use entire dataset",
)
parser.add_argument("--nvalid", type=int, default=None, help="validation samples to use")
parser.add_argument(
    "--val-freq",
    type=int,
    default=None,
    help="run extra validation every val_freq training steps",
)
parser.add_argument(
    "--checkpoint-freq",
    type=int,
    default=None,
    help="epoch frequency for checkpointing",
)
parser.add_argument(
    "--hpo",
    type=str,
    default=None,
    help="perform hyperparameter optimization, name of HPO experiment",
)
parser.add_argument("--ray-train", action="store_true", help="run training using Ray Train")
parser.add_argument(
    "--ray-local", action="store_true", default=None, help="run ray-train locally"
)
parser.add_argument("--ray-cpus", type=int, default=None, help="CPUs for ray-train")
parser.add_argument("--ray-gpus", type=int, default=None, help="GPUs for ray-train")
parser.add_argument(
    "--raytune-num-samples",
    type=int,
    default=None,
    help="Number of samples to draw from search space",
)
parser.add_argument("--comet", action="store_true", help="use comet ml logging")
parser.add_argument("--comet-offline", action="store_true", help="save comet logs locally")
parser.add_argument(
    "--comet-step-freq",
    type=int,
    default=None,
    help="step frequency for saving comet metrics",
)
parser.add_argument(
    "--experiments-dir",
    type=str,
    default=None,
    help="base directory within which trainings are stored",
)
parser.add_argument(
    "--pipeline", action="store_true", default=None, help="test is running in pipeline"
)
parser.add_argument(
    "--dtype",
    type=str,
    default=None,
    help="data type for training",
    choices=["float32", "float16", "bfloat16"],
)
parser.add_argument(
    "--attention-type",
    type=str,
    default=None,
    help="attention type for self-attention layer",
    choices=["math", "efficient", "flash", "flash_external"],
)
parser.add_argument("--test-datasets", nargs="+", default=[], help="test samples to process")
parser.add_argument("--itwinai-strategy", default="ddp", help="itwinai distributed strategy")
parser.add_argument("--itwinai-trainerv", default=1, type=int, help="itwinai trainer version")
parser.add_argument("--slurm-nnodes", type=int, help="test samples to process")


def get_outdir(resume_training, load):
    outdir = None
    if resume_training is not None:
        outdir = resume_training
    if load is not None:
        pload = Path(load)
        if pload.name == "checkpoint.pth":
            # the checkpoint is likely from a Ray Train run and we need to step one dir
            # higher up
            outdir = str(pload.parent.parent.parent)
        elif pload.name == "best_weights.pth":
            outdir = str(pload.parent)
        else:
            # the checkpoint is likely from a DDP run and we need to step up one dir less
            outdir = str(pload.parent.parent)
    if outdir is not None:
        assert os.path.isfile("{}/model_kwargs.pkl".format(outdir))
    return outdir


def itwinai_pipeline(config: Dict, args, outdir: str) -> Pipeline:
    """Create an itwinai pipeline for MLPF"""
    if args.itwinai_trainerv == 2:
        # First full integration
        config["outdir"] = outdir
        config["storage_path"] = Path(
            args.experiments_dir if args.experiments_dir else "experiments"
        ).resolve()
        config["slurm_nnodes"] = args.slurm_nnodes
        config["ray_cpus"] = args.ray_cpus
        return Pipeline(
            steps=[
                MLPFTrainer(
                    config=config,
                    epochs=config["num_epochs"],
                    strategy=args.itwinai_strategy,
                    checkpoints_location=Path(outdir) / "checkpoints",
                    # logger=MLFlowLogger(log_freq=100),
                )
            ]
        )
    if args.itwinai_trainerv == 3:
        # Full integration and loggers
        config["outdir"] = outdir
        config["storage_path"] = Path(
            args.experiments_dir if args.experiments_dir else "experiments"
        ).resolve()
        config["ray_cpus"] = args.ray_cpus
        config["slurm_nnodes"] = args.slurm_nnodes
        return Pipeline(
            steps=[
                MLPFTrainer(
                    config=config,
                    epochs=config["num_epochs"],
                    strategy=args.itwinai_strategy,
                    checkpoints_location=Path(outdir) / "checkpoints",
                    logger=LoggersCollection(
                        [
                            MLFlowLogger(log_freq=100),
                            TensorBoardLogger(log_freq=100),
                        ]
                    ),
                )
            ]
        )
    if args.itwinai_trainerv == 4:
        # Full integrationa nd checkpoint/loggers disabled
        config["outdir"] = outdir
        config["storage_path"] = Path(
            args.experiments_dir if args.experiments_dir else "experiments"
        ).resolve()
        config["ray_cpus"] = args.ray_cpus
        config["slurm_nnodes"] = args.slurm_nnodes
        return Pipeline(
            steps=[
                MLPFTrainer(
                    config=config,
                    epochs=config["num_epochs"],
                    strategy=args.itwinai_strategy,
                    checkpoints_location=Path(outdir) / "checkpoints",
                    checkpoint_every=None,
                )
            ]
        )

    raise ValueError("unrecognized itwinai-trainerv")


def main():
    # https://github.com/pytorch/pytorch/issues/11201#issuecomment-895047235
    import torch

    torch.multiprocessing.set_sharing_strategy(SHARING_STRATEGY)

    # plt.rcParams['text.usetex'] = True
    args = parser.parse_args()

    if args.resume_training and not args.ray_train:
        raise NotImplementedError(
            "Resuming an interrupted training is only supported in our \
                Ray Train-based training. Consider using `--load` instead, \
                which starts a new training using model weights from a pre-trained checkpoint."
        )

    logging.basicConfig(level=logging.INFO)
    # world_size = (
    #     args.gpus if args.gpus > 0 else 1
    # )  # will be 1 for both cpu (args.gpu < 1) and single-gpu (1)

    with open(
        args.config, "r"
    ) as stream:  # load config (includes: which physics samples, model params)
        config = yaml.safe_load(stream)

    # override some options for the pipeline test
    if args.pipeline:
        config["model"]["gnn_lsh"]["num_convs"] = 1
        config["model"]["gnn_lsh"]["width"] = 64
        config["model"]["gnn_lsh"]["embedding_dim"] = 64

        config["model"]["attention"]["num_convs"] = 1
        config["model"]["attention"]["num_heads"] = 8
        config["model"]["attention"]["head_dim"] = 8

        if config["dataset"] == "cms":
            for ds in ["train_dataset", "valid_dataset"]:
                config[ds]["cms"] = {
                    "physical_pu": {
                        "batch_size": config[ds]["cms"]["physical_pu"]["batch_size"],
                        "samples": {
                            "cms_pf_ttbar": config[ds]["cms"]["physical_pu"]["samples"][
                                "cms_pf_ttbar"
                            ]
                        },
                    }
                }
                # load only the last config split
                config[ds]["cms"]["physical_pu"]["samples"]["cms_pf_ttbar"]["splits"] = ["10"]
            config["test_dataset"] = {"cms_pf_ttbar": config["test_dataset"]["cms_pf_ttbar"]}
            config["test_dataset"]["cms_pf_ttbar"]["splits"] = ["10"]

    # override loaded config with values from command line args
    config = override_config(config, args)

    if args.hpo:
        pass
    else:
        outdir = get_outdir(args.resume_training, config["load"])
        if outdir is None:
            outdir = create_experiment_dir(
                prefix=(args.prefix or "") + Path(args.config).stem + "_",
                experiments_dir=args.experiments_dir
                if args.experiments_dir
                else "experiments",
            )

        # # Save config for later reference. Note that saving happens after parameters are
        # # overwritten by cmd line args.
        # config_filename = "train-config.yaml" if args.train else "test-config.yaml"
        # with open((Path(outdir) / config_filename), "w") as file:
        #     yaml.dump(config, file)

        # Run itwinai training
        print(f"outdir: {outdir}")
        itwinai_pipeline(config, args, outdir).execute()

        # if args.ray_train:
        #     run_ray_training(config, args, outdir)
        # else:
        #     device_agnostic_run(config, world_size, outdir)


if __name__ == "__main__":
    main()
