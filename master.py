# Copyright (c) 2023 Mobius Logic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import importlib
import inspect
import json
import logging
import os
import random
import time
from typing import Any, Optional

import yaml

# General logging level
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

# Tensorflow logging level
#  https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information/42121886#42121886
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # any {'0', '1', '2', '3'}


###############################################################################
## Main Python Functions
###############################################################################
####################
## New MarcoPolo Run
####################
def setup_marcopolo_core(args: argparse.Namespace) -> Any:
    """Set up and return the marco polo core

    Dynamic loading of the core is done based on the value of
    "core" in the input.

    Envs are loaded dynamically when "env" is given.
    Note that env is required when the core supports dynamic
    loading and prohibited when it doesn't. This is not checked
    during loading. There will be errors if env is used incorrectly

    Parameters
    ----------
    args : argparse.Namespace
        Object containing all of the global parameters.

    Side-Effects
    ------------
    * The core module named in args is dynamically imported
    * The random number seed is set from a value in args

    Returns
    -------
    MarcoPolo Core
        The core simulation object
    """
    args.vidpath = setup_video_directory(args)

    # Set master_seed
    #  this is the only way to get reproducibility from neat-python
    random.seed(args.master_seed)

    # Dynamically load the Core Manager
    logger.info("Training Core: marco_polo.cores." + args.core)
    tmp_mod = importlib.import_module("marco_polo.cores." + args.core)
    Core = tmp_mod.Core

    # if an env is requested, load it and use in core
    req_params = inspect.signature(Core.__init__).parameters

    kwargs = {"args": args}

    if "env_factory" in req_params.keys():
        if hasattr(args, "env"):
            # Dynamically load the Env creation function
            logger.info("Training Environment: marco_polo.envs." + args.env)
            env_mod = importlib.import_module("marco_polo.envs." + args.env)
            # function to return class of envs
            env_class_factory = env_mod.get_env_class
            # function to return class of env params
            env_param_class_factory = env_mod.get_env_param_class

            kwargs["env_factory"] = env_class_factory
            kwargs["env_param_factory"] = env_param_class_factory

        else:
            raise Exception("Core requires 'env' to be specified in the .yaml.")

    if "manager_factory" in req_params.keys():
        if hasattr(args, "optimizer"):
            # Dynamically load the Env creation function
            logger.info("Training Optimizer: marco_polo.optimizers." + args.optimizer)
            env_mod = importlib.import_module("marco_polo.optimizers." + args.optimizer)
            # function to return class of envs
            opt_class_factory = env_mod.get_opt_class
            # function to return class of env params
            model_class_factory = env_mod.get_model

            # Initialize marco polo population (env/team pairs) manager
            kwargs["manager_factory"] = opt_class_factory
            kwargs["model_factory"] = model_class_factory

        else:
            raise Exception("Core requires 'optimizer' to be specified in the .yaml.")

    # no env is requested, return core with env
    return Core(**kwargs)


########################
## Restart MarcoPolo Run
########################
def reload_core_from_checkpoint(
    args: argparse.Namespace,
    preload_args: Optional[dict[str, Any]] = None,
    print_args: bool = True,
) -> tuple[Any, int]:
    """Set up and return the marcopolo core from a checkpoint

    Parameters
    ----------
    args: argparse.Namespace
        Object containing all of the global parameters.
    preload_args: dict[str, Any], optional
        Arguments to apply prior to creating the marcopolo core
        object. This would be used to override any of the args
        in the checkpoint args file. An example would be to change
        the number of workers to create in the manager.
        Default is None which means not to override anything
    print_args: bool, default = True
        Whether to print the new args after reading from
        the checkpoint.

    Side-Effects
    ------------
    * The core module named in args is dynamically imported
    * The random number seed is set from a value in args
    * args is overwritten by the checkpoint data

    Returns
    -------
    tuple[MarcoPolo Core, int]
        MarcoPolo Core: The core simulation object
        The int is the starting epoch number
    """
    cp_name = args.start_from
    tag = ""
    if "logtag" in args.__dict__:
        tag = args.logtag
    start_from = os.path.join(args.log_file, "Checkpoints", cp_name)

    logger.info(f"Starting from {start_from}")

    with open(os.path.join(start_from, "args.json"), mode="r", encoding="utf8") as file:
        args.__dict__ = json.load(file)
        if len(tag) > 0:
            args.logtag = f"[{tag}:{cp_name}]->"
        else:
            args.logtag = f"[{cp_name}]->"

    # update items before load if requested
    if preload_args is not None:
        for key in preload_args.__dict__:
            # Namespace doesn't support direct assignment
            setattr(args, key, preload_args[key])

    # Initialize marco polo population (env/team pairs) manager
    core = setup_marcopolo_core(args)

    #  Immediately load in stored state
    core.reload(folder=start_from)

    if print_args:
        logger.info(f"Args after reload: {core.args}")

    # Tweak to get the reload epoch
    start_epoch = int(start_from.split("-")[-1]) + 1  # Check point at end of epoch

    logger.info(
        f"########### Restarting From Checkpoint At Epoch {start_epoch} ############\n"
    )

    # Run Evolution
    return core, start_epoch


def get_args(config_file: Optional[str] = None) -> argparse.Namespace:
    """
    Parses command line arguments and input script and return the arguments

    Parameters
    ----------
    config_file: str, optional
        Path to config file to arguments from. If both this and
        --config are given, the file specified by --config is used.
    Returns
    -------
    argparse.Namespace
        object containing all the arguments
    """
    # Init argparser
    parser = argparse.ArgumentParser()

    # MarcoPolo parameters
    parser.add_argument("--log_file", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--core", default="ePoet")
    parser.add_argument("--init", default="random")
    parser.add_argument(
        "--visualize_freq",
        type=int,
        default=0,
        help="Frequency to store gifs of agent behavior. Use 0 for no gifs.",
    )
    parser.add_argument("--frame_skip", type=int, default=10)

    parser.add_argument("--eval_jobs", type=int, default=1)
    parser.add_argument("--rollouts_per_eval_job", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=20)
    parser.add_argument("--poet_epochs", type=int, default=200)
    parser.add_argument("--master_seed", type=int, default=111)

    parser.add_argument("--mc_lower", type=int, default=200)
    parser.add_argument("--mc_upper", type=int, default=340)
    parser.add_argument("--repro_threshold", type=int, default=200)
    parser.add_argument("--num_proposal_envs", type=int, default=8)
    parser.add_argument("--max_admitted_envs", type=int, default=1)
    parser.add_argument("--max_active_envs", type=int, default=100)
    parser.add_argument("--num_start_envs", type=int, default=1)

    parser.add_argument("--checkpoint_interval", type=int, default=1)
    parser.add_argument("--checkpoint_compression", default=True)
    parser.add_argument("--reproduction_interval", type=int, default=1)

    parser.add_argument("--start_from", default=None)  # Checkpoint folder to start from
    parser.add_argument("--save_env_params", default=True)
    parser.add_argument("--logtag", default="")
    parser.add_argument("--log_pata_ec", default=True)

    # Parse CMDLine args
    args = parser.parse_args()

    ## Load args from yaml if available
    if args.config is not None:
        # args.config overides the function argument
        config_file = args.config

    if config_file:
        args = read_config_from_file(args, config_file)

    # Log input params
    logger.info(f"{args}\n")

    return args


def read_config_from_file(
    args: argparse.Namespace, filename: str
) -> argparse.Namespace:
    """Read config options from file and update args object

    Pararmeters
    -----------
    args: argparse.Namespace
        the config arguments to update
    filename: str
        path of file to read arguments from

    Returns
    -------
    argparse.Namespace
        The updated args object
    """
    with open(filename, mode="r", encoding="utf-8") as file:
        logger.info(f"Loading config from {filename}\n")
        config = yaml.safe_load(file)

        for key, val in config.items():
            vars(args)[key] = val
    return args


def setup_video_directory(args: argparse.Namespace) -> str:
    """Creates Videos directory, if appropriate, and returns the path

    The path is formed from the log_file path
    Whether the directory is created depends on whether videos will
    be created. If args.visualize_freq is zero, it is assumed
    that videos will not be created and the directory will not be
    created.

    Parameters
    ----------
    args: argparse.Namespace
        the arguments for the run

    Returns
    -------
    str
        path to videos directory. Note that this is always returned
        even if the directory is not created.
    """
    # always create the path string, as it is returned
    vidpath = os.path.join(args.log_file, "Videos")
    # only create the actual directory if it will be used
    if args.visualize_freq != 0:
        logger.info(f"Saving video to {vidpath}\n")
        os.makedirs(name=vidpath, exist_ok=True)
    else:
        logger.info("Not saving video\n")
    return vidpath


###############################################################################
## Main CMDLine Function
###############################################################################
def main() -> None:
    """
    Main run function. This function parses cmdline arguments and then calls the
    appropriate function.

    Parameters
    ----------
    None

    Side-Effects
    ------------
    None

    Returns
    -------
    None
        All information is output to disk
    """

    # init start time
    sim_start_time = time.time()

    args = get_args()

    # define log folder when launched from VSCODE
    # maintain transparent compatibility with run_MP.sh launch script
    if "VSCODE_LAUNCHED" in os.environ:
        args.log_file = os.path.join(os.environ["OUTPUT_DIR"], os.environ["RUN_NAME"])

    # new run or restart?
    if args.start_from is not None:
        core, start_epoch = reload_core_from_checkpoint(args=args)
        core.evolve_population(start_epoch=start_epoch, sim_start=sim_start_time)
    else:
        core = setup_marcopolo_core(args=args)
        core.evolve_population(start_epoch=0, sim_start=sim_start_time)


###############################################################################
## Run
###############################################################################
if __name__ == "__main__":
    main()
