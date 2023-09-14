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


"""Tianshou Manager"""

import logging
import os
import pprint
import torch
import numpy as np
import argparse  # for type hinting
from tianshou.data import Batch, Collector, ReplayBuffer, VectorReplayBuffer  # type: ignore[import]
from tianshou.env import DummyVectorEnv, ShmemVectorEnv, SubprocVectorEnv  # type: ignore[import]
from tianshou.policy import MultiAgentPolicyManager  # type: ignore[import]
from tianshou.trainer import offpolicy_trainer, onpolicy_trainer  # type: ignore[import]

from collections import OrderedDict  # for type hinting
from collections.abc import Callable  # for type hinting
from typing import Any, Optional, Union, Type  # for type hinting

from marco_polo.tools.wrappers import (
    EnvHistory,
    TeamHistory,
    RunTask,
)  # for type hinting
from marco_polo.tools.types import Role, PathString  # for type hinting


from marco_polo.optimizers.uber_es.manager import Manager as Base_Manager
from marco_polo.tools.wrappers import MP_parallel_to_aec, MP_PettingZooEnv
from marco_polo.envs._base.env_params import EnvParams


logger = logging.getLogger(__name__)


#######################################################
## Factories
#######################################################
def get_opt_class(
    args: argparse.Namespace,
) -> Type["Manager"]:
    """Returns the class to use, based on input arguments

    Parameters
    ----------
    args: argparse.Namespace
        arguments that were passed to the `main()` function

    Returns
    -------
    class
        the class to use in creating env objects
    """
    return Manager


#######################################################
## Utilities
#######################################################
def flatten_dict_of_lists(dct: dict[Any, list[Any]]) -> list[Any]:
    """Flatten a dictionary of lists into a single list."""
    flattened_lst = []
    for key, value in dct.items():
        if isinstance(value, list):
            flattened_lst.extend(value)  # Append the list values to the flattened list
        else:
            flattened_lst.append(value)
    return flattened_lst


## Needs to be worked on - there's something else going
# on that's killing randomity...
class SeedingCollector(Collector):  # type: ignore[misc] # Collector is Any
    def collect(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        kwargs["gym_reset_kwargs"] = {"seed": 0}
        return super().collect(*args, **kwargs)  # type: ignore[no-any-return] # mypy can't see superclass


################################################################################
## Main Class
################################################################################
class Manager(Base_Manager):
    """
    Compute Manager Class

    This class handles all compute by providing a standardized interface for
    optimization and evaluation. This manager is based on tianshou and is under
    development.

    Attributes
    ----------
    setup_args : argparse.Namespace
        Seems to be like a dictionary
        This stores all of the simulation parameters
    np_random : Generator
        Numpy random generator
    crew : Multiprocessing Pool
        Worker pool for asynchronous evaluation

    Methods
    -------
    Manager(argparse.Namespace, list[Func])
        Initializer, stores functions, creates worker pool
    checkpoint(str)
        Store output
    reload(str)
        Reload output from checkpoint
    evalupdate(env, stat)
        Pass eval stats to the appropriate environment update function
    optupdate(env, stat)
        Pass optimization stats to the appropriate environment update function
    start_chunk(func, int, int, **kwargs)
        Passes func off to worker pool, returns job handles
    evaluate(list[(env, team)], int, bool)
        Main evaluation function
    _visualize(list[(env, team)], int)
        Non-blocking video generating function
    optimize_step(list[(env, team)], int, int)
        Single optimization step
    optimize_chunk(list[(env, team)], int, bool)
        Main optimization function, with evals and video generation
    """

    def __init__(self, setup_args: argparse.Namespace) -> None:
        """
        Manager Initializer

        Stores simulation args and sets up the compute pool

        Parameters
        ----------
        setup_args : args.NAmespace
            Simulation parameters

        Side-Effects
        ------------
        None

        Returns
        -------
        None
        """
        # torch sees through pods, and schedules based on the blade resources
        # this lets us control how many resources torch claims
        torch.manual_seed(setup_args.master_seed)
        CORES = setup_args.tianshou["torch_threads"]
        torch.set_num_threads(CORES)
        torch.set_num_interop_threads(CORES)
        super().__init__(setup_args=setup_args)

    ########################################
    ## Aux Funcs
    ########################################
    def checkpoint(self, folder: PathString, cur_opts: list[RunTask]) -> None:
        pass

    def reload(
        self, folder: PathString, cur_opts: OrderedDict[EnvHistory, TeamHistory]
    ) -> None:
        pass

    ########################################
    ## Optim Funcs
    ########################################
    def optimize_step(
        self,
        tasks: list[RunTask],
        epoch: Optional[int] = None,
        opt_iter: Optional[int] = None,
    ) -> list[dict[Role, list[Any]]]:
        # idk what is in the list at the bottom
        #  maybe floats?
        """ """
        return self.optimize_chunk(tasks=tasks, epoch=1)

    def _create_env(
        self,
        args: argparse.Namespace,
        env_creator: Callable[..., Any],
        env_params: EnvParams,
        **kwargs: dict[str, Any],
    ) -> MP_PettingZooEnv:
        env = env_creator(**kwargs)
        env.augment(params=env_params)
        env.seed(seed=self.np_random.integers(low=2**31 - 1, dtype=int))
        return MP_PettingZooEnv(args, MP_parallel_to_aec(env))

    def optimize_chunk(
        self,
        tasks: list[RunTask],
        epoch: Optional[int] = None,
        verbose: Optional[bool] = False,
    ) -> list[dict[Role, list[Any]]]:
        results = []
        args = self.setup_args

        epoch_to_run = args.tianshou["epoch"]

        returns = [{k: [] for k in agt.keys()} for (_, agt) in tasks]

        for i, (env, agts) in enumerate(tasks):
            logger.info(f"Optimizing team: {agts.id} on env: {env.id}")

            # if (env.id, agts.id) in self.optimArchive.keys():
            #     agts = self.optimArchive[(env.id, agts.id)]

            env_creator = env.get_env_creator()

            # print(agts.keys())
            # if len(agts.keys())>1:
            tmp_env = self._create_env(
                args=self.setup_args, env_creator=env_creator, env_params=env.env_param
            )
            agts_list = [agts[k] for k in tmp_env.agents]
            policy = MultiAgentPolicyManager(agts_list, tmp_env)
            agents = tmp_env.agents

            train_envs = SubprocVectorEnv(
                [
                    lambda: self._create_env(
                        args=self.setup_args,
                        env_creator=env_creator,
                        env_params=env.env_param,
                    )
                    for _ in range(args.tianshou["train_num"])
                ]
            )

            test_envs = SubprocVectorEnv(
                [
                    lambda: self._create_env(
                        args=self.setup_args,
                        env_creator=env_creator,
                        env_params=env.env_param,
                    )
                    for _ in range(args.tianshou["test_num"])
                ]
            )

            if args.tianshou["train_num"] > 1:
                buffer = VectorReplayBuffer(
                    args.tianshou["step_per_collect"], len(train_envs)
                )
            else:
                buffer = ReplayBuffer(args.tianshou["step_per_collect"])

            train_collector = SeedingCollector(
                policy, train_envs, buffer, exploration_noise=True
            )

            # At some point we may want to write a test collector
            reward_list: dict[Role, dict[Any, Any]] = {
                agt: dict() for agt in tmp_env.agents
            }
            steps: dict[str, list[int]] = dict()

            ####################
            ## Inner Funcs
            ####################
            def save_best_fn(policy) -> None:
                torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

            def setup_eval_collect(num_epoch: int, step_idx: int) -> None:
                pass
                # reward_list.append(dict())

            def save_collecter_rewards(**kwargs: Any) -> dict[Any, Any]:
                returns = dict()
                ## Grab the reward and save it to a external buffer
                env_id = kwargs["env_id"][0]

                for rl in reward_list.values():
                    if not env_id in rl.keys():
                        rl[env_id] = [0]

                if not env_id in steps.keys():
                    steps[env_id] = [0]

                if "rew" not in kwargs.keys():
                    ## At beg/end of run, set up next storage
                    ## only if positve number of steps
                    if steps[env_id][-1] > 0:
                        for rl in reward_list.values():
                            rl[env_id].append(0)

                        steps[env_id].append(0)
                else:
                    agt = kwargs["obs_next"].agent_id[0]
                    steps[env_id][-1] += 1
                    reward_list[agt][env_id][-1] += kwargs["rew"][0][0]

                return returns

            ####################
            ## End Inner Funcs
            ####################
            test_collector = SeedingCollector(
                policy, test_envs, preprocess_fn=save_collecter_rewards
            )

            # trainer
            result = onpolicy_trainer(
                policy,
                train_collector,
                test_collector,
                epoch_to_run,
                args.tianshou["step_per_epoch"],
                args.tianshou["repeat_per_collect"],
                args.tianshou["test_num"],
                args.tianshou["batch_size"],
                step_per_collect=args.tianshou["step_per_collect"],
                test_fn=setup_eval_collect,
                # save_best_fn=save_best_fn,
                # logger=logger,
                test_in_train=False,
            )

            pprint.pprint(result)

            # print(reward_list, flatten_dict_of_lists(steps))

            flat_steps = np.array(flatten_dict_of_lists(steps))
            for agt, rwds_ in reward_list.items():
                rwds = np.array(flatten_dict_of_lists(rwds_))
                rwd_value = np.mean(rwds[flat_steps > 0])
                returns[i][agt].append(rwd_value)

            # self.optimArchive[(env.id, agts.id)] = agts
            logger.info(f"Finished Optimizing team: {agts.id} on env: {env.id}")

        return returns
