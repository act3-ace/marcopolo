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


"""Augmented Random Search (ARS) Optimization Manager"""

import time
import argparse
from collections import OrderedDict  # for type hinting
import json
import os
from typing import Optional, Type, Union  # for type hinting

from numpy.random import PCG64DXSM, Generator

from marco_polo.tools.types import Role, PathString  # for type hinting
from marco_polo.tools.wrappers import (
    EnvHistory,
    TeamHistory,
    RunTask,
)  # for type hinting
from marco_polo.optimizers.uber_es.manager import Manager as Base_Manager
from marco_polo.optimizers.ARS.modules.opt import ESTeamWrapper
from marco_polo.optimizers.ARS.modules.rollouts import run_optim_batch_distributed
from marco_polo.optimizers.uber_es.modules.opt import StepStats
from marco_polo.base_classes.serialCrew import SerialCrew
from marco_polo.tools.iotools import NumpyDecoder


#######################################################
## Factories
#######################################################
def get_opt_class(
    args: argparse.Namespace,
) -> Union[Type["Manager"], Type["SerialManager"]]:
    """Returns the class to use, based on input arguments

    Parameters
    ----------
    args : argparse.Namespace
        arguments that were passed to the `main()` function

    Returns
    -------
    Union[Type["Manager"], Type["SerialManager"]]
        the class to use in creating env objects
    """
    return Manager


################################################################################
## Main Class
################################################################################
class Manager(Base_Manager):
    """
    Compute Manager Class

    This class handles all compute by providing a standardized interface for
    optimization and evaluation. This manager subclasses the uber_es manager.
    """

    def __init__(self, setup_args: argparse.Namespace) -> None:
        """
        Manager Initializer

        Stores simulation args and sets up the compute pool.
        This primarily calls the super constructor from uber_es.

        Parameters
        ----------
        setup_args : args.Namespace
            Simulation parameters

        Side-Effects
        ------------
        None

        Returns
        -------
        None
        """

        # nothing new, just call parent
        super().__init__(setup_args=setup_args)

    ########################################
    ## Aux Funcs
    ########################################
    def reload(
        self,
        folder: PathString,
        cur_opts: OrderedDict[EnvHistory, TeamHistory],
    ) -> None:
        """
        Reload Objects from Disk

        Reads all objects to disk, calls the appropriate "Reload" functions
        for internal objects.

        Parameters
        ----------
        folder : PathString
            Directory to reload objects
        cur_opts : OrderedDict[EnvHistory, TeamHistory]
            Dictionary of active env/team pairs

        Side-Effects
        ------------
        Yes
            Creates all internal objects

        Returns
        -------
        None
        """
        # This is a copy from the uber_es manager
        #  The ESTeamsWrapper needed overwritten
        #  Adds some backwards compatability.

        # Manager directory
        if os.path.isdir(os.path.join(folder, "Manager")):
            folder = os.path.join(folder, "Manager")

        # Grab parameters
        optArcList = []
        with open(
            os.path.join(folder, "Manager.json"), mode="r", encoding="utf-8"
        ) as f:
            dct = json.load(f, cls=NumpyDecoder)
            self.np_random.bit_generator.state = dct["np_random"]
            if "optimArchive" in dct.keys():
                optArcList = dct["optimArchive"]

        # Build optimizers
        #  env.id was used for the archive, but the bracket uses the whole env
        #  so building a dictionary to translate env.id to env
        envIDDict = {k.id: k for k in cur_opts.keys()}
        for kv in optArcList:
            # grab team from active env/team dict
            team = cur_opts[envIDDict[kv[0]]]
            # safety - these should match
            assert kv[1] == team.id, "Team ID does not match optimizer archive."

            # build tmp optimizer
            tmpOpt = ESTeamWrapper(args=self.setup_args, team=team)
            # reload
            tmpOpt.reload(folder=os.path.join(folder, "".join(kv)))
            # store
            self.optimArchive[(kv[0], kv[1])] = tmpOpt

    ########################################
    ## Optim Funcs
    ########################################
    def optimize_step(
        self,
        tasks: list[RunTask],
        epoch: Optional[int] = None,
        opt_iter: Optional[int] = None,
    ) -> list[dict[Role, StepStats]]:
        """
        Single Optimization Step

        This function performs a single optimization step for each active environment
        and agent pair in "tasks". "epoch" and "opt_iter" are for logging only.
        This function passes work off to "start_chunk(), then grabs results
        asynchronously and processes them as they come in.

        Parameters
        ----------
        tasks : list[RunTask]
            List of tuple["EnvHistory", "TeamHistory"] pairs, where TeamHistory is a
            dictionary of agents indexed by their roles
        epoch : int, default=None
            Current simulation time
        opt_iter : int, default=None
            Which iteration in a loop we are on

        Side-Effects
        ------------
        Yes
            This function updates environments with the optimization stats, and
            the optimization updates the agents.

        Returns
        -------
        list[dict[Role, StepStats]]
            Returns a list of length "tasks" with statistics calculated in "ESAgt.combine_and_update()"
        """
        # setup new shared list
        job_handles = []
        start_times = []
        ret_list = []

        # loop over tasks
        for env, team in tasks:
            # check if optimizer already exists for task
            if (env.id, team.id) not in self.optimArchive:
                self.optimArchive[(env.id, team.id)] = ESTeamWrapper(
                    args=self.setup_args, team=team
                )

            # grab optimizer to continue task
            ESTeam = self.optimArchive[(env.id, team.id)]

            # pass off work
            job_handles.append(
                self.start_chunk(
                    # start_chunk args
                    eval_func=run_optim_batch_distributed,
                    num_jobs=self.setup_args.uber_es["optim_jobs"],
                    num_tasks_per_job=self.setup_args.uber_es["rollouts_per_optim_job"],
                    # eval_func args
                    env_params=env.env_param,
                    env_creator_func=env.get_env_creator(),
                    team=ESTeam,
                    setup_args=self.setup_args,
                    noise_std=self.setup_args.ARS["noise_std"],
                )
            )
            # append start time
            start_times.append(time.time())

        # loop over job handles, analyze
        for sT, jH, (env, team) in zip(start_times, job_handles, tasks):
            # get results for this env/agt optim
            task_results = [handle.get() for handle in jH]

            # grab esTeam
            ESTeam = self.optimArchive[(env.id, team.id)]

            # update esTeam
            stepstats = ESTeam.combine_and_update(
                step_results=task_results, step_t_start=sT
            )

            # update env
            self.optupdate(env=env, stats=stepstats)
            env.stats.iterations_lifetime += 1
            env.stats.iterations_current += 1

            # append results
            ret_list.append(stepstats)

        # return all results
        return ret_list


################################################################################
## Secondary Class
################################################################################
class SerialManager(SerialCrew, Manager):
    """Manager that uses a single process instead of multiprocessing

    This should be a drop in replacement for the Manager class.

    This is functionally different than using a Manager with one
    worker in that the async calls are direct calls to the specified
    function so debugging and profiling will work as expected.

    Generally this would only be used for testing.

    The class is intentionally blank. Using multi-inheritance, it has
    all the methods of the Manager class, except that it uses the
    get_crew() method from the SerialCrew class.
    """
