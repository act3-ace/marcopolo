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


"""Uber ES Optimization Manager"""

import json
import logging
import multiprocessing as mp
import os
import time
import argparse  # for type hinting
from collections import OrderedDict  # for type hinting
from collections.abc import Callable  # for type hinting
from multiprocessing.pool import AsyncResult  # for type hinting
from typing import Any, Optional, Type, Union  # for type hinting

import numpy as np
from numpy.random import PCG64DXSM, Generator

from marco_polo.optimizers.uber_es.modules.opt import ESTeamWrapper, StepStats
from marco_polo.optimizers.uber_es.modules.result_objects import EvalResult
from marco_polo.optimizers.uber_es.modules.rollouts import (
    run_eval_batch_distributed,
    run_optim_batch_distributed,
    run_viz_distributed,
)
from marco_polo.base_classes.serialCrew import SerialCrew
from marco_polo.tools.iotools import NumpyDecoder, NumpyEncoder, Telemetry, TBWriter
from marco_polo.tools.types import EnvId, Role, TeamId, PathString  # for type hinting
from marco_polo.tools.wrappers import EnvHistory, TeamHistory, RunTask

logger = logging.getLogger(__name__)


#######################################################
## Factories
#######################################################
def get_opt_class(
    args: argparse.Namespace,
) -> Union[Type["Manager"], Type["SerialManager"]]:
    """Returns the class to use, based on input arguments

    Parameters
    ----------
    args: argparse.Namespace
        arguments that were passed to the `main()` function

    Returns
    -------
    Union[Type["Manager"], Type["SerialManager"]]
        the class to use in creating env objects
    """
    if getattr(args, "parallel", True):
        return Manager
    return SerialManager


################################################################################
## Main Class
################################################################################
class Manager:
    """
    Compute Manager Class

    This class handles all compute by providing a standardized interface for
    optimization and evaluation. This manager is based on the multiprocessing
    asynchronous pool.
    """

    def __init__(self, setup_args: argparse.Namespace) -> None:
        """
        Manager Initializer

        Stores simulation args and sets up the compute pool

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

        self.setup_args = setup_args
        self.np_random = Generator(PCG64DXSM(seed=setup_args.master_seed))

        # initialize worker pool
        self.crew = self.get_crew()

        # dictionary to store optimizers between optimization chunks
        # key is (env.id, team.id)
        self.optimArchive: dict[tuple[EnvId, TeamId], ESTeamWrapper] = {}

        # Tensorboard:
        self.Telemetry = Telemetry()
        if hasattr(setup_args, "telemetry"):
            t_args = setup_args.telemetry
            if "Tensorboard" in t_args.keys():
                tb = TBWriter(t_args["Tensorboard"])
                self.Telemetry.add_logger(tb)

        # # Saving this as notes
        # #  This is how to share objects between workers in a parallel pool.
        # #  It still copies complex objects, so think of this more as a safety/convenience
        # #  idea than a performance idea.
        # # create manager for this context
        # manager = mp_ctx.Manager()
        # self.manager = manager
        # # This is pedantic
        # #  The recommendation is to keep all shared objects together for programmer
        # #  convenience, therefore, POET built a dict of them.
        # self.fiber_shared = {
        #         "envs": manager.dict(),
        #         "agents": manager.dict(),
        #         "tasks": manager.dict(),
        # }
        # # setup workers, provide reference to the shared objects
        # #  This works in tandem with a "global" call within initialize_worker_fiber
        # #  It's not the standard way of passing shared args to workers.
        # self.crew = mp_ctx.Pool(setup_args.num_workers, initializer=initialize_worker_fiber,
        #         initargs=(self.fiber_shared["agents"],
        #                   self.fiber_shared["envs"],
        #                   self.fiber_shared["tasks"],))

    ########################################
    ## Aux Funcs
    ########################################
    def get_crew(self) -> mp.pool.Pool:
        """Return a Pool of processes used to run tasks

        Parameters
        ----------
        None

        Side-Effects
        ------------
        None

        Returns
        -------
        mp.Pool
            Pool of workers to run tasks
        """
        # Get a specific context
        #  this avoids changing defaults in MP
        mp_ctx = mp.get_context("spawn")

        return mp_ctx.Pool(processes=self.setup_args.num_workers)

    def checkpoint(self, folder: PathString, cur_opts: list[RunTask]) -> None:
        """
        Store Objects to Disk

        Writes all objects to disk, calls the appropriate "checkpoint" functions
        for internal objects.

        Parameters
        ----------
        folder : PathString
            Directory to store objects
        cur_opts : list[RunTask]
            List of Tuples of currently active EnvHistory/TeamHistory pairs

        Side-Effects
        ------------
        Yes
            Cleans the optimizer archive before storing

        Returns
        -------
        None
            All objects are saved to disk
        """
        # clean dictionary
        #  Last-minute cleaning so we don't bloat the checkpoints
        #  This assume "cur_opts" is the current active set of env/team pairs
        #  If that's not true, this will fail
        self._clean_archive(cur_opts=cur_opts)

        tmp: dict[str, Any] = {"np_random": self.np_random.bit_generator.state}
        tmp["optimArchive"] = list(self.optimArchive.keys())

        # Manager directory
        folder = os.path.join(folder, "Manager")
        os.makedirs(folder, exist_ok=True)

        # store params
        with open(
            os.path.join(folder, "Manager.json"), mode="w", encoding="utf-8"
        ) as f:
            json.dump(tmp, f, cls=NumpyEncoder)

        # log optim archive
        for k, v in self.optimArchive.items():
            # setup/create directory
            opt_folder = os.path.join(folder, "".join(k))
            os.makedirs(opt_folder, exist_ok=True)
            # save optimizer
            v.checkpoint(folder=opt_folder)

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
            Dictionary of active EnvHistory/TeamHistory pairs

        Side-Effects
        ------------
        Yes
            Creates all internal objects

        Returns
        -------
        None
        """
        ## Adds some backwards compatability.
        # Manager directory
        if os.path.isdir(os.path.join(folder, "Manager")):
            folder = os.path.join(folder, "Manager")

        # grab parameters
        optArcList = []
        with open(
            os.path.join(folder, "Manager.json"), mode="r", encoding="utf-8"
        ) as f:
            dct = json.load(f, cls=NumpyDecoder)
            self.np_random.bit_generator.state = dct["np_random"]
            if "optimArchive" in dct.keys():
                optArcList = dct["optimArchive"]

        # build optimizers
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

    # def evalupdate(self, env: EnvHistory, stats) -> None:
    #     """
    #     Update Environment with Evaluation Data

    #     Calls the appropriate environment update function with evaluation statistics.

    #     Parameters
    #     ----------
    #     env : Gym.environment
    #         The simulation environment
    #     stats : EvalResult
    #         EvalResult object

    #     Side-Effects
    #     ------------
    #     Possibly
    #         If the environment has an update function, it is called and changes
    #         things within each environment

    #     Returns
    #     -------
    #     None
    #     """
    #     ## check if env has update function
    #     if hasattr(env.__class__, "evalupdate") and callable(
    #         getattr(env.__class__, "evalupdate")
    #     ):
    #         env.evalupdate(stats)

    def optupdate(self, env: EnvHistory, stats: dict[Role, StepStats]) -> None:
        """
        Update Environment with Optimization Data

        Calls the appropriate environment update function with optimization statistics.

        Parameters
        ----------
        env : EnvHistory
            The simulation environment
        stats : dict[Role, StepStats]
            Whatever is returned from the optimizer update

        Side-Effects
        ------------
        Possibly
            If the environment has an update function, it is called and changes
            things within each environment

        Returns
        -------
        None
        """
        ## check if env has update function
        if hasattr(env.__class__, "optupdate") and callable(
            getattr(env.__class__, "optupdate")
        ):
            env.optupdate(stats)

    def _clean_archive(self, cur_opts: list[RunTask]) -> None:
        """
        Remove Old Entries From Optimizer Dictionary

        This function keeps the optimizer dictionary from growing without bounds.
        It makes sure that only the active EnvHistory/TeamHistory pairings are maintained.

        Parameters
        ----------
        cur_opts : list[RunTask]
            List of tuples of current EnvHistory/TeamHistory pairings

        Side-Effects
        ------------
        Yes
            Cleans all items from optimArchive except those in cur_opts

        Returns
        -------
        None
        """
        # new archive object
        newArchive = dict()

        # loop through things to save and add to dictionary
        for key, val in cur_opts:
            new_key = (key.id, val.id)
            if new_key in self.optimArchive:
                newArchive[new_key] = self.optimArchive[new_key]

        # replace
        self.optimArchive = newArchive

    ########################################
    ## Eval Funcs
    ########################################
    def start_chunk(
        self,
        eval_func: Callable[..., Any],
        num_jobs: int,
        num_tasks_per_job: int,
        **kwargs: Any,
    ) -> list[AsyncResult[Any]]:
        """
        Multi-processing Handoff Function

        This function passes "eval_func" to the multiprocessing pool for asynchronous
        evaluation. It sets up the appropriate seed, passes "num_job" copies of
        "eval_func", and then passes arguments.

        Parameters
        ----------
        eval_func : Callable[..., Any]
            Function to run on pool
        num_jobs : int
            Number of tasks to start
        num_tasks_per_job : int
            Size of each job, generally refers to simulation rollouts
        **kwargs : Any
            Additional args to pass to functios

        Side-Effects
        ------------
        None
            May write videos to disk, but all objects are copied and not returned,
            so no internal changes from these calls

        Returns
        -------
        list[AsyncResult[Any]]
            List of async objects. Results of function can be gotten with asyc.get()
        """
        # debug logs
        logger.debug(f"Spawning {num_jobs} batches of size {num_tasks_per_job}")

        # pull seeds for rollouts
        rs_seeds = self.np_random.integers(
            low=2**31 - 1, size=num_jobs, dtype=np.int32
        )

        # return obj
        chunk_tasks = []

        # put tasks on pool
        #  append task handles to return obj
        for i in range(num_jobs):
            chunk_tasks.append(
                self.crew.apply_async(
                    func=eval_func, args=(num_tasks_per_job, rs_seeds[i]), kwds=kwargs
                )
            )

        return chunk_tasks

    def evaluate(
        self, tasks: list[RunTask], epoch: Optional[int] = None, verbose: bool = False
    ) -> list[dict[Role, EvalResult]]:
        """
        Main Evaluation Function

        This function runs all "tasks" provided. "epoch" and "verbose" are for
        logging purposes only. This function passes evaluation off to "start_chunk",
        then asynchronously gets results and processes them.

        Parameters
        ----------
        tasks : list[(RunTask]
            List of EnvHistory and the TeamHistory to run on them
        epoch : Optional[int], default=None
            Current simulation time, for logging only
        verbose : bool, default=False
            Log to console?

        Side-Effects
        ------------
        None
            All objects are copied and not returned, so no internal changes

        Returns
        -------
        list[dict[role, EvalResult]]
            List the same length as "tasks", with a dictionary holding the results
            of each agent on a team
        """
        # start time
        eval_time = time.time()

        # job handle list
        eval_handles = []

        # assign jobs
        for env, team in tasks:
            eval_handles.append(
                self.start_chunk(
                    # start_chunk args
                    eval_func=run_eval_batch_distributed,
                    num_jobs=self.setup_args.eval_jobs,
                    num_tasks_per_job=self.setup_args.rollouts_per_eval_job,
                    # eval_func args
                    env_params=env.env_param,
                    env_creator_func=env.get_env_creator(),
                    team=team,
                )
            )

        # return object
        combined_results = []

        # loop through tasks
        # for task in eval_results_done:
        for task_handles, (env, team) in zip(eval_handles, tasks):
            # get task returns for this task
            #  We have to get all of these 'cause then we cycle through the roles
            task = [job.get() for job in task_handles]

            # return obj
            combined = dict()

            # Loop through roles
            for role in team.roles():
                # return objects for this role
                returns_ = []
                lengths_ = []

                # loop through this role's results from all tasks
                for job in task:
                    returns_.extend(job[role].returns)
                    lengths_.extend(job[role].lengths)

                # reshape list as numpy object
                returns = np.array(returns_)
                lengths = np.array(lengths_)

                # create eval result for this role
                #  recompute metrics
                combined[role] = EvalResult(
                    returns=returns,
                    lengths=lengths,
                    eval_returns_mean=returns.mean(),
                    eval_returns_max=returns.max(),
                    eval_returns_min=returns.min(),
                )

            # All roles are finished, put into final return object
            combined_results.append(combined)

        # if verbose, log results to terminal
        if verbose:
            for task_result, (env, team) in zip(combined_results, tasks):
                for role, stats in task_result.items():
                    logger.info(
                        f"Epoch={epoch} - Environment={env.id} - "
                        f"Team={team.id} - Role={role} - "
                        f"Eval_mean={stats.eval_returns_mean} - "
                        f"Best_eval={stats.eval_returns_max} - "
                        f"Lengths={stats.lengths} - "
                        f"Iterations_lifetime={env.stats.iterations_lifetime} - "
                        f"Iterations_current={env.stats.iterations_current}"
                    )

        # log
        logger.debug(
            f"evaluate() Time to complete evaluation: {time.time() - eval_time}"
        )

        # return results list
        return combined_results

    def _visualize(self, tasks: list[RunTask], epoch: Optional[int] = None) -> None:
        """
        Hidden Vizualization Function

        This function is specifically for simulation vizualization. It is a blocking
        function with no return. It runs all "tasks" and outputs .gifs for each.
        This function is used in "optimize_chunk()"

        Parameters
        ----------
        tasks : list[RunTask]
            List of envHistory and the TeamHistory to run on them
        epoch : Optional[int], default=None
            Current simulation time, for logging only

        Side-Effects
        ------------
        None
            All objects are copied and not returned, so no internal changes

        Returns
        -------
        None
        """
        # pass off tasks
        vis_handles = []
        for env, team in tasks:
            vis_handles.extend(
                self.start_chunk(
                    # start_chunk args
                    eval_func=run_viz_distributed,
                    num_jobs=1,
                    num_tasks_per_job=1,
                    # eval_func args
                    env_params=env.env_param,
                    env_creator_func=env.get_env_creator(),
                    team=team,
                    env_id=env.id,
                    epoch=epoch,
                    vidpath=self.setup_args.vidpath,
                    frame_skip=self.setup_args.frame_skip,
                )
            )

        # Wait for these to complete, or else the program can end before saving images
        _ = [job.get() for job in vis_handles]

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
        and team pair in "tasks". "epoch" and "opt_iter" are for logging only.
        This function passes work off to "start_chunk(), then grabs results
        asynchronously and processes them as they come in.

        Parameters
        ----------
        tasks : list[(RunTask]
            List of tuple of EnvHistory/TeamHistory pairs
        epoch : Optional[int], default=None
            Current simulation time
        opt_iter : Optional[int], default=None
            Which iteration in a loop we are on

        Side-Effects
        ------------
        Yes
            This function updates environments with the optimization stats, and
            the optimization updates the agents.

        Returns
        -------
        list[dict[Role, StepStats]]
            Returns a list of length "tasks" with statistics calculated in "combine_and_update()"
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
            es_team = self.optimArchive[(env.id, team.id)]

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
                    team=es_team,
                    setup_args=self.setup_args,
                    noise_std=self.setup_args.uber_es["noise_std"],
                )
            )
            # append start time
            start_times.append(time.time())

        # loop over job handles, analyze
        for sT, jH, (env, team) in zip(start_times, job_handles, tasks):
            # get results for this env/team optim
            task_results = [handle.get() for handle in jH]

            # grab optimizer
            es_team = self.optimArchive[(env.id, team.id)]

            # update optimizers
            stepstats = es_team.combine_and_update(
                step_results=task_results,
                step_t_start=sT,
                decay_noise=True,
                propose_only=False,
            )

            # update env
            self.optupdate(env=env, stats=stepstats)
            env.stats.iterations_lifetime += 1
            env.stats.iterations_current += 1

            # append results
            ret_list.append(stepstats)

        # return all results
        return ret_list

    def optimize_chunk(
        self, tasks: list[RunTask], epoch: Optional[int] = None, verbose: bool = False
    ) -> list[dict[Role, list[float]]]:
        """
        Multiple Optimization Step, Evaluation, and Video Generation

        This function performs several optimization steps, as well as evaluation
        of the optimized teams after each optimization. When appropriate, it
        generates videos of agent performance as well.

        Parameters
        ----------
        tasks : list[RunTask]
            List of tuples of EnvHistory/TeamHistory pairs
        epoch : Optional[int], default=None
            Current simulation time
        verbose : bool, default=False
            Log results to console?

        Side-Effects
        ------------
        Yes
            This function calls "optimize_step()", which updates environments with
            the optimization stats, and the optimization updates the agents.

        Returns
        -------
        list[dict[Role, list[float]]]
            Returns a list of length "tasks" with evaluation means for each team
        """
        # Start time
        opt_time = time.time()

        # list of dict{key: list} for each agent Role in TeamHistory
        returns: list[dict[Role, list[float]]] = [
            {role: [] for role in team.roles()} for (_, team) in tasks
        ]

        # loop over optimization iterations
        for i in range(self.setup_args.uber_es["optim_iters"]):
            # log iter
            logger.info(f"optim_iter {i+1} of {self.setup_args.uber_es['optim_iters']}")

            # single optimization step
            _ = self.optimize_step(tasks=tasks, epoch=epoch, opt_iter=i)

            # eval for one opt
            eval_results = self.evaluate(tasks=tasks, epoch=epoch, verbose=verbose)

            for j, (env, team) in enumerate(tasks):
                for role in team.roles():
                    returns[j][role].append(eval_results[j][role].eval_returns_mean)

                self.Telemetry.write_items(
                    name=f"{env.id}:{team.id}:Scores",
                    data=returns[j],
                    step=epoch * self.setup_args.uber_es["optim_iters"] + i,
                )

                # TODO: Does this actually work?
                #       Seems like "role" will just be the last agent in the team
                L = eval_results[j][role].lengths
                self.Telemetry.write_items(
                    name=f"{env.id}:{team.id}:Lengths",
                    data={
                        "mean": np.mean(L),
                        "+std": np.mean(L) + np.std(L),
                        "-std": np.mean(L) - np.std(L),
                        "max": np.max(L),
                        "min": np.min(L),
                    },
                    step=epoch * self.setup_args.uber_es["optim_iters"] + i,
                )

        # check for visualization
        if (
            self.setup_args.visualize_freq > 0
            and epoch % self.setup_args.visualize_freq == 0
        ):
            self._visualize(tasks=tasks, epoch=epoch)

        # clean dictionary
        #  this isn't strictly necessary here, it just needs to run regularly
        #  Doing it here keeps it within the Manager object
        #  This assume "tasks" is the current active set of env/team pairs
        #  If that's not true, this will fail
        self._clean_archive(cur_opts=tasks)

        # Log total time
        logger.debug(f"optimize_chunk() Full Optstep Time: {time.time() - opt_time}")

        # return list of dict of list of mean eval result
        return returns


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
