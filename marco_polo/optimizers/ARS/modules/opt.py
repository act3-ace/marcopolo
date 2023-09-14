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


"""ARS Team Wrapper and Optimization Controller"""

import json
import logging
import os
import time
from typing import Any, Union  # for type hinting
import argparse  # for type hinting

import numpy as np
from numpy.random import PCG64DXSM, Generator

from marco_polo.optimizers.uber_es.modules.opt import ESTeamWrapper as Base_TeamWrapper
from marco_polo.optimizers.uber_es.modules.opt import StepStats
from marco_polo.tools.stats import compute_weighted_sum

from marco_polo.optimizers.ARS.modules.result_objects import (
    POResult,
)  # for type hinting
from marco_polo.optimizers.ARS.modules.filter import Filter  # for type hinting
from marco_polo.tools.types import FloatArray, Role, PathString  # for type hinting
from marco_polo.tools.wrappers import TeamHistory  # for type hinting
from marco_polo.tools.iotools import NumpyDecoder


logger = logging.getLogger(__name__)


################################################################################
## Aux Objects
################################################################################


################################################################################
## Main Class
################################################################################
class ESTeamWrapper(Base_TeamWrapper):
    """Wrapper for the optimizers that apply to a given team

    By default, the neural network models of all agents are updated by
    the methods in this class. However, a set of roles can be marked
    as 'frozen' to prevent the agents from being updated. This can be
    done via the parameter file using the key uber_es.freeze.
    Alternatively, the list of frozen roles can be changed later
    (see add_freeze(), remove_freeze(), and remove_all_freeze())
    """

    def __init__(self, args: argparse.Namespace, team: TeamHistory) -> None:
        """ES Team Wrapper for ARS Optimizer

        Parameters
        ----------
        args : argparse.Namespace
            Input arguments to parameterize Team and Optimizer
        team : TeamHistory
            Team of agents

        Returns
        -------
        None
        """
        self.team = team
        self.args = args

        # see if freeze key is in args
        try:
            self._freeze = set(args.uber_es["freeze"])
        except (AttributeError, KeyError):
            self._freeze = set()

        # build optimizer dict
        self.optimizers = {}
        for role in team.roles():
            self.optimizers[role] = ARS(args=args)

    def combine_and_update(
        self,
        step_results: list[dict[Role, POResult]],
        step_t_start: float,
        decay_noise: bool = False,
        propose_only: bool = False,
    ) -> dict[Role, StepStats]:
        """Combine results from rollouts, pass to the optimizer, and update models.

        Agents with roles that are listed in the freeze set will have the statistics
        calculated but will not have their models updated.

        Parameters
        ----------
        step_results : list[dict[Role, POResult]]
            Results from runs
        step_t_start : float
            Time when this set of steps started (used for calculating runtime)
        decay_noise : bool, default=True
            Whether to have the applied noise decay during the optimization
        propose_only : bool, default=False
            If true, only the stats will be calculated and no updates will be made

        Return
        ------
        dict[Role, StepStats]
            Statistics from the runs for each role
        """
        # return dict
        #  organize by agent key
        stats = dict()

        # loop over agents, update their network
        for role in self.team.roles():
            # grab sim results for this agent
            results = [r[role] for r in step_results]

            # freeze is implemented by using propose_only
            if role in self._freeze:
                _propose_only = True
            else:
                _propose_only = propose_only

            # calculate stats and new theta
            new_theta, stats[role] = self.optimizers[role].combine_steps(
                step_results=results,
                theta=self.team[role].theta,
                obs_filt=self.team[role].get_obs_filt(),
                step_t_start=step_t_start,
                propose_only=_propose_only,
            )

            # update agent
            if not _propose_only:
                self.team[role].update_theta(new_theta=new_theta)
                self.team[role].set_model_params(model_params=new_theta)
                # obs_filt was updated in-place
                self.team[role].get_obs_filt().update_stats()
            self.team[role].get_obs_buf().reset()  # should be redundent

        # return results
        return stats


################################################################################
## Optimization Controller
################################################################################
class ARS:
    """
    Augmented Random Search

    Algorithm V2-t from https://arxiv.org/abs/1803.07055
    """

    def __init__(self, args: argparse.Namespace) -> None:
        """


        Parameters
        ----------
        args : argparse.Namespace
            Input arguments


        Side-Effects
        ------------
        Yes
            Sets class attributes

        Return
        ------
        None
        """
        #
        self.alpha = args.ARS["learning_rate"]  # learning-rate/step-size
        self.noise_std = args.ARS["noise_std"]  # std deviation of the exploration noise

        # check top fraction to keep, then convert to int for use
        assert args.ARS["top_frac"] <= 1, "top_frac must be <= 1"
        assert args.ARS["top_frac"] >= 0, "top_frac must be >= 0"
        self.top_frac = args.ARS["top_frac"]
        self.top_n = int(
            round(
                args.ARS["top_frac"]
                * args.uber_es["optim_jobs"]
                * args.uber_es["rollouts_per_optim_job"]
            )
        )

    def __str__(self) -> str:
        return "ARS Optimization Controller"

    def checkpoint(self, folder: PathString, role: Role) -> None:
        """Save ARS Optimizer to Disk

        Parameters
        ----------
        folder : PathString
            Directory to store output
        role : Role
            Agent role for file naming

        Side-Effects
        ------------
        None

        Return
        ------
        None
            Saves output to disk
        """
        with open(
            os.path.join(folder, f"{role}_opt.json"), mode="w", encoding="utf-8"
        ) as f:
            json.dump(self.__dict__, f)

    def reload(self, folder: PathString, role: Role) -> None:
        """Load ARS Optimizer From Disk

        Parameters
        ----------
        folder : PathString
            Directory to read input
        role : Role
            Agent role for file naming

        Side-Effects
        ------------
        Yes
            Sets optimizer attributes

        Return
        ------
        None
        """
        with open(
            os.path.join(folder, f"{role}_opt.json"), mode="r", encoding="utf-8"
        ) as f:
            self.__dict__ = json.load(f, cls=NumpyDecoder)

    def get_noise(self, seed: int, theta_len: int) -> FloatArray:
        """Generate Random Noise for Gradient Estimation

        This function generates exploratory noise for simulation rollouts.
        The noise follows a standard normal distribution.

        Parameters
        ----------
        seed : int
            Initial value for the random generator.
        theta_len : int
            How many random numbers to return

        Side-Effects
        ------------
        None

        Return
        ------
        FloatArray
            Array of random floats of length theta
        """
        # set random state
        random_state = Generator(PCG64DXSM(seed=seed))
        # draw normal distribution
        return random_state.normal(scale=self.noise_std, size=theta_len)

    def calc_max_min_theta(
        self, noise_seeds: FloatArray, returns: FloatArray, theta: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        """
        Holdover from the original POET Code

        Poet calculated max/min theta values after the update, but differently from
        how it actually calculates the gradient. I'm separating it out, adding a min,
        and then keeping it for historic purposes only.

        Parameters
        ----------
        noise_seeds : FloatArray
            Array of seed values used to generate the random noise
        returns : FloatArray
            Rewards values from the simulation rollouts
        theta : FloatArray
            Current parameterization of the model

        Side-Effects
        ------------
        None

        Returns
        -------
        tuple[FloatArray, FloatArray]
            Min/Max point estimates of network parameterizations
        """
        # used for both
        theta_len = len(theta)
        retList: list[FloatArray] = []

        # loop over max/min
        for i in range(0, 2):
            # setup
            pos_row, neg_row = (
                returns.argmax(axis=0) if (i == 0) else returns.argmin(axis=0)
            )
            noise_sign = 1.0
            noise_seed = noise_seeds[pos_row]

            # check
            if returns[pos_row, 0] < returns[neg_row, 1]:
                noise_sign = -1.0
                noise_seed = noise_seeds[neg_row]

            # calculate
            retList.append(
                theta
                + noise_sign * self.get_noise(seed=noise_seed, theta_len=theta_len)
            )

        return retList[0], retList[1]

    def combine_steps(
        self,
        step_results: list[POResult],
        theta: FloatArray,
        obs_filt: Filter,
        step_t_start: float,
        propose_only: bool = False,
    ) -> tuple[FloatArray, StepStats]:
        """Calculate Statistics from All Simulation Rollouts

        If "propose_only" is true, then only statistics are calculated but no
        updates are performed.


        Parameters
        ----------
        step_results : list[POResult]
            Results from the simulation rollouts
        theta : FloatArray
            Current network parameterization
        obs_filt : Filter
            The model observation filter
        step_t_start : float
            Time that this calculation started
        propose_only : bool, default=False
            Do updates or only calculate statistics?

        Side-Effects
        ------------
        Yes
            Updates the observation filter for the model

        Returns
        -------
        tuple[FloatArray, StepStats]
            Tuple containing the new network parameterization and statistics from
            the simulations
        """
        # constants
        theta_len = len(theta)

        # Extract results
        nList = []
        rList = []
        lList = []
        oList = []
        for r in step_results:
            nList.append(r.noise_inds)
            rList.append(r.returns)
            lList.append(r.lengths)
            oList.append(r.obs_buf)

        # reshape results
        noise_inds = np.concatenate(nList)
        po_returns = np.concatenate(rList)
        po_lengths = np.concatenate(lList)
        po_obs = np.concatenate(oList)

        # sort and subset results
        #  get max of +/- rollouts
        #  partition top n - idk why the "-" in the kth position
        #  grab just the top n
        top_n_idx = po_returns.max(axis=1).argpartition(kth=-self.top_n)[-self.top_n :]

        noise_inds = noise_inds[top_n_idx]
        po_returns = po_returns[top_n_idx, :]
        po_lengths = po_lengths[top_n_idx]
        po_obs = po_obs[top_n_idx]

        # get max/min possible grads
        #  has to be after subsetting results, but before theta update
        po_theta_max, po_theta_min = self.calc_max_min_theta(
            noise_seeds=noise_inds, returns=po_returns, theta=theta
        )

        # calculate gradients and update theta
        ret_std = po_returns.std()
        w_sum, _ = compute_weighted_sum(
            weights=(po_returns[:, 0] - po_returns[:, 1]) / ret_std,
            vec_generator=(
                self.get_noise(seed=seed, theta_len=theta_len) for seed in noise_inds
            ),
            theta_size=theta_len,
        )

        # Calculate gradient
        grads = w_sum * self.alpha / self.top_n  # need for logging

        # are we updating?
        if not propose_only:
            # update theta
            theta += grads

            # update observations
            for obs in po_obs:
                obs_filt.update(other=obs)

        # grab final time
        step_t_end = time.time()

        # return new theta and stats
        return theta, StepStats(
            po_returns_mean=po_returns.mean(),
            po_returns_median=np.median(po_returns),
            po_returns_std=po_returns.std(),
            po_returns_max=po_returns.max(),
            po_theta_max=po_theta_max,
            po_returns_min=po_returns.min(),
            po_len_mean=po_lengths.mean(),
            po_len_std=po_lengths.std(),
            noise_std=self.noise_std,
            learning_rate=self.alpha,
            theta_norm=np.square(theta).sum(),
            grad_norm=float(np.square(grads).sum()),
            update_ratio=float(0.0),
            episodes_this_step=self.top_n,
            timesteps_this_step=po_lengths.sum(),
            time_elapsed_this_step=step_t_end - step_t_start,
        )
