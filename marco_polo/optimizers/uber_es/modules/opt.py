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

# The following code is modified from uber-research/poet
# (https://github.com/uber-research/poet)
# under the Apache 2.0 License.


"""Uber ES Team Wrapper and Optimization Controller"""


import json
import logging
import os
import time
from collections import namedtuple
from typing import Any, cast  # for type hinting
import argparse  # for type hinting

import numpy as np
from numpy.random import PCG64DXSM, Generator

from marco_polo.optimizers.uber_es.modules.optimizers import Adam
from marco_polo.optimizers.uber_es.modules.result_objects import POResult
from marco_polo.tools.stats import compute_CS_ranks, compute_weighted_sum
from marco_polo.tools.types import FloatArray, Role, PathString
from marco_polo.tools.wrappers import TeamHistory, AgtHistory

logger = logging.getLogger(__name__)


################################################################################
## Aux Objects
################################################################################
StepStats = namedtuple(
    "StepStats",
    [
        "po_returns_mean",
        "po_returns_median",
        "po_returns_std",
        "po_returns_max",
        "po_theta_max",
        "po_returns_min",
        "po_len_mean",
        "po_len_std",
        "noise_std",
        "learning_rate",
        "theta_norm",
        "grad_norm",
        "update_ratio",
        "episodes_this_step",
        "timesteps_this_step",
        "time_elapsed_this_step",
    ],
)


################################################################################
## Optimization Controller
################################################################################
class ESTeamWrapper:
    """Wrapper for the optimizers that apply to a given team

    By default, the neural network models of all agents are updated by
    the methods in this class. However, a set of roles can be marked
    as 'frozen' to prevent the agents from being updated. This can be
    done via the parameter file using the key uber_es.freeze.
    Alternatively, the list of frozen roles can be changed later
    (see add_freeze(), remove_freeze(), and remove_all_freeze())
    """

    def __init__(self, args: argparse.Namespace, team: TeamHistory) -> None:
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
            self.optimizers[role] = OptimizationController(args=args, agent=team[role])

    def __getattr__(self, name: str) -> Any:
        """Returns an attribute with ``name``, unless ``name`` starts with an underscore."""
        if name.startswith("_"):
            raise AttributeError(f"accessing private attribute '{name}' is prohibited")
        return getattr(self.team, name)

    def __len__(self) -> int:
        return len(self.team)

    def get_freeze(self) -> list[Role]:
        """Return list of frozen agent roles

        Parameters
        ----------
        None

        Returns
        -------
        list[Role]
            copy of the list of roles
        """
        return [cast(Role, i) for i in self._freeze]

    def remove_freeze(self, role: Role) -> None:
        """Remove role from set of frozen agent roles

        Parameters
        ----------
        role : Role
            role to remove

        Raises
        ------
        KeyError:
            If role is not contained in the set of frozen roles.
        """
        self._freeze.remove(role)

    def add_freeze(self, role: Role) -> None:
        """Adds role to set of frozen agent roles

        If the role already exists, this has no effect.

        Parameters
        ----------
        role : Role
            role to add

        Returns
        -------
        None
        """
        self._freeze.add(role)

    def remove_all_freeze(self) -> None:
        """Remove all roles from set of frozen agent roles"""
        self._freeze.clear()

    def checkpoint(self, folder: str) -> None:
        """Store Objects to Disk

        This function stores the optimizers to disk. It does not store the agents
        or the args - those get stored elsewhere and can be reloaded.
        Additionally, the set of roles to freeze is stored.

        Parameters
        ----------
        folder : str
            Directory to store objects

        Side-Effects
        ------------
        None

        Returns
        -------
        None
            All objects are saved to disk
        """
        # convert freeze to list as a set is not JSON serializable
        obj_data = {"_freeze": self.get_freeze()}
        file_path = os.path.join(folder, "es_team_wrapper.json")
        with open(file_path, mode="w", encoding="utf-8") as file:
            json.dump(obj_data, file)

        # loop through agent optimizers, store by agent role
        for role, optimizer in self.optimizers.items():
            optimizer.checkpoint(folder=folder, role=role)

    def reload(self, folder: str) -> None:
        """Reload Objects from Disk

        Reloads the freeze set and the optimizers.

        Parameters
        ----------
        folder : str
            Directory to reload objects

        Side-Effects
        ------------
        All
            Creates all internal objects

        Returns
        -------
        None
        """
        file_path = os.path.join(folder, "es_team_wrapper.json")
        try:
            with open(file_path, mode="r", encoding="utf-8") as file:
                json_dict = json.load(file)
                self._freeze = set(json_dict["_freeze"])
        except FileNotFoundError:
            pass  # backwards compatibility for before freeze was added/saved

        # the optimizer dictionary is created on initialization
        #  this goes through and reset parameters
        for role, optimizer in self.optimizers.items():
            optimizer.reload(folder=folder, role=role)

    def shift_weights(self, noise_std: float, seed: int) -> dict[Role, int]:
        """Randomly shift the weights of the agents

        Values will not be changed for any role listed in this object's
        freeze set.

        Parameters
        ----------
        noise_std: float
            scaling factor used for random value adjustments
        seed: int
            random number seed used for generating each agent's seed

        Returns
        -------
        dict[Role, int]
            seeds for each role (calculated from the seed that was given)
        """
        random_state = Generator(PCG64DXSM(seed=seed))
        seeds = dict()
        for role, agent in self.team.items():
            seed = random_state.integers(low=0, high=2**63)
            if role not in self._freeze:
                agent.shift_weights(noise_std=noise_std, seed=seed)
            seeds[role] = seed
        return seeds

    def combine_and_update(
        self,
        step_results: list[dict[Role, POResult]],
        step_t_start: float,
        decay_noise: bool = True,
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
                step_t_start=step_t_start,
                decay_noise=decay_noise,
                propose_only=_propose_only,
            )

            # update agent
            if not _propose_only:
                self.team[role].update_theta(new_theta=new_theta)
                self.team[role].set_model_params(model_params=new_theta)

        # return results
        return stats


################################################################################
## what is this? Nate, name this plz.
################################################################################
class OptimizationController:
    """
    OptimizationController

    This class maintains all of the Adam learning parameters for each model.
    """

    def __init__(self, args: argparse.Namespace, agent: AgtHistory) -> None:
        # Meta options
        self.l2_coeff = args.uber_es["l2_coeff"]
        self.returns_normalization = args.uber_es["returns_normalization"]
        self.normalize_grads_by_noise_std = args.uber_es["normalize_grads_by_noise_std"]

        # learning rate
        self.learning_rate = args.uber_es["learning_rate"]
        self.lr_decay = args.uber_es["lr_decay"]
        self.lr_limit = args.uber_es["lr_limit"]

        # noise
        self.noise_std = args.uber_es["noise_std"]
        self.noise_decay = args.uber_es["noise_decay"]
        self.noise_limit = args.uber_es["noise_limit"]
        self.init_noise_std = self.noise_std

        # setup optimizer
        self.optimizer = Adam(
            theta=agent.get_zeroed_model_params(), stepsize=self.learning_rate
        )

    def __str__(self) -> str:
        return "Optimization Controller"

    def checkpoint(self, folder: PathString, role: Role) -> None:
        json_dict = self.__dict__.copy()
        del json_dict["optimizer"]

        file_path = os.path.join(folder, f"{role}_opt.json")
        with open(file_path, mode="w", encoding="utf-8") as f:
            json.dump(json_dict, f)

        self.optimizer.checkpoint(folder=folder, role=role)

    def reload(self, folder: PathString, role: Role) -> None:
        file_path = os.path.join(folder, f"{role}_opt.json")
        with open(file_path, mode="r", encoding="utf-8") as f:
            json_dict = json.load(f)
            for k, v in json_dict.items():
                self.__dict__[k] = v

        self.optimizer.reload(folder=folder, role=role)

    def reset_optimizers(self) -> None:
        self.optimizer.reset()
        self.noise_std = self.init_noise_std

    def get_noise(self, seed: int, theta_len: int) -> FloatArray:
        random_state = Generator(PCG64DXSM(seed=seed))
        # BUG: This is supposed to be a normal distribution
        #      However, it performs better, so we leave it.
        #      Should read: random_state(scale=noise_std, size=self.actor_param_count)
        return 2 * (random_state.random(theta_len) - 0.5)

    def compute_grads(
        self, noise_seeds: FloatArray, returns: FloatArray, theta: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        """Computes and returns gradients for the thetas based on run results

        Function takes the run results, normalized them, and computes the weighted sum of the
        noise vectors to construct the gradient G = sum_n E(theta + epsilon)epsilon.
        It also returns the theta of the largest return value.

        Parameters
        ----------
        noise_seeds : FloatArray
            Seed for each theta estimate
        returns : FloatArray
            Returns from the simulations
        theta : FloatArray
            Current network parameterization

        Returns
        -------
        tuple[FloatArray, FloatArray]
            Gradient vector for update, and theta estimate with maximum reward
        """

        theta_len = len(theta)
        pos_row, neg_row = returns.argmax(axis=0)
        noise_sign = 1.0
        po_noise_seed_max = noise_seeds[pos_row]

        if returns[pos_row, 0] < returns[neg_row, 1]:
            noise_sign = -1.0
            po_noise_seed_max = noise_seeds[neg_row]

        # BUG: This is supposed to be a normal distribution
        #      Be careful if get_noise() is updated, the scale here will be off
        po_theta_max = theta + noise_sign * self.noise_std * self.get_noise(
            po_noise_seed_max, theta_len
        )

        if self.returns_normalization == "centered_ranks":
            proc_returns = compute_CS_ranks(returns)
        elif self.returns_normalization == "normal":
            proc_returns = (returns - returns.mean()) / (returns.std() + 1e-5)
        else:
            raise NotImplementedError(
                "Invalid return normalization `{}`".format(self.returns_normalization)
            )

        # BUG: This is supposed to be a normal distribution
        #      This is currently incorrect, but if get_noise() is fixed, it will
        #      be fine.
        grads, _ = compute_weighted_sum(
            weights=proc_returns[:, 0] - proc_returns[:, 1],
            vec_generator=(self.get_noise(seed, theta_len) for seed in noise_seeds),
            theta_size=theta_len,
        )

        # NOTE: I don't think this is correct - JB
        #       returns is an array, either 1D (positive rollouts only)
        #       or 2D (positive and negative). But, we combine those into a single
        #       weighted sum, so we only use len(returns)/2 values.
        grads /= len(returns)
        if self.normalize_grads_by_noise_std:
            grads /= self.noise_std
        return grads, po_theta_max

    def combine_steps(
        self,
        step_results: list[POResult],
        theta: FloatArray,
        step_t_start: float,
        decay_noise: bool = True,
        propose_only: bool = False,
    ) -> tuple[FloatArray, StepStats]:
        # Extract
        nList = []
        rList = []
        lList = []
        for r in step_results:
            nList.append(r.noise_inds)
            rList.append(r.returns)
            lList.append(r.lengths)

        # Reshape results
        noise_inds = np.concatenate(nList)
        po_returns = np.concatenate(rList)
        po_lengths = np.concatenate(lList)

        # Calculate gradients
        grads, po_theta_max = self.compute_grads(
            noise_seeds=noise_inds, returns=po_returns, theta=theta
        )

        # update
        if not propose_only:
            update_ratio, theta = self.optimizer.update(
                theta, -grads + self.l2_coeff * theta
            )

            self.optimizer.stepsize = max(
                self.optimizer.stepsize * self.lr_decay, self.lr_limit
            )

            if decay_noise:
                self.noise_std = max(
                    self.noise_std * self.noise_decay, self.noise_limit
                )

        else:  # only make proposal
            update_ratio = 0.0
            # leave theta alone

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
            learning_rate=self.optimizer.stepsize,
            theta_norm=np.square(theta).sum(),
            grad_norm=float(np.square(grads).sum()),
            update_ratio=float(update_ratio),
            episodes_this_step=len(po_returns),
            timesteps_this_step=po_lengths.sum(),
            time_elapsed_this_step=step_t_end - step_t_start,
        )
