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
from collections import OrderedDict
import json
import logging
import os
from numpy.random import PCG64DXSM, Generator

from marco_polo.optimizers.uber_es.manager import Manager  # for type hinting
from marco_polo.tools.types import Role, PathString  # for type hinting
from marco_polo.tools.wrappers import EnvHistory, TeamHistory  # for type hinting

logger = logging.getLogger(__name__)


################################################################################
## Aux Transfer Funcs
################################################################################


################################################################################
## Optim Func
################################################################################


################################################################################
## Evolve Class
################################################################################
class EvolutionTemplate:
    """
    Template for Evolution Algorithms

    This class implements a sequential-style evolution, similar to a Moran process,
    where environments are checked if they are ready to reproduce, and then they
    produce a single child that replaces the parent environment.

    Attributes
    ----------
    None
    """

    def __init__(
        self, args: argparse.Namespace, manager: Manager, transfer_role: Role
    ) -> None:
        """
        PataECEvolution Initilizer

        Parameters
        ----------
        args : argparse.Namespace
            This stores all of the simulation parameters
        manager : Manager Object
            This class handles the compute and multithreading
        transfer_role : str
            Which role is used to determine the evolution

        Side-Effects
        ------------
        None

        Returns
        -------
        None
        """

        # self.args = args
        # self.novelty = pata_ec(args, manager, transfer_role)
        # self.manager = manager
        # self.transfer_role = transfer_role
        # self.np_random = Generator(PCG64DXSM(seed=args.master_seed))

        pass

    ########################################
    ## Aux Funcs
    ########################################
    def checkpoint(self, folder: PathString) -> None:
        """
        Save Object Attributes to Disk

        This class has no attributes to save.

        Parameters
        ----------
        folder : PathString
            Path to folder that will contain checkpoint information

        Side-effects
        ------------
        None

        Returns
        -------
        None
        """

        pass

    def reload(self, folder: PathString) -> None:
        """
        Reload Object Attributes

        This class has no attributes to reload.

        Parameters
        ----------
        folder : PathString
            Folder containing checkpoint

        Side-effects
        ------------
        None

        Returns
        -------
        None
        """

        pass

    ########################################
    ## Main Func
    ########################################
    def Evolve(
        self,
        bracket: OrderedDict[EnvHistory, TeamHistory],
        archived_envs: OrderedDict[str, EnvHistory],
        epoch: int,
        repro_threshold: float,
        max_active_envs: int = 8,
        max_children: int = 8,
        max_admitted: int = 1,
    ) -> tuple[
        list[EnvHistory],
        OrderedDict[str, TeamHistory],
        OrderedDict[EnvHistory, TeamHistory],
        int,
        int,
        OrderedDict[str, EnvHistory],
    ]:
        """Try to evolve new environmental niches from old ones.

        Check if it's time to evolve. If so, get list of candidate optimizers for
        reproduction. Score each optimizer on each env, clip and rank the scores.
        Based on those scores, get a list of children up to max_children, loop through
        the potetntial children evaluating each until we find max_admitted suitable candidates

        Parameters
        ----------
        brackets : OrderedDict[env, agent]
            Ordered dictionary of env, EnvStat pair, viable envs and agents will be derived from this.
        archived_envs : OrderedDict[int, env]
            Ordered dictionary of environments, keyed by ID
        epoch : int
            Current global epoch number
        repro_threshold : float
            Reproduction threashold
        max_active_envs : int, default=8
            Maximum number of environments to keep active, older envs will be archived
        max_children : int, default=8
            Maximum number of mutations to attempt, attempted mutations may not pass mc
        max_admitted : int, default=1
            How many mutations to keep

        Side-Effects
        ------------
        Some
            Updates the id counters and prng states in agents and environments.
            If anything is archived, the novelty archive is updated.
            Calls "_playall()", but only on new environments.

        Returns
        -------
        new_envs : List[env]
            List of newly added environments
        new_teams : OrderedDict[int, agent]
            Ordered dictionary of new team IDs and teams
        bracket : OrderedDict[env, agent]
            Updated environment/agent bracket
        ANNECS : int
            Number of archived environments that were solved
        not_ANNECS : int
            Number of archived environments that were not solved
        to_archive : OrderedDict[int, env]
            Ordered dictionary of newly archived environments
        """

        # init vars for return
        new_envs: list[EnvHistory] = []
        new_teams: OrderedDict[str, TeamHistory] = OrderedDict()
        ANNECS = 0
        not_ANNECS = 0
        to_archive: OrderedDict[str, EnvHistory] = OrderedDict()

        to_remove = list()

        for env, team in bracket.items():
            # check if team is ready for evolution
            if env.stats.transfer_threshold >= repro_threshold:
                # generate new env
                new_env = env.get_mutated_env()

                # fill with some stats
                new_env.stats.created_at = epoch
                # new_env_params.stats.recent_scores.append(child_stats.eval_returns_mean)
                # new_env_params.stats.transfer_threshold = child_stats.eval_returns_mean
                new_env.stats.team = team.copy()

                # this score isn't really valid.
                new_env.stats.best_score = env.stats.recent_scores[
                    -1
                ]  # give it most recent score
                new_env.stats.best_team = new_env.stats.team.copy()

                # update stats
                new_envs.append(new_env)
                new_teams[new_env.stats.team.id] = new_env.stats.team
                ANNECS += 1
                to_archive[env.id] = env

                # update bracket things
                to_remove.append(env)

        # add new ones
        for env in new_envs:
            bracket[env] = env.stats.team

        # remove old ones
        for env in to_remove:
            del bracket[env]

        # return info
        return new_envs, new_teams, bracket, ANNECS, not_ANNECS, to_archive
