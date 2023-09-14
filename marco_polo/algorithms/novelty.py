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


import argparse  # for type hinting
import json
import logging
import os
from collections import OrderedDict

import numpy as np

from marco_polo.tools.iotools import (
    NumpyDecoder,
    NumpyEncoder,
    load_keyed_tuple,
    save_keyed_tuple,
)
from marco_polo.optimizers.uber_es.manager import Manager
from marco_polo.tools.stats import compute_CS_ranks
from marco_polo.tools.types import EnvId, FloatArray, Role, PathString
from marco_polo.tools.wrappers import (
    EnvHistory,
    RunTask,
    TeamHistory,
)  # for type hinting

logger = logging.getLogger(__name__)


################################################################################
## Aux Things
################################################################################
def _cap_score(score: float, lower: float, upper: float) -> float:
    """
    Restrict the range of returned value

    This function ensures that the score of interest is greater than some lower
    bound and less than some upper bound. This is not a normalization, more similar
    to a truncation.

    Parameters
    ----------
    score: float
        Current score to restrict
    lower: float
        Lower bound on restriction
    upper: float
        Upper bound on restriction

    Side-effects
    ------------
    None

    Returns
    -------
    float
        value between lower and upper
    """

    if score < lower:
        score = lower
    elif score > upper:
        score = upper

    return score


def _euclidean_distance(x: FloatArray, y: FloatArray) -> np.float_:
    """
    Calculate the euclidean distance between 2 vectors

    Originally, this was a complicated function with protections for non-standard
    input format and x/y objects of differing lengths. At this point, we expect
    numpy arrays of equal length, or else the simulation is wrong somewhere else.

    Parameters
    ----------
    x: numpy.array of float
        array of agent scores
    y: numpy.array of float
        array of agent scores

    Side-effects
    ------------
    None

    Returns
    -------
    float
        distance between the vectors
    """

    # print("\n")
    # print(type(x))
    # print(type(y))
    # print(len(x))
    # print(len(y))
    # print("\n")

    # if type(x) is not list:
    #     #print("novelty._euclidean_distance:: inputs not of type list but of type", type(x))
    #     x = np.array([x])
    #     y = np.array([y])

    # n, m = len(x), len(y)

    # if n > m:
    #     # compute distance between shared dimensions
    #     a = np.linalg.norm(y - x[:m])
    #     # impute distance between last point and extra dimensions
    #     b = np.linalg.norm(y[-1] - x[m:])
    # else:
    #     a = np.linalg.norm(x - y[:n])
    #     b = np.linalg.norm(x[-1] - y[n:])
    # # combine as if 2-d euclidean
    # return np.sqrt(a**2 + b**2)

    return np.linalg.norm(x - y)


################################################################################
## PATA_EC Class
################################################################################
class pata_ec:
    """
    Performance of All Transferred Agents Environment Characterization

    Attributes
    ----------
    manager :
        Compute manager object defined under the optimizers directory
    rl : str
        Agent role in the game
    mc_lower : float
        Lower bound on evaluation scores
    mc_upper : float
        Upper bound on evaluation scores
    archived_pata_ec : dict[tuple(int, int), float]
        Dictionary of archived team scores, for reference to avoid recalculating things
    pata_list : list[numpy.array(numpy.float32)]
        Current centered/scaled scores
    k : int
        How many nearest environments to use as a score
    log_pata_ec : bool
        Should we log raw and scaled scores every epoch
    log_file : string
        Root directory for pata_ec logging
    """

    def __init__(self, args: argparse.Namespace, manager: Manager, role: Role) -> None:
        """
        pata_ec class initializer

        Parameters
        ----------
        args : argparse.Namespace
            Seems to be like a dictionary
            This stores all of the simulation parameters
        manager : Manager Object
            Compute manager object
        role : Role
            The role of the agent to use for novelty calculations

        Side-effects
        ------------
        None

        Returns
        -------
        None
        """

        self.manager = manager
        self.role = role
        self.mc_lower = args.mc_lower
        self.mc_upper = args.mc_upper
        self.archived_pata_ec: dict[tuple[EnvId, EnvId], float] = {}
        self.pata_list: list[FloatArray] = []
        self.k = 5
        self.log_pata_ec = args.log_pata_ec
        self.log_file = os.path.join(args.log_file, "pata_ec")

        # if logging, build output directory
        if self.log_pata_ec:
            os.makedirs(self.log_file, exist_ok=True)

    ########################################
    ## Funcs
    ########################################
    def checkpoint(self, folder: PathString) -> None:
        """
        Store Objects to Disk

        Stores all parameters except the manager object. It checks if the archive
        has objects and stores if it does.

        Parameters
        ----------
        folder : str
            Directory to store objects

        Side-effects
        ------------
        None

        Returns
        -------
        None
            All objects are saved to disk
        """
        # check archive, save if it has things
        if self.archived_pata_ec:
            save_keyed_tuple(
                self.archived_pata_ec, os.path.join(folder, "archived_pata_ec.csv")
            )

        # get all attributes, but remove the 2
        tmp = {
            k: v
            for k, v in self.__dict__.items()
            if k not in {"archived_pata_ec", "manager"}
        }

        # store
        with open(os.path.join(folder, "pata_ec.json"), mode="w", encoding="utf8") as f:
            json.dump(tmp, f, cls=NumpyEncoder)

    def reload(self, folder: PathString) -> None:
        """
        Reload Objects from Disk

        This function assumes that the args and manager objects are handled elsewhere.
        Selectively reloads archive if it was stored

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
        with open(
            os.path.join(folder, "pata_ec.json"), mode="r", encoding="utf8"
        ) as file:
            # load main stuff
            dct = json.load(file, cls=NumpyDecoder)

            # check if archive exists
            filepath = os.path.join(folder, "archived_pata_ec.csv")
            if os.path.isfile(filepath):
                dct["archived_pata_ec"] = load_keyed_tuple(filepath, float)

            # load into novelty
            for key, val in dct.items():
                if key not in {"manager"}:
                    self.__dict__[key] = val

    def _log_child_pata(
        self,
        filename: str,
        arc_opts: list[EnvHistory],
        act_opts: list[EnvHistory],
        child_opts: list[EnvHistory],
        pataList: list[FloatArray],
    ) -> None:
        """
        Write PATA_EC of Proposed Child Environments

        Parameters
        ----------
        filename : str
            Output file name
        arc_opts : list[env]
            List of archived environments to pair with scores
        act_opts : list[env]
            List of active environments to pair with scores
        child_opts : list[env]
            List of child environments to pair with scores
        pataList : list[iterable[float]]
            List of scores to write out

        Side-Effects
        ------------
        None

        Returns
        -------
        None
            Output written to file
        """
        # constuct a dict with keys given by a tuple of each pair
        tmp_dct: dict[tuple[EnvId, EnvId], float] = {}
        r_keys = [env.id for env in arc_opts + act_opts]
        l_keys = [env.id for env in child_opts]
        for l_key, scores in zip(l_keys, pataList):
            for r_key, value in zip(r_keys, scores):
                tmp_dct[(l_key, r_key)] = value
        filepath = os.path.join(self.log_file, filename)
        save_keyed_tuple(tmp_dct, filepath)

    def novelty(
        self,
        arc_opts: OrderedDict[str, EnvHistory],
        opts: list[EnvHistory],
        opt_list: list[EnvHistory],
        epoch: int,
    ) -> dict[EnvId, float]:
        """
        Calculate New Novelty

        This calculates PATA_EC based novelty for all envs in `opt_list`.
        It does not save or store any of this information. It takes all teams
        from the current/archived population and runs them on the proposed environments.

        Parameters
        ----------
        arc_opts : OrderedDict[str, EnvHistory]
            Ordered dictionary of archived environments
        opts : list[EnvHistory]
            List of active environments
        opt_list : list[EnvHistory]
            List of proposed environments with or without teams
        epoch : int
            Current simulation time

        Side-Effects
        ------------
        None

        Returns
        -------
        novelty : dict[EnvId, float]
            Dictionary of environment IDs and their novelty scores
        """
        # NOTE: We need to be very careful with this function and the update_novelty().
        #       The env/team pairs must be evaluated in the same order or else
        #       the distance calculation is wrong.

        # setup objects and counters
        num_opt = len(arc_opts) + len(opts)
        tasklist: list[RunTask] = []
        novelty: dict[EnvId, float] = {}
        # these 2 for storing output
        capList: list[FloatArray] = []
        CNList: list[FloatArray] = []

        # loop over new envs and all teams (associated with archived envs), build task list
        for env1 in opt_list:
            # loop over archive
            for env2 in arc_opts.values():
                tasklist.append((env1, env2.stats.team))

            # loop over active population
            for env2 in opts:
                tasklist.append((env1, env2.stats.team))

        # evaluate all teams on environments
        stats = self.manager.evaluate(tasks=tasklist)

        # Loop through envs, get team performance, calculate novelty
        stats_iter = iter(stats)
        for env1 in opt_list:
            # intermediate list
            capped_scores = []

            # loop over teams
            for _ in range(num_opt):
                # get mean result from run
                #  cap score for stability
                capped_scores.append(
                    _cap_score(
                        score=next(stats_iter)[self.role].eval_returns_mean,
                        lower=self.mc_lower,
                        upper=self.mc_upper,
                    )
                )

            # save capped scores for later
            capList.append(np.array(capped_scores))

            # calculate ranked and centered scores
            csRank = compute_CS_ranks(np.array(capped_scores))
            # store for later
            CNList.append(csRank)

            # calculate Euclidean distance between current env and other envs
            distances = np.array(
                [_euclidean_distance(csRank, c) for c in self.pata_list]
            )

            # sort, get K closest envs, and calculate the average distance as novelty
            top_k_indicies = (distances).argsort()[: self.k]
            novelty[env1.id] = distances[top_k_indicies].mean()

        # logging child pata_ec every epoc for analysis
        if self.log_pata_ec:
            # write plain capped scores
            self._log_child_pata(
                filename=f"Epoch_{epoch}_Child_Capped.csv",
                arc_opts=list(arc_opts.values()),
                act_opts=opts,
                child_opts=opt_list,
                pataList=capList,
            )

            # output centered and normalized scores
            self._log_child_pata(
                filename=f"Epoch_{epoch}_Child_Processed.csv",
                arc_opts=list(arc_opts.values()),
                act_opts=opts,
                child_opts=opt_list,
                pataList=CNList,
            )

        # cleanup
        del capList, CNList

        # return novelty dict
        return novelty

    def update_archive(
        self,
        cur_arc_opts: OrderedDict[str, EnvHistory],
        cur_opts: OrderedDict[EnvHistory, TeamHistory],
        new_opts: list[EnvHistory],
        new_arc_opts: OrderedDict[str, EnvHistory],
    ) -> None:
        """
        Update Archived Novelty Scores

        This function gets called whenever an environment gets archived. It then
        runs the archived team against all existing environments and updates the
        archive dictionary with the final scores.

        Parameters
        ----------
        cur_arc_opts : OrderedDict[str, EnvHistory]
            Ordered dictionary of archived environments with teams
        cur_opts : OrderedDict[EnvHistory, TeamHistory]
            Ordered dictionary of the currently active environments
        new_opts : list[EnvHistory]
            List of new environments that were just added to the current optimizers
        new_arc_opts : OrderedDict[str, EnvHistory]
            Ordered dictionary of newly archived environments

        Side-Effects
        ------------
        self.archived_pata_ec
            Update internal archive with new scores

        Returns
        -------
        None
        """
        # NOTE: Order here doesn't matter as much as the other 2 because we're
        #       just loading into a dict.
        #       Do need to make sure the order within this function is consistent,
        #       otherwise things will get labeled wrong.
        # do 3 loops to avoid building a super long list of things to loop over.

        # evals to compute
        tasklist = []

        # current archive on new env
        for env1 in new_opts:
            for env2 in cur_arc_opts.values():
                tasklist.append((env1, env2.stats.team))

        # loop over new agents
        for env2 in new_arc_opts.values():
            # current archive
            for env1 in cur_arc_opts.values():
                tasklist.append((env1, env2.stats.team))

            # current active
            for env1 in cur_opts.keys():
                tasklist.append((env1, env2.stats.team))

            # new archive
            for env1 in new_arc_opts.values():
                tasklist.append((env1, env2.stats.team))

        # evaluate all agents on environments
        stats = self.manager.evaluate(tasks=tasklist)

        # loop through and add to archive
        stats_iter = iter(stats)

        # Note: the keys to archived_pata_ec are a tuple: (env1.id, env2.id). This is
        # actually tracking a comparison of env1 to team2. Using env2.id instead of
        # team2.id should give the same result since the team doesn't change for
        # archived cases. Using env2 instead of team2 is kept for historical reasons.
        # current archive on new env
        for env1 in new_opts:
            for env2 in cur_arc_opts.values():
                self.archived_pata_ec[(env1.id, env2.id)] = _cap_score(
                    score=next(stats_iter)[self.role].eval_returns_mean,
                    lower=self.mc_lower,
                    upper=self.mc_upper,
                )

        # loop over new agents
        for env2 in new_arc_opts.values():
            # current archive
            for env1 in cur_arc_opts.values():
                self.archived_pata_ec[(env1.id, env2.id)] = _cap_score(
                    score=next(stats_iter)[self.role].eval_returns_mean,
                    lower=self.mc_lower,
                    upper=self.mc_upper,
                )

            # current active
            for env1 in cur_opts.keys():
                self.archived_pata_ec[(env1.id, env2.id)] = _cap_score(
                    score=next(stats_iter)[self.role].eval_returns_mean,
                    lower=self.mc_lower,
                    upper=self.mc_upper,
                )

            # new archive
            for env1 in new_arc_opts.values():
                self.archived_pata_ec[(env1.id, env2.id)] = _cap_score(
                    score=next(stats_iter)[self.role].eval_returns_mean,
                    lower=self.mc_lower,
                    upper=self.mc_upper,
                )

    def purge_archive(self, to_purge: set[EnvId]) -> None:
        """
        Clear archive of Forgotten Environments

        When an environment gets removed, but wasn't solved, it is forgotten instead
        of logged. In that instance, the archive still has traces of other archived
        agents playing on this environment. This function removes those traces.

        Parameters
        ----------
        to_purge : set[EnvId]
            Set of environment IDs to remove from the archive

        Side-Effects
        ------------
        Yes
            Removes matching entries from the archive

        Returns
        -------
        None
        """
        # archive keys are (env.id, env_agt.id)
        # we need to remove all matching env.id

        # instantiate list of keys to delete
        delList = []

        # loop through archive, collect keys to remove
        for k in self.archived_pata_ec.keys():
            if k[0] in to_purge:
                delList.append(k)

        # remove matching keys
        for k in delList:
            del self.archived_pata_ec[k]

    def _log_active_pata(
        self, filename: str, optList: list[EnvHistory], pataList: list[FloatArray]
    ) -> None:
        """
        Write PATA_EC of Active Environments

        Parameters
        ----------
        filename : str
            Output file name
        optList : list[EnvHistory]
            List of environments to pair with scores
        pataList : list[FloatArray]
            List of scores to write out

        Side-Effects
        ------------
        None

        Returns
        -------
        None
            Output written to file
        """
        # constuct a dict with keys given by a tuple of each pair
        tmp_dct = {}
        r_keys = [env.id for env in optList]
        l_keys = r_keys[:]
        for l_key, scores in zip(l_keys, pataList):
            for r_key, value in zip(r_keys, scores):
                tmp_dct[(l_key, r_key)] = value
        filepath = os.path.join(self.log_file, filename)
        save_keyed_tuple(tmp_dct, filepath)

    def update_novelty(
        self, arc_opts: OrderedDict[str, EnvHistory], opts: list[EnvHistory], epoch: int
    ) -> None:
        """
        Update Novelty List

        This function updates the internal list of centered and scaled scores for
        the existing envvironments. It calculates new scores for actively evolving
        agents or pulls old scores from the archival dict.

        Parameters
        ----------
        arc_opts : OrderedDict[str, EnvHistory]
            Ordered dictionary of archived optimizers with agents
        opts : list[EnvHistory]
            List of active optimizers with agents
        epoch : int
            Current simulation epoch

        Side-Effects
        ------------
        self.pata_list
            Internal list of normalized scores updated

        Returns
        -------
        None
            Output written to file
        """

        # NOTE: We need to be very careful with this function and the novelty().
        #       The env/agent pairs must be evaluated in the same order or else
        #       the distance calculation is wrong.

        # setup objects and counters
        all_opt = list(arc_opts.values()) + opts

        tasklist = []
        raw_pata_list: list[FloatArray] = []
        self.pata_list.clear()

        # loop over all envs and all agents, build task list
        # do NOT need to copy the teams here, they are safe
        for env1 in arc_opts.values():
            for env2 in opts:
                tasklist.append((env1, env2.stats.team))

        for env1 in opts:
            for env2 in opts:
                tasklist.append((env1, env2.stats.team))

        # run new tasks
        stats = self.manager.evaluate(tasks=tasklist, epoch=epoch)

        logger.debug(f"Return Stats: {stats}")

        # grab iterator of stats
        stats_iter = iter(stats)

        # loop through everyone, update score vec
        for env1 in all_opt:
            # intermediate list
            capped_scores = []

            # loop over archive first
            for env2 in arc_opts.values():
                # archive key
                capped_scores.append(self.archived_pata_ec[(env1.id, env2.id)])

            # loop over active second
            for _ in opts:
                capped_scores.append(
                    _cap_score(
                        score=next(stats_iter)[self.role].eval_returns_mean,
                        lower=self.mc_lower,
                        upper=self.mc_upper,
                    )
                )

            # track raw scores
            raw_pata_list.append(np.array(capped_scores))
            # compute new centered and scaled ranks, update pata list
            self.pata_list.append(compute_CS_ranks(np.array(capped_scores)))

        # logging pata_ec every epoc for analysis
        if self.log_pata_ec:
            # write plain capped scores
            self._log_active_pata(
                filename=f"Epoch_{epoch}_Active_Capped.csv",
                optList=all_opt,
                pataList=raw_pata_list,
            )

            # output centered and normalized scores
            self._log_active_pata(
                filename=f"Epoch_{epoch}_Active_Processed.csv",
                optList=all_opt,
                pataList=self.pata_list,
            )

        # clean up
        del raw_pata_list, tasklist
