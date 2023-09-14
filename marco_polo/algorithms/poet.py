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
from itertools import chain
from typing import cast, Iterable
from numpy.random import PCG64DXSM, Generator

from marco_polo.algorithms.novelty import pata_ec
from marco_polo.optimizers.uber_es.manager import Manager  # for type hinting
from marco_polo.tools.iotools import NumpyDecoder, NumpyEncoder
from marco_polo.tools.types import Role, PathString  # for type hinting
from marco_polo.tools.wrappers import (
    EnvHistory,
    NoneTeam,
    TeamHistory,
    RunTask,
)  # for type hinting

logger = logging.getLogger(__name__)


################################################################################
## Aux Transfer Funcs
################################################################################
def _playall(
    manager: Manager,
    env_list: Iterable[EnvHistory],
    team_list: list[TeamHistory],
    transfer_role: Role,
    epoch: int,
) -> None:
    """
    Combinatorial Environment/Team Transfers

    This function handles team transfer events. Events involve a first play
    of every environment against every team other than its own. From there,
    possible successes recieve 1 optimization step and are played again. If these
    teams pass a transfer threshold, they become the new team for that environment.

    Parameters
    ----------
    manager : Manager
        Manager object to deal with threads
    env_list : Iterable[EnvHistory]
        iterable-object of environments to use
    team_list : list[TeamHistory]
        List of teams to use. This supports None as an team
    transfer_role : Role
        Role of the *agent* in a team that is used to determine
        whether a transfer happens
    epoch : int
        Current simulation time

    Side-Effects
    ------------
    Many
        If a transfer happens, environment statistics, best team,
        and current team are updated. Teams get copied many times
        when passed around.

    Returns:
    --------
    None
    """
    # log that we're starting transfers
    a_len = len(team_list)
    logger.info(f"Epoch: {epoch} Computing direct transfers: {a_len*(a_len-1)}")

    logger.debug("team_list coming into _playall: " + str(team_list))
    # https://stackoverflow.com/a/57872832
    # setup direct-transfer tasks
    #  this involves pairing each env with all teams OTHER THAN IT'S OWN
    direct_tasks: list[RunTask] = []
    for i, env in enumerate(env_list):
        for team_idx in chain(range(0, i), range(i + 1, a_len)):
            # This does NOT need copied, because we just eval it here
            team = team_list[team_idx]
            direct_tasks.append((env, team))

    logger.debug("Direct transfer tasks: " + str(direct_tasks))
    # run evals
    returns = manager.evaluate(tasks=direct_tasks, epoch=epoch)

    # Check evals for proposal transfers
    logger.debug("Direct transfer scores:")
    proposal_tasks: list[RunTask] = []
    for stats, (env, team) in zip(returns, direct_tasks):
        # Did this team do better than the current team?
        logger.debug(
            f"Attempted Transfer: {env.id} - {stats[transfer_role].eval_returns_mean}"
        )
        if stats[transfer_role].eval_returns_mean > env.stats.transfer_threshold:
            # log successful direct transfer test
            logger.info(
                f"Direct Transfer: Epoch={epoch} - Env={env} - Team={team} - Score={stats[transfer_role].eval_returns_mean}"
            )
            # need to run proposal on direct transfer success
            #  THIS NEEDS TO BE A COPY
            #  We perform an optimization on this team, so it must be independent
            proposal_tasks.append((env, team.copy()))

    # log proposals
    logger.info(f"Epoch: {epoch} Computing proposal transfers: {len(proposal_tasks)}")

    # make sure there are proposals, otherwise skip
    if proposal_tasks:
        # single optimization step for env/team
        _ = manager.optimize_step(tasks=proposal_tasks, epoch=epoch, opt_iter=None)

        # evaluate optimized env/team pair
        returns = manager.evaluate(tasks=proposal_tasks, epoch=epoch)

        # check results of proposals
        for stats, (env, team) in zip(returns, proposal_tasks):
            # grab mean cause I'm tired of typing
            eval_mean = stats[transfer_role].eval_returns_mean

            # Did this team do better than the current team?
            if eval_mean > env.stats.transfer_threshold:
                # log successful direct transfer test
                logger.info(
                    f"Proposal Transfer: Epoch={epoch} - Env={env} - Team={team}"
                )

                # update env stats
                env.stats.recent_scores.clear()
                env.stats.recent_scores.append(eval_mean)
                env.stats.transfer_threshold = eval_mean
                env.stats.iterations_current = 1
                # lifespan best score and team
                if env.stats.best_score < eval_mean:
                    env.stats.best_score = eval_mean
                    env.stats.best_team = team.copy()
                    # this team is the same as "team", but gets a new id because of the copy
                    #  THIS COPY STATEMENTS ISREQUIRED HERE!!!!
                    #  It must be an independent team

                # update env/team pairing
                #  This does NOT need copied, because "proposal_tasks" is a copy
                env.stats.team = team


def AllActiveBestScore(
    manager: Manager,
    bracket: OrderedDict[EnvHistory, TeamHistory],
    transfer_role: Role,
    epoch: int,
) -> None:
    """
    Play All Agents and Perform Transfers

    Passes all work down to "_playall()".
    Then updates the active env/team bracket.

    Parameters
    ----------
    manager : Manager
        Manager object to deal with threads
    bracket : OrderedDict[EnvHistory, TeamHistory]
        Ordered dictionary of env, team pairs
    transfer_role : Role
        Which role is used to determine the evolution
    epoch : int
        Current simulation time

    Side-Effects
    ------------
    Many
        See "_playall()" for more details, many environment stats are updated.
    """
    # Play all envs against all teams
    _playall(
        manager=manager,
        env_list=bracket.keys(),
        team_list=list(bracket.values()),
        transfer_role=transfer_role,
        epoch=epoch,
    )

    logger.debug("Bracket after _playall:" + str(bracket))

    # update bracket
    for env in bracket:
        logger.debug("Teams in Envs: " + str(env.stats.team))
        bracket[env] = env.stats.team


################################################################################
## Optim Func
################################################################################
def PoetOptLoop(
    tasks: list[RunTask], epoch: int, manager: Manager, verbose: bool = False
) -> None:
    """

    Parameters
    ---------
    tasks : list[RunTask]
        List of RunTask to optimize
    epoch: int
        Main loop counter
    manager : Manager
        Manager object to deal with compute
    verbose : bool, default = False
        Manage logging

    Side-effects
    ------------
    Many
        This function updates the environment statistics based on rollouts.
        Internally, "manager.optimize_chunk()" changes things within environments
        and their corresponding teams.

    Returns
    -------
    None
    """
    # pass to manager to optimize
    #  returns a list of eval means for each optim step
    results = manager.optimize_chunk(tasks=tasks, epoch=epoch, verbose=verbose)

    # use the results to update environment stats
    for stats, (env, team) in zip(results, tasks):
        # grab eval results
        #  this is a list of mean evaluation values that is optim_iters long
        #  since team is actually several agents, but we're playing a 1-player
        #  game, grab the first key
        eval_mean_list = stats[next(iter(team.agent_group))]
        # store most recent evals
        env.stats.recent_scores.extend(eval_mean_list)
        # transfer threshold is best of N best scores
        env.stats.transfer_threshold = max(env.stats.recent_scores)

        # lifespan best score and team
        #  only checks the team after a whole lest of optimization, so we miss
        #  the teams/scores inbetween. Ergo, there is a chance that
        #  transfer_threshold is transiently better than best_score
        if env.stats.best_score < eval_mean_list[-1]:
            env.stats.best_score = eval_mean_list[-1]
            env.stats.best_team = team.copy()
            # this team is the same as "team", but gets a new id because of the copy
            #  THIS COPY STATEMENTS ISREQUIRED HERE!!!!
            #  It must be an independent team
            # NOTE: Is this missing a total_teams update?
            #       It's a new team copy, we don't every log it?
            #       Or should we be doing this in the `optimize()` function in the manager?


################################################################################
## Evolve Class
################################################################################
class PataECEvolution:
    """
    Classic POET Evolution

    This class performs the classic POET evolution. Internally, it has a separate
    class to calculate novelty, which is then used to test for environment
    reproduction. This class handles environment archiving as well.

    Attributes
    ----------
    args : argparse.Namespace
        This stores all of the simulation parameters
    novelty : Novelty Class Object
        This class handles the novelty calculations
    manager : Manager Object
        This class handles the compute and multithreading
    transfer_role : Role
        Which role is used to determine the evolution
    np_random : Generator
        Numpy random generator
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
        transfer_role : Role
            Which role is used to determine the evolution

        Side-Effects
        ------------
        Yes
            Sets internal variables

        Returns
        -------
        None
        """

        self.args = args
        self.novelty = pata_ec(args, manager, transfer_role)
        self.manager = manager
        self.transfer_role = transfer_role
        self.np_random = Generator(PCG64DXSM(seed=args.master_seed))

    ########################################
    ## Aux Funcs
    ########################################
    def checkpoint(self, folder: PathString) -> None:
        """
        Save a checkpoint of the PataECEvolution object

        Parameters
        ----------
        folder: PathString
            Path to folder that will contain checkpoint information

        Side-effects
        ------------
        None

        Returns
        -------
        None
        """
        # Build folder path
        folder = os.path.join(folder, "POET")
        # Make folder
        os.makedirs(folder, exist_ok=True)

        # Dictionary of things to save
        tmp = {
            "transfer_role": self.transfer_role,
            "np_random": self.np_random.bit_generator.state,
        }

        # Save data
        with open(
            os.path.join(folder, "PataECEvolution.json"), mode="w", encoding="utf8"
        ) as f:
            json.dump(tmp, f, cls=NumpyEncoder)

        # Checkpoint novelty
        #  uses built-in functionality
        self.novelty.checkpoint(folder)

    def reload(self, folder: PathString) -> None:
        """
        Reload a checkpointed PataECEvolution object from the supplied folder

        Parameters
        ----------
        folder: PathString
            Folder containing checkpoint

        Side-effects
        ------------
        Yes
            Sets internal variables

        Returns
        -------
        None
        """
        # Build folder path
        folder = os.path.join(folder, "POET")

        # Reload data
        with open(
            os.path.join(folder, "PataECEvolution.json"), mode="r", encoding="utf8"
        ) as f:
            # Read in
            dct = json.load(f, cls=NumpyDecoder)

            # Set attributes
            self.transfer_role = dct["transfer_role"]
            self.np_random.bit_generator.state = dct["np_random"]

        # Create and reload novelty object
        self.novelty = pata_ec(self.args, self.manager, self.transfer_role)
        self.novelty.reload(folder)

    def _check_env_status(
        self, bracket: OrderedDict[EnvHistory, TeamHistory], repro_threshold: float
    ) -> list[EnvHistory]:
        """
        Hidden Score Function

        Check the latest scores on envs and return a list of candidates for reproduction.

        Parameters
        ----------
        bracket : OrderedDict[EnvHistory, TeamHistory]
            Ordered dictionary of environment/team pairs
        repro_threshold : float
            Minimal score to pass for reproduction

        Side-effects
        ------------
        None

        Returns
        -------
        list[EnvHistory]
            List of environments that are score high enough to reproduce
        """
        # initialize loop objects
        repro_candidates = []

        # Loop through all active niches
        #  This always loops in the same order - orderedDict()
        for env in bracket.keys():
            # check if ready to reproduce
            #  use transfer_threshold, which is best of most recent 5 runs
            #  just a little more stable than a single run
            if env.stats.transfer_threshold >= repro_threshold:
                repro_candidates.append(env)

        return repro_candidates

    def _get_new_env(
        self, list_repro: list[EnvHistory], num_offspring: int
    ) -> tuple[list[tuple[EnvHistory, int, EnvHistory]], list[RunTask]]:
        """
        Generate Offspring Environment from Random Parent Environment

        This function uses the list of potential parent environments to generate
        the mutated children to possibly add to the currently active environments.
        It picks a parent env from the list, produces a mutated cppn
        and a new environment parameter set for all new agents.

        Parameters
        ----------
        list_repro : list[EnvHistory]
            List of environments to use as possible parents
        num_offspring : int
            How many new children to return.

        Side-Effects
        ------------
        Environment Wrapper
            This process increments the environment ID counter and updates the prng
            state within the chosen parent environment.

        Return
        ------
        children : list[tuple[EnvHistory, int, EnvHistory]]
            List of Inherited, mutated CppnEnvParams from parent with associated wrapper class,
            random seed, and parent environment
        tasks : list[RunTask]
            List of tuples of environment and team
        """
        # Grab list of ID of parent environment
        env_parent_list = self.np_random.choice(
            a=list_repro, size=num_offspring, replace=True, shuffle=False
        )

        # generate seed for child
        seed_list = self.np_random.integers(low=0, high=2**31 - 1, size=num_offspring)

        # loop and setup return objs
        tasks = []
        children = []
        for i in range(num_offspring):
            # generate child env
            child_env = env_parent_list[i].get_mutated_env()

            # add to return list
            children.append((child_env, seed_list[i], env_parent_list[i]))

            # build task list
            tasks.append((child_env, env_parent_list[i].stats.team))

            # log
            logger.info(f"Parent: {env_parent_list[i].id} - Child: {child_env.id}")

        # return child list and task list
        return children, tasks

    def _pass_mcc(self, score: float) -> bool:
        """
        Check if score in Minimum Criterion Coevolution (MCC) range.

        Parameters
        ----------
        score : float
            Current score to check

        Side-Effects
        ------------
        None

        Returns
        -------
        bool
            Whether or not the score passed
        """
        # return checks
        #  1) check that score is above lower
        #  2) check that score is below upper
        return (
            cast(float, self.args.mc_lower) <= score <= cast(float, self.args.mc_upper)
        )

    def _get_child_list(
        self,
        archived_envs: OrderedDict[str, EnvHistory],
        env_list: list[EnvHistory],
        parent_list: list[EnvHistory],
        max_children: int,
        epoch: int,
    ) -> list[tuple[EnvHistory, int, EnvHistory]]:
        """
        Returns a list of viable new environments

        This function does several things.
        First, it updates the novelty scores on existing environments.
        Then, it gets a potential new environment and evaluates it. If that environment
        passes a score threshold, it is stored as a passing new offspring.
        Then, it calculates the novelty scores for all passing offspring.
        Finally, it returns a list of pasisng offspring tuples, sorted by novelty score.

        This function calls several other internal functions and relies on novelty
        calculations.

        Parameters
        ----------
        archived_envs : OrderedDict[str, EnvHistory]
            Current environment archive
        env_list : list[EnvHistory]
            Current active environments
        parent_list : list[EnvHistory]
            List of environments that are viable for reproduction
        max_children : int
            Maximum number of new environments to test
        epoch : int
            Current simulation time

        Side-Effects
        ------------
        Several
            The agent/team id counters are incremented with every copy event.
            Evals update the internal state of current environments.
            Novelty scores get updated.
            The internals of new environments get modified.

        Returns
        -------
        list[tuple[EnvHistory, int, EnvHistory]]
            List of (new env, seed, parent env) tuple
        """
        # update current novelty
        self.novelty.update_novelty(arc_opts=archived_envs, opts=env_list, epoch=epoch)

        # setup return list to hold viable children
        #  These are defined as the potential children than pass MC checks.
        child_list = []
        child_novelties = []
        potential_children = {}
        pass_mcc = 0

        # get list of offspring
        #  children_list is a list of (child, seed, parent)
        children_list, task_list = self._get_new_env(
            list_repro=parent_list, num_offspring=max_children
        )

        # evaluate whole list
        stats = self.manager.evaluate(tasks=task_list, epoch=epoch)

        # check list for mcc
        for i in range(max_children):
            # grab stats
            child_stats = stats[i][self.transfer_role]

            # see if performance is viable
            if self._pass_mcc(score=child_stats.eval_returns_mean):
                # increment counter
                pass_mcc += 1

                # unpack child info
                new_env, seed, env_parent = children_list[i]

                # fill stats
                new_env.stats.created_at = epoch
                new_env.stats.recent_scores.append(child_stats.eval_returns_mean)
                new_env.stats.transfer_threshold = child_stats.eval_returns_mean
                assert env_parent.stats.team is not None
                new_env.stats.team = env_parent.stats.team.copy()

                new_env.stats.best_score = child_stats.eval_returns_mean
                new_env.stats.best_team = new_env.stats.team.copy()
                # this team is the same as "team", but gets a new id because of the copy
                #  THESE COPY STATEMENTS ARE REQUIRED HERE!!!!
                #  The child needs an independent team to own, thus the deep copy

                # add new child to list
                potential_children[new_env] = (new_env, seed, env_parent)

        # log number of passing envs
        logger.info(
            f"Attempted {max_children} reproductions - {pass_mcc} were successful"
        )

        # If there's kids, do stuff
        if potential_children:
            ## Compute the novelty scores:
            novelties = self.novelty.novelty(
                arc_opts=archived_envs,
                opts=env_list,
                opt_list=list(potential_children.keys()),
                epoch=epoch,
            )

            # loop over dictionary of new envs
            for env, vals in potential_children.items():
                # grab children and novelties
                child_novelties.append(novelties[env.id])
                child_list.append(vals)

            # sort child list according to novelty for high to low
            # https://www.geeksforgeeks.org/python-sort-values-first-list-using-second-list/
            child_list = [
                x
                for _, x in sorted(
                    zip(child_novelties, child_list), key=lambda x: x[0], reverse=True
                )
            ]

        # return child list
        return child_list

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
        brackets : OrderedDict[EnvHistory, TeamHistory]
            Ordered dictionary of env, EnvStat pair, viable envs and teams will be derived from this.
        archived_envs : OrderedDict[str, EnvHistory]
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
            Updates the id counters and prng states in teams and environments.
            If anything is archived, the novelty archive is updated.
            Calls "_playall()", but only on new environments.

        Returns
        -------
        new_envs : list[EnvHistory]
            List of newly added environments
        new_teams : OrderedDict[str, TeamHistory]
            Ordered dictionary of new team IDs and teams
        bracket : OrderedDict[EnvHistory, TeamHistory]
            Updated environment/team bracket
        ANNECS : int
            Number of archived environments that were solved
        not_ANNECS : int
            Number of archived environments that were not solved
        to_archive : OrderedDict[str, EnvHistory]
            Ordered dictionary of newly archived environments
        """
        # get current list of envs and teams
        env_list = list(bracket.keys())

        # _playall() runs env_i against team_j for i != j
        # we want to run a single env [env_0] against all teams, so
        # we pad the list of teams with a dummy team at j = 0
        team_list_aug = [NoneTeam(None)] + list(bracket.values())
        cur_env_len = len(env_list)

        # check current envs to see if viable for reproduction
        list_repro = self._check_env_status(
            bracket=bracket, repro_threshold=repro_threshold
        )

        # objects to return
        new_envs: list[EnvHistory] = []
        new_teams: OrderedDict[str, TeamHistory] = OrderedDict()
        ANNECS = 0
        not_ANNECS = 0
        to_archive: OrderedDict[str, EnvHistory] = OrderedDict()

        # early escape if no reproductive envs
        if len(list_repro) == 0:
            logger.info("no suitable niches to reproduce")
            return new_envs, new_teams, bracket, ANNECS, not_ANNECS, to_archive
        else:
            logger.info(f"Epoch:{epoch} - List of niches to reproduce: {list_repro}")

        # Get list of potential children
        # These children have passed initial MCC check, and are ranked by novelty
        child_list = self._get_child_list(
            archived_envs=archived_envs,
            env_list=env_list,
            parent_list=list_repro,
            max_children=max_children,
            epoch=epoch,
        )

        for child in child_list:
            # unpack child object
            #  (new env, seed, parent env)
            new_env, seed, _ = child

            # Perform transfers
            #  include own team, because it didn't do a proposal the first time
            _playall(
                manager=self.manager,
                env_list=[new_env],
                team_list=team_list_aug,
                transfer_role=self.transfer_role,
                epoch=epoch,
            )

            # check MCC a second time
            #  probably to make sure it isn't too easy now?
            if self._pass_mcc(score=new_env.stats.recent_scores[-1]):
                # Pass checks! Add to current population
                new_envs.append(new_env)
                # no copy on these teams - it's fine
                new_teams[new_env.stats.team.id] = new_env.stats.team
                # break loop if finished
                if len(new_envs) >= max_admitted:
                    break

        # add new environments to active bracket
        #  This might put us over the max pop size, we'll fix that next
        for env in new_envs:
            bracket[env] = env.stats.team

        # Check if we've reached max pop size
        #  take care of archiving tasks if yes
        to_purge = set()
        num_remove = len(new_envs) + cur_env_len - max_active_envs
        if num_remove > 0:
            # get oldest keys from bracket
            #  this works cause bracket is an orderedDict
            rem_envs = env_list[0:num_remove]

            # loop over keys, remove, and score
            for env in rem_envs:
                # remove element from current bracket
                del bracket[env]

                # check if solved
                #  "solved" is defined as "was able to reproduce"
                if env.stats.recent_scores[-1] >= repro_threshold:
                    # log that we solved an env and store in archiv
                    ANNECS += 1
                    to_archive[env.id] = env
                else:
                    # Not strictly necessary, but we're removing an unsolved env,
                    #  so log that it was there, but don't keep, in case it comes
                    #  back later
                    not_ANNECS += 1
                    to_purge.add(env.id)

            # update archive with newly-archived environments
            self.novelty.update_archive(
                cur_arc_opts=archived_envs,
                cur_opts=bracket,
                new_opts=new_envs,
                new_arc_opts=to_archive,
            )

            # clean archive of env to forget
            if not_ANNECS:
                self.novelty.purge_archive(to_purge=to_purge)

        # return info
        return new_envs, new_teams, bracket, ANNECS, not_ANNECS, to_archive
