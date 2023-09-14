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


"""Multi-Player  """

import copy
import json
import logging
import os
import random
import time
from collections import OrderedDict
import argparse  # for type hinting
from collections.abc import Callable  # for type hinting
from typing import Any  # for type hinting
import numpy as np

from marco_polo.algorithms.evolutiontemplate import EvolutionTemplate

from marco_polo.cores.ePoet.core import Core as BaseCore
from marco_polo.envs._base.env_params import EnvParams  # for type hinting
from marco_polo.tools.iotools import NumpyDecoder
from marco_polo.tools.section_logging import SectionLogger
from marco_polo.tools.wrappers import AgtHistory, EnvHistory, TeamHistory
from marco_polo.tools.types import Role, PathString

logger = logging.getLogger(__name__)


################################################################################
## Core Class
################################################################################
class Core(BaseCore):
    """
    Example core for Multiplayer Optimization. This core uses a simple sequential
    evolution algorithm to maintain the same number of environments. This is a good
    starting core for multiplayer development as it includes checkpointing and
    reloading.

    Attributes
    ----------
    args : argparse.Namespace
        Seems to be like a dictionary
        This stores all of the simulation parameters
    manager : Manager Object
        Class that handles compute/multi-threading
    evolver : PataECEvolution Object
        This object handles environment evolution
    total_envs : list[env]
        List of all environments in their order of creation
    archived_envs : OrderedDict[int, env]
        Ordered dictionary of envronments that have been archived
    total_teams : OrderedDict[int, env]
        Ordered dictionary of all teams in their order of creation
    brackets : OrderedDict[env, agent]
        Pairing of active agent/environments
    ANNECS : int
        Accumulated number of novel environments created and solved
    not_ANNECS : int
        Accumulated number of novel environments created and NOT solved
    """

    def __init__(
        self,
        args: argparse.Namespace,
        env_factory: Callable[..., Any],
        env_param_factory: Callable[..., Any],
        manager_factory: Callable[..., Any],
        model_factory: Callable[..., Any],
    ) -> None:
        """
        Core Class Initializer

        This function sets up the simulation starting state. It creates a
        compute manager and an evolution manager. It also creates the first
        agent(s) and environment(s).

        Parameters
        ----------
        args : argparse.Namespace
            Seems to be like a dictionary
            This stores all of the simulation parameters
        env_factory: Callable[..., Any],
            Factory for creating environment classes.
        env_param_factory: Callable[..., Any],
            Factory for creating environment parameter classes.
        manager_factory: Callable[..., Any],
            Factory for creating optimization manager classes.
        model_factory: Callable[..., Any],
            Factory for creating model classes.

        Side-effects
        ------------
        Many
            It builds all of the internal objects to start the simulation

        Returns
        -------
        None
        """

        # store sim args for later
        self.args = args
        self.env_factory = env_factory
        self.env_param_factory = env_param_factory
        self.manager_factory = manager_factory
        self.model_factory = model_factory

        # setup environment and parameter classes
        self.env_class = env_factory(args=args)
        self.env_param_class = env_param_factory(args=args)

        # Setup manager with any incoming args
        Manager = manager_factory(args=args)
        self.manager = Manager(setup_args=args)

        # Setup the environmnet evolution class
        self.role_name = Role("I DONT GET USED")
        self.evolver = EvolutionTemplate(
            args=args, manager=self.manager, transfer_role=self.role_name
        )

        # total env/team trackers
        self.total_teams: OrderedDict[str, TeamHistory] = OrderedDict()
        self.total_envs: list[EnvHistory] = []
        # active environment/team pairing
        self.brackets: OrderedDict[EnvHistory, TeamHistory] = OrderedDict()
        # archive
        self.archived_envs: OrderedDict[str, EnvHistory] = OrderedDict()

        # create initial env/team pairs (number given by args.num_start_envs)
        self._create_initial_env_teams(
            EnvCreator=self.env_class, env_param_class=self.env_param_class
        )

        # stats
        # accumulated number of novel environments created and solved
        self.ANNECS = 0  # pylint: disable=invalid-name
        # accumulated but not solved
        self.not_ANNECS = 0  # pylint: disable=invalid-name

    ########################################
    ## Funcs
    ########################################
    def _create_initial_env_teams(
        self, EnvCreator: Callable[..., Any], env_param_class: Callable[..., EnvParams]
    ) -> None:
        """
        Creates the initial environments and agents.
        """

        # temp env for creating agents
        tmp_env = EnvCreator()
        tmp_env.augment(params=env_param_class(self.args))

        # create initial team
        initial_agent_set = OrderedDict()
        for agent in tmp_env.agents:
            # copy params so we can tweak for each agent
            tmp_m_params = copy.copy(self.args)

            # grab input and output shape
            tmp_m_params.model_params["input_size"] = np.prod(
                tmp_env.observation_spaces[agent].shape
            )
            tmp_m_params.model_params["output_size"] = np.prod(
                tmp_env.action_spaces[agent].shape
            )

            # create initial agent
            #  args dict is because we made a true copy for different inputs/outputs
            #  original POET stuff used same in/outs for all agents
            initial_agent_set[agent] = AgtHistory(
                self.model_factory(
                    env=tmp_env,
                    role=agent,
                    args=tmp_m_params,
                    seed=self.args.master_seed,
                )
            )
        initial_team = TeamHistory(agent_group=initial_agent_set)

        # pair initial env/team and store
        starting_env = EnvHistory(
            env_param=env_param_class(self.args), env_creator=EnvCreator
        )
        starting_env.stats.team = initial_team
        self.total_teams[initial_team.id] = initial_team
        self.total_envs.append(starting_env)
        self.brackets[starting_env] = initial_team

        # check if we add other/more optimizers
        # Make sure we don't start with more than the maximum number of optimizers
        #  decrement by 1 because we already created 1 optimizer
        nStartOptims = min(self.args.max_active_envs, self.args.num_start_envs)
        nStartOptims -= 1

        # loop over remaing environments/teams to create
        for i in range(nStartOptims):
            # mutate new env
            new_env = starting_env.get_mutated_env()
            # new_env.augment(new_env.env_param)

            # create new team
            new_agent_set = OrderedDict()
            for agent in tmp_env.agents:
                # copy params so we can tweak for each agent
                #  this will create exactly the same agents for all starting envs
                #  Is this what we want? What if we want to evolve agents here too?
                tmp_m_params = copy.copy(self.args)

                # grab input and output shape
                tmp_m_params.model_params["input_size"] = np.prod(
                    tmp_env.observation_spaces[agent].shape
                )
                tmp_m_params.model_params["output_size"] = np.prod(
                    tmp_env.action_spaces[agent].shape
                )

                # create initial agent
                #  args dict is because we made a true copy for different inputs/outputs
                #  original POET stuff used same in/outs for all agents
                new_agent_set[agent] = AgtHistory(
                    self.model_factory(
                        env=tmp_env,
                        role=agent,
                        args=tmp_m_params,
                        seed=self.args.master_seed + i,
                    )
                )
            new_team = TeamHistory(agent_group=new_agent_set)

            # pair env/team and store
            new_env.stats.team = new_team
            self.total_teams[new_team.id] = new_team
            self.total_envs.append(new_env)
            self.brackets[new_env] = new_team

    def reload(self, folder: PathString) -> None:
        """
        Reload a checkpointed PataECEvolution object from the supplied folder.

        Parameters
        ----------
        folder: PathString
            Folder containing checkpoint.

        Side-effects
        ------------
        Yes
            Sets internal variables

        Returns
        -------
        None

        """

        ## We're going to load everything in as a dict and then let the
        ## manifest enforce order.

        # wrap in section logger for formatting
        with SectionLogger(logger, "Reloading") as section:
            ##########
            # Clear Attributes
            ##########
            # Some of the attributes were populated by initial envs/agents/teams
            # Clear them all here so we can add/rebuild later
            self.total_teams = OrderedDict()
            self.total_envs = []
            self.brackets = OrderedDict()
            # self.archived_envs = OrderedDict() # this is already empty at the beginning

            section.print_raw("Cleared Attributes")

            ##########
            # Reload Agents
            ##########
            # Get agents by directory
            agent_dict = dict()
            a_dir = os.path.join(folder, "Agents")
            # list of directories that should be agent names
            agent_paths = [
                name
                for name in os.listdir(a_dir)
                if os.path.isdir(os.path.join(a_dir, name))
            ]

            # agent/role dictionary
            with open(
                os.path.join(a_dir, "ID_Role.json"), mode="r", encoding="utf8"
            ) as file:
                arDict = json.load(file)

            # blank env for agent params
            agt_env = self.env_class()
            agt_env.augment(params=self.env_param_class(self.args))

            # create default agents, fill with reloaded params
            for agt_id in agent_paths:
                # clean copy of params
                tmp_m_params = copy.copy(self.args)
                # input/output
                inp = np.prod(agt_env.observation_spaces[arDict[agt_id]].shape)
                out = np.prod(agt_env.action_spaces[arDict[agt_id]].shape)
                # augment params
                tmp_m_params.model_params["input_size"] = inp
                tmp_m_params.model_params["output_size"] = out

                # agent
                tmp_agt = AgtHistory(
                    self.model_factory(
                        env=agt_env,
                        role=Role(arDict[agt_id]),
                        args=tmp_m_params,
                        seed=self.args.master_seed,
                    )
                )
                tmp_agt.reload(os.path.join(a_dir, agt_id))
                agent_dict[tmp_agt.id] = tmp_agt

            section.print_raw(f"Reloaded Agents: {str(agent_dict.keys())}")

            ##########
            # Reload Teams
            ##########
            t_dir = os.path.join(folder, "Teams")
            # list of directories that should be team names
            team_id_list = [
                name
                for name in os.listdir(t_dir)
                if os.path.isfile(os.path.join(t_dir, name))
            ]

            # create default teams, load with reloaded agents
            for team_id in team_id_list:
                tmp_team = TeamHistory(agent_group=None)
                tmp_team.reload(
                    team_file=os.path.join(t_dir, team_id), agent_dict=agent_dict
                )
                self.total_teams[tmp_team.id] = tmp_team

            section.print_raw(f"Reloaded Teams: {str(self.total_teams)}")

            ##########
            # Reload Environmnets
            ##########
            tmp_total_envs = dict()
            e_dir = os.path.join(folder, "Environments")
            # list of directories that should be env names
            env_paths = [
                name
                for name in os.listdir(e_dir)
                if os.path.isdir(os.path.join(e_dir, name))
            ]

            # create default envs, load with reloaded params
            for env_id in env_paths:
                # params
                #  this gets reloaded in the EnvHistory wrapper too - why?
                tmp_params = self.env_param_class(self.args)
                tmp_params.reload(folder=os.path.join(e_dir, env_id))

                # env
                tmp_env = EnvHistory(env_param=tmp_params, env_creator=self.env_class)
                tmp_env.reload(
                    folder=os.path.join(e_dir, env_id), team_dict=self.total_teams
                )
                # save so we can sort them later
                tmp_total_envs[tmp_env.id] = tmp_env

            section.print_raw(f"Reloaded Environments: {str(tmp_total_envs)}")

            ##########
            # Reload Manifest
            ##########
            with open(
                os.path.join(folder, "manifest.json"), mode="r", encoding="utf8"
            ) as f:
                manifest = json.load(f, cls=NumpyDecoder)

            # total env list, in proper order
            self.total_envs = [tmp_total_envs[e] for e in manifest["total_envs"]]

            # archived envs, in proper order
            for env_id in manifest["archived_envs"]:
                self.archived_envs[env_id] = tmp_total_envs[env_id]

            # current bracket, in proper order
            for env_id, team_id in manifest["brackets"].items():
                self.brackets[tmp_total_envs[env_id]] = self.total_teams[team_id]

            # Counters
            self.ANNECS = manifest["ANNECS"]
            self.not_ANNECS = manifest["not_ANNECS"]
            AgtHistory.id_counter = manifest["AgtHistory.id_counter"]
            TeamHistory.id_counter = manifest["TeamHistory.id_counter"]
            EnvHistory.id_counter = manifest["EnvHistory.id_counter"]

            section.print_raw(
                "Reloaded Manifest: envs, archive, brackets, and counters"
            )

            ##########
            ## Reload Manager
            ##########
            self.manager.reload(folder=folder, cur_opts=self.brackets)

            section.print_raw("Reloaded Manager")

            ##########
            # Reload Evolution
            ##########
            self.evolver.reload(folder=folder)

            section.print_raw("Reloaded Evolver")

            ##########
            # Reset global rng
            ##########
            #  this has to be last!
            rstate = manifest["random"]
            rstate[1] = tuple(rstate[1])
            random.setstate(tuple(rstate))

            section.print_raw("Reloaded Global PRNG")

    def optimize(self, epoch: int) -> None:
        """
        Agent Optimization

        This function calls the specific optimization routine for a batch of parameter
        estimates.

        Parameters
        ----------
        epoch: int
            Current main loop iteration

        Side-Effects
        ------------
        Many
            All side effects occur in "EvolutionTemplate()"

        Returns
        -------
        None
        """

        with SectionLogger(logger, "Optimization", f"Epoch: {epoch}") as section:
            # call out to optimization function
            results = self.manager.optimize_chunk(
                tasks=self.brackets.items(), epoch=epoch
            )

            for stats, (env, team) in zip(results, self.brackets.items()):
                # calculate average of average results
                #  mean of a list of lists
                #  this averages every agents' score into a single number
                #  basically a cheap escape
                eval_mean = np.mean(list(stats.values()))

                # store most recent evals
                env.stats.recent_scores.append(eval_mean)
                # transfer threshold is best of N best scores
                env.stats.transfer_threshold = max(env.stats.recent_scores)

                # lifespan best score and agent
                #  only checks the agent after a whole lest of optimization, so we miss
                #  the agents/scores inbetween. Ergo, there is a chance that
                #  transfer_threshold is transiently better than best_score
                if env.stats.best_score < eval_mean:
                    env.stats.best_score = eval_mean
                    env.stats.best_team = team.copy()
                    # this team is the same as "team", but gets a new id because of the copy
                    #  THIS COPY STATEMENTS ISREQUIRED HERE!!!!
                    #  It must be an independent agent

                    # add to stats
                    self.total_teams[env.stats.best_team.id] = env.stats.best_team

            section.print_time()

    def transfer(self, epoch: int) -> None:
        """EvolutionTemplate Single-Transfer Function

        This function transfers agents within a team, so that the team is always
        comprised of the best agent.

        Parameters
        ----------
        epoch: int
            Current main loop iteration

        Side-Effects
        ------------
        Many
            Teams switch

        Returns
        -------
        None
        """

        with SectionLogger(logger, "Transfer", f"Epoch: {epoch}") as section:
            # if debug, check current bracket against new bracket (below)
            logger.debug("Bracket before transfer:" + str(self.brackets))

            # run eval to prep transfers
            results = self.manager.evaluate(
                tasks=self.brackets.items(), epoch=epoch, verbose=True
            )

            # loop over results and bracket
            for stats, (env, team) in zip(results, self.brackets.items()):
                # average eval for each agent
                eval_means = {k: v.eval_returns_mean for k, v in stats.items()}
                # get name of best agent and its score
                best_agent_name = max(eval_means, key=eval_means.__getitem__)
                best_mean = eval_means[best_agent_name]

                # loop over whole team, replace other agents if the best one is
                #  "better enough"
                #  would a t-test make sense here?
                #  Gonna use 3x standard deviation of the means
                std_mean_3 = 3 * np.std(list(eval_means.values()))
                for agt in team.keys():
                    # check if best is "better enough"
                    if best_mean > (eval_means[agt] + std_mean_3):
                        # log
                        logger.info(
                            f"Team {str(team)} is replaceing agent {agt} with agent {best_agent_name}"
                        )
                        # replace
                        team[agt] = team[best_agent_name].copy()
                        # NOTE: should we be creating a new team here?
                        #       since the agents are technically changed

            # NOTE: The following is not technically correct.
            #       This is where we have this in poet, but really we should update
            #       the total_teams dict whenever we copy and save a new team. I
            #       think this is protecting us from oversights elsewhere, by being
            #       the final function before we checkpoint.

            # check if we're created new teams
            #  whenever a team transfers or beats the best score of a previous team,
            #  new teams are created to log those events.
            # Here, we add those teams to the total team dict
            # loop over active envs
            for env in self.brackets.keys():
                # loop over possible new teams
                #  we don't need their paired team in the bracket because it is
                #  identical to "stats.team" by definition
                for team in [env.stats.team, env.stats.best_team]:
                    # check if it is novel
                    if team.id not in self.total_teams:
                        self.total_teams[team.id] = team

            # if debug, check new bracket against original bracket (above)
            logger.debug("Bracket after transfer:" + str(self.brackets))

            section.print_time()
