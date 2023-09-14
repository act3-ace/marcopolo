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

"""Single-Player POET Evolution Core """

import argparse  # for type hinting
import json
import logging
import os
import random
import subprocess
import time
from collections import OrderedDict
from collections.abc import Callable  # for type hinting
from typing import Any, cast  # for type hinting
import numpy as np

from marco_polo.algorithms.poet import AllActiveBestScore, PataECEvolution, PoetOptLoop
from marco_polo.envs._base.env_params import EnvParams  # for type hinting
from marco_polo.tools.iotools import NumpyDecoder, NumpyEncoder
from marco_polo.tools.section_logging import SectionLogger
from marco_polo.tools.types import Role, PathString  # for type hinting
from marco_polo.tools.wrappers import AgtHistory, EnvHistory, TeamHistory

logger = logging.getLogger(__name__)


################################################################################
## Core Class
################################################################################
class Core:
    """
    ES Core Manager

    This core sets up multiprocessing using an env created from the passed env factory.
    The algorithm is from the ePOET paper, instantiates a custom NN, and optimizes using
    the evolutionary strategy from openAI/Uber. This is a re-org of the ePOET paper with
    no algorithmic changes.

    This is the most basic (single player) Core, from which other Cores, implementing different
    algorithms, can be created.

    Attributes
    ----------
    args : argparse.Namespace
        Simulation parameters
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
    brackets : OrderedDict[env, team]
        Pairing of active environments/teams
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
        teams(s) and environment(s).

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
        self.env_class = env_factory(args)
        self.env_param_class = env_param_factory(args)

        # Setup manager with any incoming args
        Manager = manager_factory(args=args)
        self.manager = Manager(setup_args=args)

        # Setup the environment evolution class
        self.role_name = self._get_role_name(self.env_class, self.env_param_class)
        self.evolver = PataECEvolution(
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
    def _get_role_name(
        self, EnvCreator: Callable[..., Any], env_param_class: Callable[..., EnvParams]
    ) -> Role:
        tmp = EnvCreator()
        tmp.augment(params=env_param_class(self.args))
        if hasattr(tmp, "possible_agents"):
            if len(tmp.possible_agents) > 1:
                raise ValueError("Too many players, POET expects a 1 player game.")
            return cast(
                Role, tmp.possible_agents[0]
            )  # mypy can't figure out the type without cast

        raise AttributeError(
            "env.possible_agents not found, please ensure your env implements the inferface properly."
        )

    def _update_model_params(self, env: Any) -> None:
        """Update input and output size based on env class."""
        model_params = self.args.model_params
        if type(env.observation_spaces) is dict:
            a = env.possible_agents[0]
            model_params["input_size"] = np.prod(env.observation_spaces[a].shape)
            model_params["output_size"] = np.prod(env.action_spaces[a].shape)
        else:
            model_params["input_size"] = np.prod(env.observation_spaces.shape)
            model_params["output_size"] = np.prod(env.action_spaces.shape)

    def _create_initial_env_teams(
        self, EnvCreator: Callable[..., Any], env_param_class: Callable[..., EnvParams]
    ) -> None:
        """Create the initial environments and teams."""

        # temporary environment to update model params
        tmp_env = EnvCreator()
        tmp_env.augment(params=env_param_class(self.args))
        self._update_model_params(tmp_env)

        # Make sure we don't start with more than the maximum number of envs
        n_start_envs = min(self.args.max_active_envs, self.args.num_start_envs)

        for i in range(n_start_envs):
            seed = self.args.master_seed + i

            # create env
            env_params = env_param_class(self.args)
            env = EnvCreator()
            env.augment(params=env_params)
            env_history = EnvHistory(env_param=env_params, env_creator=EnvCreator)

            # create team - in this case it has one agent in a dict
            model = self.model_factory(
                env=env, role=self.role_name, args=self.args, seed=seed
            )
            agent = AgtHistory(model)
            agent_group = {self.role_name: agent}
            team = TeamHistory(agent_group=agent_group)

            # pair env/team and store
            env_history.stats.team = team
            self.total_teams[team.id] = team
            self.total_envs.append(env_history)
            self.brackets[env_history] = team

    def checkpoint(self, name: str) -> None:
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
        with SectionLogger(logger, "Checkpointing", f"Epoch: {name}") as section:
            filename = self.args.logtag + "cp-" + name
            folder = os.path.join(self.args.log_file, "Checkpoints", filename)
            if os.path.isdir(folder):
                section.print_raw(
                    f"Directory already exists at {folder}. Overwriting checkpoint."
                )
            else:
                section.print_raw(f"Saving checkpoint to {folder}")

            os.makedirs(folder, exist_ok=True)

            with open(
                os.path.join(folder, "args.json"), mode="w", encoding="utf8"
            ) as f:
                json.dump(self.args.__dict__, f, cls=NumpyEncoder)

            manifest: dict[str, Any] = {}
            # manifest["args"] = dict(self.args.__dict__)
            logger.debug(f"Bracket Check: {self.brackets}")

            manifest["total_envs"] = [env.id for env in self.total_envs]
            manifest["total_teams"] = list(self.total_teams.keys())
            manifest["archived_envs"] = list(self.archived_envs.keys())
            manifest["brackets"] = {
                env.id: team.id for env, team in self.brackets.items()
            }
            manifest["ANNECS"] = self.ANNECS
            manifest["not_ANNECS"] = self.not_ANNECS
            manifest["AgtHistory.id_counter"] = AgtHistory.id_counter
            manifest["TeamHistory.id_counter"] = TeamHistory.id_counter
            manifest["EnvHistory.id_counter"] = EnvHistory.id_counter
            manifest["random"] = random.getstate()

            file_path = os.path.join(folder, "manifest.json")
            with open(file_path, mode="w", encoding="utf-8") as f:
                json.dump(manifest, f, cls=NumpyEncoder)

            ## Checkpoint Environmnets
            for env in self.total_envs:
                env.checkpoint(folder)

            ## Checkpoint Teams and Agents
            arDict = dict()
            for team in self.total_teams.values():
                # checkpoint team
                team.checkpoint(folder)
                for role, agtHist in team.items():
                    # checkpoint agent
                    agtHist.checkpoint(folder)
                    # add agent id/role to dict
                    arDict[agtHist.id] = role

            # Log agent id/role mapping
            #  This is primarily used in multi-agent reloads
            file_path = os.path.join(folder, "Agents", "ID_Role.json")
            with open(file_path, mode="w", encoding="utf-8") as f:
                json.dump(arDict, f)

            ## Checkpoint Manager
            self.manager.checkpoint(folder=folder, cur_opts=self.brackets.items())

            ## Checkpoint Poet Tools
            self.evolver.checkpoint(folder)

            # Compress checkpoint if requested
            if self.args.checkpoint_compression:
                section.print_raw("Compressing checkpoint")
                self._compress_checkpoint(folder)

    def _compress_checkpoint(self, folder: PathString) -> None:
        """Use lbzip2 to compress checkpoint folder using external tool."""
        # NOTES on json compression, if we ever go that route
        #  it'll probably be faster than this, since it compresses from memory
        # NOTE: https://martinheinz.dev/blog/57
        # NOTE: https://medium.com/@busybus/zipjson-3ed15f8ea85d
        # NOTE: https://janakiev.com/blog/python-json/
        # NOTE: https://stackoverflow.com/questions/17742789/running-multiple-bash-commands-with-subprocess

        # NOTE: https://stackoverflow.com/questions/45621476/python-tarfile-slow-than-linux-command
        #       This is why we subprocess instead of using the python tarfile module
        # NOTE: https://anaconda.org/conda-forge/lbzip2
        ## Compress checkpoint
        # build command
        # -C change to the following directory
        # -c output to this place (sent to stdout here)
        # -f read from this file
        # | lbzip2 pipe stdin to lbzip2
        # -9 lbzip2 level 9
        # -n number of cores to use
        # > output to here
        tar_args = (
            "tar -C "
            + os.path.dirname(p=folder)
            + " --remove-files -cf - "
            + os.path.basename(p=folder)
            + " | lbzip2 -9 -n "
            + str(self.args.num_workers)
            + " > "
            + folder
            + ".tar.bz2"
        )

        # send to shell
        subprocess.run(args=tar_args, shell=True)

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
            agent_dir = os.path.join(folder, "Agents")
            # list of directories that should be agent names
            agent_paths = [
                name
                for name in os.listdir(agent_dir)
                if os.path.isdir(os.path.join(agent_dir, name))
            ]

            # blank env for agent params
            agt_env = self.env_class()
            agt_env.augment(params=self.env_param_class(self.args))

            # create default agents, fill with reloaded params
            for agent_id in agent_paths:
                # agent
                tmp_agent = AgtHistory(
                    self.model_factory(
                        env=agt_env,
                        role=self.role_name,
                        args=self.args,
                        seed=self.args.master_seed,
                    )
                )
                tmp_agent.reload(os.path.join(agent_dir, agent_id))
                agent_dict[tmp_agent.id] = tmp_agent

            section.print_raw(f"Reloaded Agents: {str(agent_dict.keys())}")

            ##########
            # Reload Teams
            ##########
            # list of directories that should be team names
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

    def epoch_update(self, epoch: int) -> None:
        """
        Currently does nothing, here for reasons?

        Parameters
        ----------
        epoch: int
            Current simulation time

        Side-Effects
        ------------
        None

        Returns
        -------
        None
        """
        # safe-keeping for Curiosity currently

    def reproduce(self, epoch: int) -> None:
        """
        Basic Reproduction Function

        This function handles the current environment/team pairings, tracks the
        archived environments, total environments/teams, and has some basic statistics.
        The actual work is passed down to Evolve()

        Parameters
        ----------
        epoch: int
            Current simulation time

        Side-Effects
        ------------
        Many
            This function directly updates internal tracking variables and statistics
            Evolve() modifies the environments' and teams' internal states

        Returns
        -------
        None
        """

        with SectionLogger(logger, "Evolution", f"Epoch: {epoch}") as section:
            (
                new_envs,
                new_teams,
                self.brackets,
                AN,
                no_AN,
                to_archive,
            ) = self.evolver.Evolve(
                bracket=self.brackets,
                archived_envs=self.archived_envs,
                epoch=epoch,
                repro_threshold=self.args.repro_threshold,
                max_active_envs=self.args.max_active_envs,
                max_children=self.args.num_proposal_envs,
                max_admitted=self.args.max_admitted_envs,
            )

            # update internal metrics
            self.total_envs += new_envs
            self.total_teams.update(new_teams)
            self.ANNECS += AN
            self.not_ANNECS += no_AN
            self.archived_envs.update(to_archive)

            # log stats
            section.print_status_banner("Summary")
            for line in self.get_summary():
                section.print_raw(line)

    def get_summary(self) -> list[str]:
        """Return summary of the system

        example usage might be:
        summary = self.get_summary()
        for line in summary:
            print(line)

        Returns
        -------
        list[str]
            list of strings describing the system. Each item is a single line.
        """
        return [
            f"Total Environments: {str(self.total_envs)}",
            f"Archived Environments: {str(self.archived_envs)}",
            f"Current Bracket: {str(self.brackets)}",
            f"ANNECS: {self.ANNECS}",
            f"Not ANNECS: {self.not_ANNECS}",
        ]

    def optimize(self, epoch: int) -> None:
        """
        Team Optimization

        This function calls the specific optimization routine for a batch of parameter
        estimates.

        Parameters
        ----------
        epoch: int
            Current main loop iteration

        Side-Effects
        ------------
        Many
            All side effects occur in "PoetOptLoop()"

        Returns
        -------
        None
        """

        with SectionLogger(logger, "Optimization", f"Epoch: {epoch}") as section:
            # call out to optimization function
            PoetOptLoop(
                tasks=list(self.brackets.items()),
                epoch=epoch,
                manager=self.manager,
                verbose=True,
            )
            # NOTE: Are we missing a total_teams update in here?
            #       We might be safe because the transfers do it.

            section.print_time()

    def transfer(self, epoch: int) -> None:
        """
        Peform Transfer Mechanism

        This work is passed down to "AllActiveBestScore()".

        Parameters
        ----------
        epoch: int
            Current main loop iteration

        Side-Effects
        ------------
        Many
            This function directly updates the env/team pairing bracket
            "AllActiveBestScore()" modifies environment/team internals

        Returns
        -------
        None
        """

        with SectionLogger(logger, "Transfer", f"Epoch: {epoch}") as section:
            logger.debug("Bracket before transfer:" + str(self.brackets))

            AllActiveBestScore(
                manager=self.manager,
                bracket=self.brackets,
                transfer_role=self.role_name,
                epoch=epoch,
            )

            logger.debug("Bracket after transfer:" + str(self.brackets))

            # NOTE: The following is not technically correct.
            #       This is where we put it ages ago, but really we should update
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

            section.print_time()

    def evolve_population(self, start_epoch: int = 0, sim_start: float = 0.0) -> None:
        """
        Main Optimization Loop

        For each step of the epoch evolves the env's, optimizing the functions,
        and evalutes the transfer potential. All actions are implemented by the Core
        object, this function simply calls them in the appropriate order.

        Parameters
        ----------
        start_epoch : int, default=0
            Number of start_epoch to start on, important for reloads
        sim_start : float, default=0.0
            Start time of the simulation

        Side-Effects
        ------------
        None

        Returns
        -------
        None
            Output is written to disk in a directory specific within args.
        """

        # Main Loop
        #  Algorithm 2 in POET paper (https://doi.org/10.48550/arXiv.1901.01753)
        for ep in range(start_epoch, self.args.poet_epochs):
            section = SectionLogger(logger, "Epoch", f"Epoch: {ep}", print_start=False)

            # Environment Evolution Step
            self.epoch_update(epoch=ep)

            # Reproduction Step
            if (ep > 0) and ((ep % self.args.reproduction_interval) == 0):
                self.reproduce(epoch=ep)

            # Optimization Step
            #  this function optimizes for n iterations until the transfer step
            #  this is a change from the original algorithm
            self.optimize(epoch=ep)

            # Transfer Step
            self.transfer(epoch=ep)

            # Checkpoint
            if (ep % self.args.checkpoint_interval) == 0:
                self.checkpoint(name=str(ep))

            # log epoch
            epoch_end = time.time()
            section.print_status_banner("")
            section.print_time()  # for this epoch only
            section.print_raw(f"Simulation Time: {(epoch_end - sim_start):.2f} seconds")
            section.print_end_banner()

        self.report()

    def report(self) -> None:
        report_filename = f"report_{time.time()}.rpt"
        file_path = os.path.join(self.args.log_file, report_filename)
        with open(file_path, mode="w", encoding="utf-8") as file:
            file.writelines("Environments: \n")

            for env in self.total_envs:
                out = f"Env_ID: {env.id}, "

                for k, v in env.stats.__dict__.items():
                    out += f"{k}: {str(v)}, "

                file.writelines(out + "\n")

            file.writelines("\nTeams: \n")

            for team in self.total_teams.values():
                out = f"Team_ID: {team.id}, "
                for role, agent in team.items():
                    theta_hash = hash(agent.get_theta().tobytes())
                    out += f"{role}: {str(agent.id)} Theta Hash: {theta_hash}, "

                file.writelines(out + "\n")
