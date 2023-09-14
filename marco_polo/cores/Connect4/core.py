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


import argparse  # for type hinting
import json
import logging
import os
import random
import subprocess
import time

from collections import OrderedDict
from typing import Any, Callable

import numpy as np

from marco_polo.tools.iotools import NumpyDecoder, NumpyEncoder
from marco_polo.tools.section_logging import SectionLogger
from marco_polo.tools.wrappers import AgtHistory, EnvHistory, TeamHistory

logger = logging.getLogger(__name__)

from marco_polo.optimizers.uber_es import get_opt_class, get_model

from ...envs.PettingZoo.env import PettingZooEnv, PettingZooEnvParams
from marco_polo.tools.wrappers import MP_AEC_to_Parallel


def get_env() -> Callable[..., Any]:
    return MP_AEC_to_Parallel(PettingZooEnv())


################################################################################
## Core Class
################################################################################
class Core:
    """
    Connect 4 Core

    Simple core for multiplayer Connect 4 from the PettingZoo repo (https://github.com/Farama-Foundation/PettingZoo).
    Can be used as a base for other PettingZoo environments and for league play.

    Attributes
    ----------
    args : argparse.Namespace
        Simulation parameters
    """

    def __init__(self, args: argparse.Namespace) -> None:
        # self.env_factory = Env_Factory(Game_Params)  # Returns something that creates niches? This is going to be a lot of the code that we we have written at this point.
        # self.model_factory = MLP_Factory(Model_Pramas)	 # Returns something that creates models?
        self.args = args
        os.makedirs(os.path.join(args.log_file, "Rollouts"), exist_ok=True)
        os.makedirs(os.path.join(args.log_file, "Videos"), exist_ok=True)

        Manager = get_opt_class(args)
        # The manager expects any incoming args, as well as a list of gym envs
        self.manager = Manager(args)
        self.manager.verbose = False

        self.iterations = 0

        ## Create intial Env Params
        starting_env = EnvHistory(
            PettingZooEnvParams(), get_env, agents=["player_0", "player_1"]
        )

        self.envs = [starting_env]

        # Here, we're going to use an ordered dictionary to keep track of matchings
        # agent_group: OrderedDict[Role, AgtHistory] = OrderedDict()
        agent_group = OrderedDict()

        for agent in starting_env.agents:
            agent_group[agent] = AgtHistory(
                get_agent(starting_env, agent, args, args.master_seed)
            )

        initial_agents = TeamHistory(agent_group=agent_group)

        self.brackets = OrderedDict()
        self.brackets[starting_env] = initial_agents

    ########################################
    ## poet_algo.PopulationManager.evolve_population() Funcs
    ########################################
    def iteration_update(self, iteration: int) -> None:
        # safe-keeping for Curiosity currently
        pass

    def reproduce(self, iteration: int) -> None:
        pass

    def optimize(self, iteration: int) -> None:
        opt_time = time.time()
        tasks = list(self.brackets.items())
        self.manager.optimize_chunk(tasks, epoch=1)

        # self.manager.evaluate(tasks)

        # logger.info(f"optimize() Full Opt_Chunk Time:{time.time() - opt_time}")

    def transfer(self, iteration: int) -> None:
        pass
        # self.brackets = AllActiveBestScore(self.manager, self.brackets, True, "role1")

    def evolve_population(self, start_epoch: int, sim_start: float) -> None:
        self.optimize(start_epoch)
