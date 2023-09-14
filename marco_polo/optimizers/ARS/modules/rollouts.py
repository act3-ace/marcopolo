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


"""ARS Simulation Rollouts"""

import logging
from collections import namedtuple
import argparse  # for type hinting
from collections.abc import Callable  # for type hinting
from typing import Any  # for type hinting

import numpy as np
from numpy.random import PCG64DXSM, Generator

from marco_polo.optimizers.uber_es.modules.rollouts import _simulate_basic
from marco_polo.optimizers.ARS.modules.opt import ESTeamWrapper  # for type hinting
from marco_polo.tools.types import Role  # for type hinting
from marco_polo.envs._base.env_params import EnvParams  # for type hinting
from marco_polo.optimizers.ARS.modules.result_objects import POResult

logger = logging.getLogger(__name__)


################################################################################
## Aux Objects
################################################################################


########################################
## Optim Funcs
########################################
def run_optim_batch_distributed(
    num_rollouts: int,
    rs_seed: int,
    env_params: EnvParams,
    env_creator_func: Callable[..., Any],
    team: ESTeamWrapper,
    setup_args: argparse.Namespace,
    noise_std: float,
) -> dict[Role, POResult]:
    """
    Distributed Evaluation Routine

    This fuction is design for asynchronous evaluations. It does not track frames
    or output images, just tracks actions, rewards, and simulation length. It calls
    "_simulate_basic()" to perform simulations, then handles results for each agent.

    Parameters
    ----------
    num_rollouts : int
        How many evaluations to perform
        The actual amount is 2x this, since we eval + and - gradient perturbations
    rs_seed : int
        Random seed for PRNG
    env_params : EnvParams
        Parameter object specifying the environment
    env_creator_func : Callable[..., Any]
        Function to create the environment from the CPPN
        This is an artificant of pickle, because the full env can't be sent over pipes
    team : ESTeamWrapper
        Dictionary of agents to play
    setup_args : argparse.Namespace
        Input args for the agent wrapper
    noise_std : float
        Standard deviation when drawing gradient perturbations

    Side-Effects
    ------------
    None
        When run asynchronously, everything is copied to this func, so no side-effects

    Returns
    -------
    dict[role, POResult]
        Dictionary of POResult objects for each agent
    """
    # set new prng
    random_state = Generator(PCG64DXSM(seed=rs_seed))
    # draw seeds for weight shifting
    seeds = random_state.integers(low=0, high=(2**63), size=(num_rollouts,))

    # setup simulator
    env = env_creator_func()
    env.augment(env_params)

    # turn on observation saving
    for agt in team.values():
        agt.set_obs_flag(flag=True)

    # simulation return objects
    returns = {role: np.zeros(shape=(num_rollouts, 2)) for role in team.roles()}
    observations = {role: [None] * num_rollouts for role in team.roles()}
    lengths = np.zeros(shape=(num_rollouts, 2), dtype="int")
    agt_seeds = []
    results = dict()

    # loop over rollouts, do + and - point estimates
    for i, seed in enumerate(seeds):
        # reset observation buffer
        for agt in team.values():
            agt.get_obs_buf().reset()

        # shift weights and store seed
        agt_seeds.append(team.shift_weights(noise_std=noise_std, seed=seed))

        # first rollout
        r, lengths[i, 0], _ = _simulate_basic(
            env=env, team=team, random_state=random_state
        )

        # update returns
        for role in team.roles():
            returns[role][i, 0] = r[role]

        # shift weights in other direction
        _ = team.shift_weights(noise_std=-noise_std, seed=seed)

        # second rollout
        r, lengths[i, 1], _ = _simulate_basic(
            env=env, team=team, random_state=random_state
        )

        # store returns and observations
        for role, agt in team.items():
            returns[role][i, 1] = r[role]
            observations[role][i] = agt.get_obs_buf().copy()

    # turn off observation saving
    for agt in team.values():
        agt.set_obs_flag(flag=False)

    # loop over agents, fill return
    for role in team.keys():
        results[role] = POResult(
            returns=returns[role],
            noise_inds=[a[role] for a in agt_seeds],
            lengths=lengths,
            rollouts=[],
            obs_buf=observations[role],
        )

    # return results
    return results
