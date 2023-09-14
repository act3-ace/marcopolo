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


"""Uber ES Simulation Rollouts"""

import argparse
import logging
import os
import subprocess
from collections.abc import Callable
from typing import Any, Optional, Union

import imageio
import numpy as np
from numpy.random import PCG64DXSM, Generator

from marco_polo.envs._base.env_params import EnvParams

# from marco_polo.optimizers.uber_es.modules.model import Model
from marco_polo.base_classes.model import BaseModel  # for type hinting
from marco_polo.optimizers.uber_es.modules.opt import ESTeamWrapper  # for type hinting
from marco_polo.optimizers.uber_es.modules.result_objects import EvalResult, POResult
from marco_polo.tools.wrappers import EnvHistory, TeamHistory
from marco_polo.tools.types import Role  # for type hinting

logger = logging.getLogger(__name__)


################################################################################
## Aux Multithreading Funcs
################################################################################
# # this has to be here!
# #  "globals" are actually only module global, not program or namespace global.
# #  so, to share globals between functions, they must all come from the
# #  same module and be imported to wherever they are getting used.

# # also, this is not necessary
# #  This is a cheap trick to leave a reference to shared objects in each worker,
# #  then we don't have to pass the objects as arguments.
# #  The standard way is to pass shared objects as arguments.
# #  There is no behavioral difference between these methods.
# def initialize_worker_fiber(arg_thetas, arg_niches, arg_tasks):
#     global tasks, thetas, niches
#     #from .noise_module import noise
#     thetas = arg_thetas
#     niches = arg_niches
#     tasks = arg_tasks


################################################################################
## Eval Funcs
################################################################################
########################################
## Eval Funcs
########################################
def run_viz_distributed(
    num_rollouts: int,
    rs_seed: int,
    env_params: EnvParams,
    env_creator_func: Callable[..., Any],
    env_id: str,
    team: TeamHistory,
    epoch: Optional[int],
    vidpath: str,
    frame_skip: int,
) -> None:
    """
    Distributed Vizualization Routine

    This fuction is design for asynchronous simulation vizualization. It takes a
    single team, performs a single rollout, and saves the video. There is no return
    and it is designed to be non-blocking.

    Parameters
    ----------
    num_rollouts: int
        number of visualization runs to do
        (this will usually be 1)
    rs_seed : int
        Random seed for PRNG
    env_params : EnvParams
        environment parameter object specifying the environment
    env_creator_func : Callable[..., Any]
        Function to create the environment from the environment parameters
        This is an artificant of pickle, because the full env can't be sent over pipes
    env_id : str
        ID of environment
    team : TeamHistory
        dictionary of agents to play
    epoch : Optional[int]
        Current simulation time
    vidpath : str
        Full path to directory for video storage
    frame_skip : int
        How many frames to skip when recording the gif

    Side-Effects
    ------------
    None
        When run asynchronously, everything is copied to this func, so no side-effects

    Returns
    -------
    None
        Saved gif to directory
    """
    # set new prng
    random_state = Generator(PCG64DXSM(seed=rs_seed))

    # setup simulator
    env = env_creator_func()
    env.augment(env_params)

    # set file name
    run_name = f"Epoch:{epoch}_envID:{env_id}"

    # simulate and write .gif
    for _ in range(num_rollouts):
        _simulate_viz(
            env=env,
            team=team,
            random_state=random_state,
            run_name=run_name,
            vidpath=vidpath,
            frame_skip=frame_skip,
        )


def _simulate_viz(
    env: EnvHistory,
    team: TeamHistory,
    random_state: Generator,
    run_name: str = "",
    vidpath: str = "",
    frame_skip: int = 1,
    batch_render: bool = False,
) -> None:
    """
    Simulation Function for Vizualization Only

    This function is similar to "_simulate_basic()" but specialized for
    vizualization only. It tracks frames and outputs a gif, but does not return
    rewards or observations.

    Parameters
    ----------
    env : EnvHistory
        Built from parameters and a gym creation function
    team : TeamHistory
        Agent team playing the environment
    random_state : Numpy Generator
        PRNG
    run_name : str, default=""
        Name of gif
    vidpath : str, default=""
        Full path to file output
    frame_skip : int, default=1
        render frame if i_framet % frame_skip == 0
    batch_render: bool, default=False
        whether to use batch rendering (i.e. run entire run first,
        then render the results). Requires that the renderer
        supports the render_all function.

    Side-Effects
    ------------
    None
        When run asynchronously, everything is copied to this func, so no side-effects

    Returns
    -------
    None
        Saved gif to directory
    """
    max_episode_length = getattr(env, "max_episode_steps", 2000)

    # set environment seed
    env.seed(random_state.integers(2**31 - 1, dtype=int))

    # initial obs, and to clear the env
    obs, _ = env.reset()

    # rendering data structures
    render_data = []
    frames = []
    result = [max_episode_length, False]  # default result

    # Perform Run
    for t in range(max_episode_length):
        # if appropriate time, save frame
        if t % frame_skip == 0:
            if batch_render:
                render_data.append(env.bundle_step_data())
            else:
                frames.append(env.render())

        # action object for each agent
        action = dict()

        # loop over agents, send obs, get action
        for role, agent in team.items():
            action[role] = agent.get_action(obs[role], t=t, mean_mode=False)

        # send actions to game, get observations and etc
        obs, _, term, trunc, _ = env.step(action)

        # check if done
        if all([term[k] or trunc[k] for k in term.keys()]):
            break

    # Lets be sure to always capture the last frame.
    if batch_render:
        render_data.append(env.bundle_step_data())
    else:
        frames.append(env.render())

    if batch_render:
        frames = env.render_all(data=render_data, result=result)

    # if there are any frames, store as video
    if frames:
        filename = os.path.join(vidpath, f"{run_name}.gif")
        write_video(filename=filename, frames=frames, frame_skip=frame_skip)

    # close viewer, if applicable
    env.render(close=True)


def write_video(filename: str, frames: list[Any], frame_skip: int) -> None:
    """Write the frames to a video

    Parameters
    ----------
    filename: str
        name of file to write
    frames: list[Any]
        frames of video to write
    frame_skip: int
        frame skipped in output - used to set the frame rate

    Side-Effects
    ------------
    None

    Returns
    -------
    None
        Saved gif to directory
    """
    # write gif
    #  original mimwrite() took fps, which we defined as 60/frame_skip
    #  update takes "duration", defined as 1000/fps
    #  therefore, we implement duration=int(1000*frame_skip/60)
    imageio.mimwrite(uri=filename, ims=frames, duration=int(1000 * frame_skip / 60))

    # compress gif
    # using direct function call because I can set the optimization that way
    # don't need to remove extensions - imagio doesn't add any
    # no point in reducing colorspace (--colors xx)- BPW videos have <32 colors anyway
    # no need to increase lossiness (--lossy=xx)- not enough going on in BPW
    # References:
    #  imagio notes - https://imageio.readthedocs.io/en/stable/examples.html#optimizing-a-gif-using-pygifsicle
    #  https://github.com/LucaCappelletti94/pygifsicle
    #   pygifesicle just uses subprocess to call gifsicle - no point in using
    #  https://www.lcdf.org/gifsicle/
    subprocess.call(
        ["gifsicle", "--no-warnings", "--optimize=2", filename, "--output", filename]
    )

    # debugging
    logger.debug(f"capturing gif: {filename}")


########################################
## Eval Funcs
########################################
def run_eval_batch_distributed(
    num_rollouts: int,
    rs_seed: int,
    env_params: EnvParams,
    env_creator_func: Callable[..., Any],
    team: dict[Role, BaseModel],
) -> dict[Role, EvalResult]:
    """
    Distributed Evaluation Routine

    This fuction is design for asynchronous evaluations. It does not track frames
    or output images, just tracks actions, rewards, and simulation length. It calls
    "_simulate_basic()" to perform simulations, then handles results for each agent.

    Parameters
    ----------
    num_rollouts : int
        How many evaluations to perform
    rs_seed : int
        Random seed for PRNG
    env_params : EnvParams
        Environment parameter object specifying the environment
    env_creator_func : Callable[..., Any]
        Function to create the environment from the envrionment parameters
        This is an artificant of pickle, because the full env can't be sent over pipes
    team : dict[role, BaseModel]
        Dictionary of agents to play

    Side-Effects
    ------------
    None
        When run asynchronously, everything is copied to this func, so no side-effects

    Returns
    -------
    dict[role, EvalResult]
        Dictionary of EvalResult objects for each agent
    """
    # set new prng
    random_state = Generator(PCG64DXSM(seed=rs_seed))

    # setup simulator
    env = env_creator_func()
    env.augment(env_params)

    # simulation return objects
    returns = {role: np.zeros(shape=(num_rollouts,)) for role in team.keys()}
    lengths = np.zeros(shape=(num_rollouts,), dtype="int")
    results = {}

    # loop over number of rollouts
    for i in range(num_rollouts):
        # simulate
        # store results
        r, lengths[i], _ = _simulate_basic(
            env=env, team=team, random_state=random_state
        )
        # update returns
        for role in team.keys():
            returns[role][i] = r[role]

    # loop over agents, fill EvalResult
    for role in team.keys():
        results[role] = EvalResult(
            returns=returns[role],
            lengths=lengths,
            eval_returns_mean=returns[role].mean(),
            eval_returns_max=returns[role].max(),
            eval_returns_min=returns[role].min(),
        )

    # return results
    return results


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
    Distributed Optimization Routine

    This fuction is designed for asynchronous optimization. It does not track frames
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
        Environment parameter object specifying the environment
    env_creator_func : Callable[..., Any]
        Function to create the environment from the environment parameters
        This is an artificant of pickle, because the full env can't be sent over pipes
    team : ESTeamWrapper
        Dictionary of agents to play
    setup_args : args.Namespace
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

    # simulation return objects
    returns = {role: np.zeros(shape=(num_rollouts, 2)) for role in team.roles()}
    lengths = np.zeros(shape=(num_rollouts, 2), dtype="int")
    agt_seeds = []
    results = dict()

    # loop over rollouts, do + and - point estimates
    for i, seed in enumerate(seeds):
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

        # store returns
        for role in team.roles():
            returns[role][i, 1] = r[role]

    # loop over agents, fill return
    for role in team.roles():
        results[role] = POResult(
            returns=returns[role],
            noise_inds=[a[role] for a in agt_seeds],
            lengths=lengths,
            rollouts=[],
        )

    # return results
    return results


def _simulate_basic(
    env: Any, team: Union[ESTeamWrapper, dict[Role, BaseModel]], random_state: Generator
) -> tuple[dict[Role, float], int, list[dict[Role, Any]]]:
    """
    Simulation Function for Evalutions Only

    This function is similar to "_simulate_viz()" but specialized for
    evaluation only. It tracks actions, rewards, and simulation time, but does
    not track frames or output video.

    Parameters
    ----------
    env : Any
        Built from parameters and a gym creation function
    team : Union[ESTeamWrapper, dict[Role, BaseModel]]
        Team playing the environment
    random_state : Numpy Generator
        PRNG

    Side-Effects
    ------------
    None
        When run asynchronously, everything is copied to this func, so no side-effects

    Returns
    -------
    tuple[dict[Role, float], int, list[dict[Role, Any]]]
        total_reward : dict[role, score]
            Dictionary of scores for each agent
        t : int
            Simulation time when finished
        actions : list[dict[role, action]]
            List of actions from each agent on each timestep
    """
    max_episode_length = getattr(env, "max_episode_steps", 2000)

    # set environment seed
    env.seed(random_state.integers(2**31 - 1, dtype=int))

    # initial obs, and to clear the env
    obs, _ = env.reset()

    # setup return things
    actions = []
    total_reward = {role: 0.0 for role in team.roles()}

    # function that does the step
    take_step = env.step

    # check if there is an optimized version for no visualization case
    # if so, use that instead
    take_step_no_viz = getattr(env, "step_noviz", None)
    if callable(take_step_no_viz):
        take_step = take_step_no_viz

    # Perform Run
    for t in range(max_episode_length):
        # action object for each agent
        action = dict()

        # loop over agents, send obs, get action
        for role, agent in team.items():
            action[role] = agent.get_action(obs[role], t=t, mean_mode=False)

        # store actions for later
        actions.append(action)

        # send actions to game, get observations and etc
        obs, reward, term, trunc, info = take_step(action)

        # update reward
        for role in total_reward:
            total_reward[role] += reward[role]

        # check if done
        if all([term[k] or trunc[k] for k in term.keys()]):
            break

    # return
    return total_reward, t, actions
