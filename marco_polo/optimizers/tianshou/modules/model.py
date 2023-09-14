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


# import pickle
import os

# from argparse import ArgumentParser
import argparse
from typing import Any, Union
import gymnasium as gym
import numpy as np
import copy

# import torch
from tianshou.data import Batch  # type: ignore[import]
from pettingzoo.utils.env import ParallelEnv  # type: ignore[import]
from tianshou.policy import BasePolicy, PPOPolicy  # type: ignore[import]
from tianshou.utils.net.common import Net  # type: ignore[import]
from tianshou.utils.net.continuous import ActorProb, Critic  # type: ignore[import]

import torch
from torch import nn
from torch.distributions import Independent, Normal
from torch.distributions.distribution import Distribution
from torch.distributions.uniform import Uniform
from torch.optim.lr_scheduler import LambdaLR

from marco_polo.tools.types import FloatArray, Role, PathString


#######################################################
## Factories
#######################################################
def get_model(
    env: ParallelEnv, role: Role, args: argparse.Namespace, seed: int
) -> BasePolicy:
    """Returns the class to use, based on input arguments

    Parameters
    ----------
    env: pettingzoo.util.env.ParallelEnv
        A parallel env that holds to pettingzoos's interface
    role: String
        Agent role in env
    args: argparse.Namespace
        arguments that were passed to the `main()` function
    seed: int
        seed for agent creation

    Returns
    -------
    policy
        agent policy model
    """

    actor, critic = construct_actor_critic(env=env, player_name=role, args=args)
    policy = construct_policy(actor=actor, critic=critic, args=args)

    return policy


#######################################################
## Auxiliary Functions
#######################################################
def dist(*logits: Any) -> Distribution:
    return Independent(Normal(logits[0], scale=logits[1] * 0.1), 1)


def construct_actor_critic(
    env: ParallelEnv, player_name: str, args: argparse.Namespace
) -> tuple[ActorProb, Critic]:
    observation_space = (
        env.observation_spaces["observation"]
        if isinstance(env.observation_spaces, gym.spaces.Dict)
        else env.observation_spaces
    )

    action_space = (
        env.action_spaces["action"]
        if isinstance(env.action_spaces, gym.spaces.Dict)
        else env.action_spaces
    )

    # print(observation_space)
    # print(action_space)
    max_action = action_space[player_name].high[0]

    # model
    a_net = Net(
        state_shape=observation_space[player_name].shape or observation_space.n,
        hidden_sizes=[40, 40],
        activation=nn.Tanh,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    actor = ActorProb(
        a_net,
        action_shape=action_space[player_name].shape or action_space[player_name].n,
        max_action=max_action,
        unbounded=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    c_net = Net(
        state_shape=observation_space[player_name].shape
        or observation_space[player_name].n,
        hidden_sizes=[40, 40],
        activation=nn.Tanh,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    critic = Critic(c_net, device="cuda" if torch.cuda.is_available() else "cpu").to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    torch.nn.init.constant_(actor.sigma_param, -0.5)
    for m in list(actor.modules()) + list(critic.modules()):
        if isinstance(m, torch.nn.Linear):
            # orthogonal initialization
            torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            torch.nn.init.zeros_(m.bias)

    # do last policy layer scaling, this will make initial actions have (close to)
    # 0 mean and std, and will help boost performances,
    # see https://arxiv.org/abs/2006.05990, Fig.24 for details
    for m in actor.mu.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.zeros_(m.bias)
            m.weight.data.copy_(0.01 * m.weight.data)

    return (actor, critic)


def construct_policy(
    actor: ActorProb, critic: Critic, args: argparse.Namespace
) -> BasePolicy:
    optim = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), lr=args.tianshou["lr"]
    )

    lr_scheduler = None
    if args.tianshou["lr_decay"]:
        # decay learning rate to 0 linearly
        max_update_num = (
            np.ceil(args.tianshou["step_per_epoch"] / args.tianshou["step_per_collect"])
            * args.tianshou["epoch"]
        )

        _lr = _lr_scheduler(max_update_num)
        lr_scheduler = LambdaLR(optim, lr_lambda=_lr.lr)

    policy = MC_PPOPolicy(
        args,
        actor,
        critic,
        optim,
        dist,
        # discount_factor=args.gamma,
        # gae_lambda=args.gae_lambda,
        # max_grad_norm=args.max_grad_norm,
        # vf_coef=args.vf_coef,
        # ent_coef=args.ent_coef,
        # reward_normalization=args.rew_norm,
        action_scaling=True,
        # action_bound_method=args.bound_action_method,
        lr_scheduler=lr_scheduler,
        # action_space=env.action_space,
        # eps_clip=args.eps_clip,
        # value_clip=args.value_clip,
        # dual_clip=args.dual_clip,
        # advantage_normalization=args.norm_adv,
        # recompute_advantage=args.recompute_adv
    )

    return policy


#######################################################
## Auxiliary Classes
#######################################################
class Identify:
    def __init__(self, weights: FloatArray) -> None:
        self.weights = weights

    def sample(self) -> FloatArray:
        return self.weights


class _lr_scheduler:
    def __init__(self, max_update_num: float) -> None:
        self.max_update_num = max_update_num

    def lr(self, epoch: int) -> float:
        return 1 - epoch / self.max_update_num


#######################################################
## Main Class
#######################################################
## Add saving and loading using the MC checkpoint lanauge.
class MC_PPOPolicy(PPOPolicy):  # type: ignore[misc] # PPOPolicy is Any
    def __init__(
        self, init_args: argparse.Namespace, *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.init_args = init_args

    def checkpoint(self, folder: PathString) -> None:
        path = os.path.join(os.path.dirname(folder), "Model.pth")
        torch.save(self.state_dict(), path)

    def reload(self, folder: PathString) -> None:
        path = os.path.join(os.path.dirname(folder), "Model.pth")
        self.load_state_dict(torch.load(path))

    def get_action(
        self, obs: FloatArray, t: float = 0, mean_mode: bool = False
    ) -> FloatArray:
        # obs.info = dict()
        obs_bat = Batch(obs=obs.reshape(1, -1), info=dict())
        act_bat = self.forward(obs_bat)
        return act_bat.act.numpy().reshape(-1)  # type: ignore[no-any-return] # from base class

    def get_theta(self) -> FloatArray:
        theta = []
        for param in self.actor.parameters():
            theta.append(param.detach().numpy().reshape(-1))

        for param in self.critic.parameters():
            theta.append(param.detach().numpy().reshape(-1))

        return np.concatenate(theta)

    def __deepcopy__(self, memo: dict[int, Any]) -> BasePolicy:
        result = construct_policy(
            actor=copy.deepcopy(self.actor),
            critic=copy.deepcopy(self.critic),
            args=self.init_args,
        )
        memo[id(self)] = result

        return result
