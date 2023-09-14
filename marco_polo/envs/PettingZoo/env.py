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

# The following code is modified from Farama-Foundation/PettingZoo
# (https://github.com/Farama-Foundation/PettingZoo)
# under the MIT License.

"""PettingZoo Connect 4 """

from pettingzoo.classic import connect_four_v3  # type: ignore[import]
import argparse  # for type hinting
from typing import Any, cast  # for type hinting
from collections.abc import Callable  # for type hinting

import logging

logging.getLogger("pettingzoo.utils.env_logger").setLevel(logging.WARNING)
import numpy as np

from marco_polo.envs.PettingZoo.params import PettingZooEnvParams
from marco_polo.tools.types import FloatArray


################################################################################
## Auxiliary Functions
################################################################################
def softmax(x: FloatArray) -> FloatArray:
    exp = np.exp(x)
    return exp / cast(float, np.exp(x).sum())


#######################################################
## Factories
#######################################################
def get_env_class(args: argparse.Namespace) -> Callable[..., Any]:
    """Returns the class to use, based on input arguments

    Parameters
    ----------
    args: argparse.Namespace
        arguments that were passed to the `main()` function

    Returns
    -------
    class
        the class to use in creating env objects
    """
    return PettingZooEnv


################################################################################
## Main Class
################################################################################
class PettingZooEnv:
    def __init__(self) -> None:
        """ """
        self.env = connect_four_v3.env(render_mode="rgb_array")

    def __getattr__(self, name):
        """Returns an attribute with ``name``, unless ``name`` starts with an underscore."""

        return getattr(self.env, name)

    def last(self):
        observation, reward, termination, truncation, info = self.env.last()
        # self.action_mask = observation["action_mask"].reshape(-1)
        # observation = np.concatenate([observation["observation"].reshape(-1),self.action_mask])

        return (observation, reward, termination, truncation, info)

    def step(self, act):
        # act = np.random.sample(act)
        act = np.random.choice(range(len(act)), p=softmax(act))
        # if self.action_mask[act]==0:
        #     return(True)

        self.env.step(act)
        # return(False)

    def augment(self, params):
        pass

    def seed(self, seed):
        pass

    def render(self, *args, **kwargs):
        if kwargs.get("close", False):
            self.env.close()
            return

        return self.env.render()
