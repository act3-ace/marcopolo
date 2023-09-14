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


"""ARS Basic Agent Model"""

import json

# import logging
import argparse  # for type hinting

import numpy as np
from numpy.random import PCG64DXSM, Generator
from os import path as osp
from typing import Any, Optional, Union

from marco_polo.base_classes.model import BaseModel
from marco_polo.optimizers.uber_es.modules.model import Model as Uber_Model

from marco_polo.tools.iotools import NumpyDecoder, NumpyEncoder
from marco_polo.optimizers.ARS.modules.filter import Filter
from pettingzoo.utils.env import ParallelEnv  # type: ignore[import]
from marco_polo.tools.types import FloatArray, Role, PathString  # for type hinting

# logger = logging.getLogger(__name__)


################################################################################
## Agent Factory
################################################################################
def get_model(
    env: ParallelEnv, role: Role, args: argparse.Namespace, seed: int
) -> BaseModel:
    """Returns the class to use, based on input arguments

    Parameters
    ----------
    env : pettingzoo.util.env.ParallelEnv
        A parallel env that holds to pettingzoos's interface, not used here
    role : Role
        Agent role in env, not used here
    args : argparse.Namespace
        arguments that were passed to the `main()` function
    seed : int
        seed for agent creation

    Returns
    -------
    BaseModel
        agent policy model
    """
    return Model(model_params=args.model_params, seed=seed)


################################################################################
## Aux Funcs
################################################################################


################################################################################
## Model Class
################################################################################
class Model(Uber_Model):
    """simple feedforward model"""

    def __init__(
        self,
        model_params: dict[str, Any],
        seed: int,
        args: Optional[argparse.Namespace] = None,
    ) -> None:
        """Constructor for ARS Model

        Child class of the Uber ES model.

        Parameters
        ----------
        model_params : dict[str, Any]
            Dictionary defining model structure
        seed : int
            Initial value for internal PRNG
        args : Optional[argparse.Namespace], default=None
            Values to parameters the rest of the model

        Side-Effects
        ------------
        Yes
            Sets all class attributes

        Returns
        -------
        None
        """
        # super constructor
        super().__init__(model_params=model_params, seed=seed, args=args)

        # setup observation filter and storage buffer
        self.save_obs = False
        self.normalize_obs = model_params.get("normalize_obs", True)
        self.obs_buf = Filter(num_inputs=self.input_size)
        self.obs_filt = Filter(num_inputs=self.input_size)

    # aux getters/setters
    #  These are incredibly important due to the use of a wrapper class
    def set_obs_flag(self, flag: bool) -> None:
        self.save_obs = flag

    def get_obs_flag(self) -> bool:
        return self.save_obs

    def get_obs_buf(self) -> Filter:
        return self.obs_buf

    def get_obs_filt(self) -> Filter:
        return self.obs_filt

    def get_action(
        self, x: FloatArray, t: float = 0, mean_mode: bool = False
    ) -> Union[np.int_, FloatArray]:
        """Generate Model Action

        Parameters
        ----------
        x : FloatArray
            Input observations
        t : float, default=0
            Time scale
        mean_mode : bool, default=False
            Generate noise in the network and maintain the average?

        Side-Effects
        ------------
        Yes
            Updates observation buffer
            Uses internal PRNG

        Returns
        -------
        Union[np.int_, FloatArray]
            Agent action
        """
        # flatten observations
        #  this will get done twice, the second time in super().get_action()
        h = np.array(x).flatten()

        # when doing optimization, save the observations for update later
        if self.save_obs:
            self.obs_buf.observe(x=h)

        # normalize observations?
        if self.normalize_obs:
            h = self.obs_filt.normalize(x=h)

        return super().get_action(x=h, t=t, mean_mode=mean_mode)

    def reload(self, filename: PathString) -> None:
        # read state
        with open(filename, mode="r", encoding="utf-8") as f:
            data = json.load(f, cls=NumpyDecoder)

        # load state into object
        self.theta = data["theta"]
        self.np_random.bit_generator.state = data["np_random"]
        self.set_model_params(model_params=self.theta)

        self.save_obs = data.get("save_obs", False)
        self.normalize_obs = data.get("normalize_obs", False)

        # load observation filters if exist
        if osp.isfile(osp.join(osp.dirname(filename), "obs_buf.json")):
            self.obs_buf.reload(filename=filename, newname="obs_buf.json")

        if osp.isfile(osp.join(osp.dirname(filename), "obs_filt.json")):
            self.obs_filt.reload(filename=filename, newname="obs_filt.json")

    def checkpoint(self, filename: PathString) -> None:
        # build dict to save
        manifest = {
            "theta": self.theta,
            "np_random": self.np_random.bit_generator.state,
            "save_obs": self.save_obs,
            "normalize_obs": self.normalize_obs,
        }

        # save state
        with open(filename, mode="w", encoding="utf-8") as f:
            json.dump(manifest, f, cls=NumpyEncoder)

        # save observation filters
        self.obs_buf.checkpoint(filename=filename, newname="obs_buf.json")
        self.obs_filt.checkpoint(filename=filename, newname="obs_filt.json")

    def shift_weights(self, noise_std: float, seed: int) -> None:
        # set random state
        random_state = Generator(PCG64DXSM(seed=seed))
        # draw normal
        t = noise_std * random_state.standard_normal(size=self.actor_param_count)
        # update local theta
        theta = self.theta + t
        # set model parameters to this theeta
        #  NOTE: it no longer matches the model theta!
        self.set_model_params(theta)
