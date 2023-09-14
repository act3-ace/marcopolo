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


"""Uber ES Basic Agent Model"""

import argparse  # for type hinting
import json
import logging
from typing import Any, cast, Optional, Union  # for type hinting

import numpy as np
from numpy.random import PCG64DXSM, Generator

from pettingzoo.utils.env import ParallelEnv  # type: ignore [import]

from marco_polo.base_classes.model import BaseModel
from marco_polo.tools.iotools import NumpyDecoder, NumpyEncoder
from marco_polo.tools.types import (
    ActivationType,
    FloatArray,
    Role,
    PathString,
)  # for type hinting

logger = logging.getLogger(__name__)


################################################################################
## Agent Factory
################################################################################
def get_model(
    env: ParallelEnv, role: Role, args: argparse.Namespace, seed: int
) -> BaseModel:
    """Returns the class to use, based on input arguments

    Parameters
    ----------
    env: pettingzoo.util.env.ParallelEnv
        A parallel env that holds to pettingzoos's interface, not used here
    role: Role
        Agent role in env, not used here
    args: argparse.Namespace
        arguments that were passed to the `main()` function
    seed: int
        seed for model creation

    Returns
    -------
    BaseModel
        The model
    """
    return Model(model_params=args.model_params, seed=seed)


################################################################################
## Aux Funcs
################################################################################
# possible activation functions for each layer of the NN
def sigmoid(x: ActivationType) -> ActivationType:
    return 1 / (1 + np.exp(-x))


def relu(x: ActivationType) -> ActivationType:
    return np.maximum(x, 0)


def passthru(x: ActivationType) -> ActivationType:
    return x


def softmax(x: ActivationType) -> ActivationType:
    arg: ActivationType = x - np.max(x)
    e_x = np.exp(arg)
    return e_x / cast(float, e_x.sum(axis=0))  # cast is to help mypy


################################################################################
## Model Class
################################################################################
class Model(BaseModel):
    """simple feedforward model"""

    def __init__(
        self,
        model_params: dict[str, Any],
        seed: int,
        args: Optional[argparse.Namespace] = None,
    ) -> None:
        """Constructor for Uber ES Model

        Much of this construction came from Uber's original POET implementation.
        Primary additions include documentation and model checkpointing.

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
        self.args = args
        self.np_random = Generator(PCG64DXSM(seed=seed))

        self.output_noise = model_params["output_noise"]

        self.rnn_mode = False  # in the future will be useful
        self.time_input = 0  # use extra sinusoid input
        self.sigma_bias = model_params["noise_bias"]  # bias in stdev of output
        self.sigma_factor = 0.5  # multiplicative in stdev of output
        if model_params["time_factor"] > 0:
            self.time_factor = float(model_params["time_factor"])
            self.time_input = 1
        self.input_size = model_params["input_size"]
        self.output_size = model_params["output_size"]

        self.shapes = [(self.input_size + self.time_input, model_params["layers"][0])]
        self.shapes += [
            (model_params["layers"][i], model_params["layers"][i + 1])
            for i in range(len(model_params["layers"]) - 1)
        ]
        self.shapes += [(model_params["layers"][-1], self.output_size)]

        self.sample_output = model_params["sample_output"]

        if len(model_params["activations"]) != (len(model_params["layers"]) + 1):
            raise ValueError(
                "The model activations need to be 1 function greater \
                than the number of layers"
            )

        self.activations = [eval(x) for x in model_params["activations"]]

        self.actor_weight = []
        self.actor_bias = []
        self.actor_bias_log_std = []
        self.actor_bias_std = []
        self.actor_param_count = 0

        idx = 0
        for shape in self.shapes:
            self.actor_weight.append(np.zeros(shape=shape))
            self.actor_bias.append(np.zeros(shape=shape[1]))
            self.actor_param_count += np.product(shape) + shape[1]
            if self.output_noise[idx]:
                self.actor_param_count += shape[1]
            log_std = np.zeros(shape=shape[1])
            self.actor_bias_log_std.append(log_std)
            out_std = np.exp(self.sigma_factor * log_std + self.sigma_bias)
            self.actor_bias_std.append(out_std)
            idx += 1

        # initialize action scale
        self.scale = np.ones(self.output_size)
        # declare if properly specified
        if model_params.get("action_scale", False):
            self.scale = np.array(model_params["action_scale"])
            # self.scale = np.asarray(a=model_params["action_scale"], dtype=float)

        self.render_mode = False
        self.theta = self.get_random_model_params()

    # aux getters/setters
    #  These are incredibly important due to the use of a wrapper class
    def __repr__(self) -> str:
        return "{}".format(self.__dict__)

    def __getstate__(self) -> dict[str, Any]:
        return self.__dict__

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__ = state

    def get_action(
        self, x: ActivationType, t: float = 0, mean_mode: bool = False
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
            Uses internal PRNG

        Returns
        -------
        Union[np.int_, FloatArray]
            Agent action
        """
        # if mean_mode = True, ignore sampling.
        h = np.array(x).flatten()
        if self.time_input == 1:
            time_signal = float(t) / self.time_factor
            h = np.concatenate([h, [time_signal]])
        num_layers = len(self.actor_weight)
        for i in range(num_layers):
            w = self.actor_weight[i]
            b = self.actor_bias[i]
            h = np.matmul(h, w) + b
            if self.output_noise[i] and (not mean_mode):
                out_size = self.shapes[i][1]
                out_std = self.actor_bias_std[i]
                output_noise = self.np_random.standard_normal(size=out_size) * out_std
                h += output_noise
            h = self.activations[i](h)

        if self.sample_output:
            return np.argmax(self.np_random.multinomial(n=1, pvals=h, size=1))

        return h * self.scale

    def set_model_params(self, model_params: FloatArray) -> None:
        pointer = 0
        for i in range(len(self.shapes)):  # pylint: disable=consider-using-enumerate
            w_shape = self.shapes[i]
            b_shape = self.shapes[i][1]
            s_w = np.product(w_shape)
            s = s_w + b_shape
            chunk = np.array(model_params[pointer : pointer + s])
            self.actor_weight[i] = chunk[:s_w].reshape(w_shape)
            self.actor_bias[i] = chunk[s_w:].reshape(b_shape)
            pointer += s
            if self.output_noise[i]:
                s = b_shape
                self.actor_bias_log_std[i] = np.array(
                    model_params[pointer : pointer + s]
                )
                self.actor_bias_std[i] = np.exp(
                    self.sigma_factor * self.actor_bias_log_std[i] + self.sigma_bias
                )
                if self.render_mode:
                    logger.info(f"bias_std: {self.actor_bias_std[i]}, layer: {i}")
                pointer += s

    def reload(self, filename: PathString) -> None:
        with open(filename, mode="r", encoding="utf-8") as f:
            data = json.load(f, cls=NumpyDecoder)
        self.theta = data["theta"]
        self.np_random.bit_generator.state = data["np_random"]
        self.set_model_params(model_params=self.theta)

    def checkpoint(self, filename: PathString) -> None:
        manifest = {
            "theta": self.theta,
            "np_random": self.np_random.bit_generator.state,
        }
        with open(filename, mode="w", encoding="utf-8") as f:
            json.dump(manifest, f, cls=NumpyEncoder)

    def shift_weights(self, noise_std: float, seed: int) -> None:
        # set random state
        random_state = Generator(PCG64DXSM(seed=seed))
        # draw uniform, shift and scale
        # BUG: This is supposed to be a normal distribution
        #      However, it performs better this way, so we leave it
        #      Should read: noise_std * random_state.standard_normal(size=self.actor_param_count)
        t = 2 * noise_std * (random_state.random(self.actor_param_count) - 0.5)
        # update local theta
        theta = self.theta + t
        # set model parameters to this theeta
        #  NOTE: it no longer matches the model theta!
        self.set_model_params(theta)

    def update_theta(self, new_theta: FloatArray) -> None:
        self.theta = new_theta

    def get_theta(self) -> FloatArray:
        return self.theta

    def get_random_model_params(self, stdev: float = 0.1) -> FloatArray:
        return self.np_random.normal(scale=stdev, size=self.actor_param_count)

    def get_zeroed_model_params(self) -> FloatArray:
        return np.zeros(self.actor_param_count)
