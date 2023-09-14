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


import json
import os
from collections.abc import Sized
from typing import Any

import numpy as np

from marco_polo.tools.iotools import NumpyDecoder, NumpyEncoder
from marco_polo.tools.types import FloatArray, Role, PathString  # for type hinting


################################################################################
## Parent Optimizer
################################################################################
class Optimizer(object):
    def __init__(self, theta: Sized) -> None:
        self.dim = len(theta)
        self.t = 0

    def update(
        self, theta: FloatArray, globalg: FloatArray
    ) -> tuple[np.float_, FloatArray]:
        self.t += 1
        step = self._compute_step(globalg)
        ratio = np.linalg.norm(step) / np.linalg.norm(theta)
        return ratio, theta + step

    def _compute_step(self, globalg: FloatArray) -> FloatArray:
        raise NotImplementedError

    def checkpoint(self, folder: PathString) -> None:
        raise NotImplementedError

    def reload(self, folder: PathString) -> None:
        raise NotImplementedError


################################################################################
## Simple Stochastic Gradient Descent
################################################################################
class SimpleSGD(Optimizer):
    def __init__(self, stepsize: float) -> None:
        self.stepsize = stepsize

    def compute(
        self, theta: FloatArray, globalg: FloatArray
    ) -> tuple[np.float_, FloatArray]:
        step = -self.stepsize * globalg
        ratio = np.linalg.norm(step) / np.linalg.norm(theta)
        return ratio, theta + step

    # why is there no step method?
    # what is this compute method, if not mostly the step method?


################################################################################
## Full Stochastic Gradient Descent
################################################################################
class SGD(Optimizer):
    def __init__(
        self, theta: FloatArray, stepsize: float, momentum: float = 0.9
    ) -> None:
        # parent constructor
        Optimizer.__init__(self, theta)

        # set vars
        self.v = np.zeros(self.dim, dtype=np.float32)
        self.stepsize, self.momentum = stepsize, momentum

    def _compute_step(self, globalg: FloatArray) -> FloatArray:
        self.v = self.momentum * self.v + (1.0 - self.momentum) * globalg
        step = -self.stepsize * self.v
        return step

    def checkpoint(self, folder: PathString) -> None:
        file_path = os.path.join(folder, "_SGD.json")
        with open(file_path, mode="w", encoding="utf-8") as f:
            json.dump(self.__dict__, f, cls=NumpyEncoder)

    def reload(self, folder: PathString) -> None:
        file_path = os.path.join(folder, "_SGD.json")
        with open(file_path, mode="r", encoding="utf-8") as f:
            self.__dict__ = json.load(f, cls=NumpyDecoder)


################################################################################
## Adam
################################################################################
class Adam(Optimizer):
    def __init__(
        self,
        theta: FloatArray,
        stepsize: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-08,
    ) -> None:
        # parent constructor
        super().__init__(theta)

        # vars
        self.stepsize = stepsize
        self.init_stepsize = stepsize
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = np.zeros(self.dim, dtype=np.float32)
        self.v = np.zeros(self.dim, dtype=np.float32)

    def checkpoint(self, folder: PathString, role: Role = Role("")) -> None:
        file_path = os.path.join(folder, f"{role}_Adam.json")
        with open(file_path, mode="w", encoding="utf-8") as file:
            json.dump(self.__dict__, file, cls=NumpyEncoder)

    def reload(self, folder: PathString, role: Role = Role("")) -> None:
        file_path = os.path.join(folder, f"{role}_Adam.json")
        with open(file_path, mode="r", encoding="utf-8") as file:
            self.__dict__ = json.load(file, cls=NumpyDecoder)

    def reset(self) -> None:
        self.m = np.zeros(self.dim, dtype=np.float32)
        self.v = np.zeros(self.dim, dtype=np.float32)
        self.t = 0
        self.stepsize = self.init_stepsize

    def _compute_step(self, globalg: FloatArray) -> FloatArray:
        a: float = (
            self.stepsize
            * np.sqrt(1 - self.beta2**self.t)
            / (1 - self.beta1**self.t)
        )
        self.m = self.beta1 * self.m + (1 - self.beta1) * globalg
        self.v = self.beta2 * self.v + (1 - self.beta2) * (globalg * globalg)
        step = -a * self.m / (np.sqrt(self.v) + self.epsilon)
        return step

    # Artifact from POET, not deleting but probably not using.
    # def propose(
    #     self, theta: FloatArray, globalg: FloatArray
    # ) -> tuple[np.float_, FloatArray]:
    #     a = self.stepsize
    #     # Unlike _compute_step(), propose may not have t updated prior to
    #     # being called. If t==0, the scaling factor is effectively set to 1.
    #     if self.t > 0:
    #         a *= np.sqrt(1 - self.beta2**self.t) / (1 - self.beta1**self.t)

    #     m = self.beta1 * self.m + (1 - self.beta1) * globalg
    #     v = self.beta2 * self.v + (1 - self.beta2) * (globalg * globalg)
    #     step = -a * m / (np.sqrt(v) + self.epsilon)
    #     ratio: np.floating[Any] = np.linalg.norm(step) / np.linalg.norm(theta)
    #     return ratio, theta + step
