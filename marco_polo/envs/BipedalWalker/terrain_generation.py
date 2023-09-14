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


"""Functions to generate terrain for the Bipedal Walker"""
import json
from typing import Any, Optional

import numpy as np


from marco_polo.envs.BipedalWalker.bpw_constants import (
    TERRAIN_GRASS,
    TERRAIN_HEIGHT,
    TERRAIN_LENGTH,
    TERRAIN_STARTPAD,
    TERRAIN_STEP,
)
from marco_polo.envs.BipedalWalker.cppn import CppnEnvParams
from marco_polo.tools.types import FloatArray


################################################################################
## Main Functions
################################################################################
def generate_terrain_coords(
    env_params: Optional[CppnEnvParams], random: np.random.Generator
) -> tuple[FloatArray, FloatArray]:
    """Generate and return the coordinates of the terrain

    The method used for terrain depends on the values in env_params

    Parameters
    ----------
    env_params: Optional[CppnEnvParams]
        the environment parameters
    random: np.random.Generator
        The random number object to use for generating terrain

    Returns
    -------
    FloatArray, FloatArray
        arrays of x and y values of terrain
    """
    if env_params is None:
        return _flat_terrain()

    # env_params.terrain_func is optional. If missing, default to original
    terrain_func = env_params.get("terrain_function", "original")

    if terrain_func == "original":
        return _default_simple_terrain(env_params, random)
    elif terrain_func.startswith("file:"):
        return _terrain_from_file(terrain_func)

    raise ValueError("Unknown value for env_params.terrain_func")


def _flat_terrain() -> tuple[FloatArray, FloatArray]:
    """Flat line at y=0, used when there are no env_params."""
    terrain_x = np.arange(TERRAIN_LENGTH) * TERRAIN_STEP
    terrain_y = np.zeros(TERRAIN_LENGTH)
    return terrain_x, terrain_y


def _terrain_from_file(terrain_func: str) -> tuple[FloatArray, FloatArray]:
    """Read terrain from json file"""
    filename = terrain_func.replace("file:", "", 1)
    with open(filename, mode="r", encoding="utf8") as file:
        data = json.load(file)

    terrain_x = np.array([94 * (float(i) + 1) for i in data["x"]])
    terrain_y = np.array(float(i[0]) + TERRAIN_HEIGHT for i in data["y"])

    return terrain_x, terrain_y


def _default_simple_terrain(
    env_params: CppnEnvParams, random: np.random.Generator
) -> tuple[FloatArray, FloatArray]:
    """Original terrain generation, used when no generation function is given

    Parameters
    ----------
    env_params: Optional[CppnEnvParams]
        the environment parameters
    random: np.random.Generator
        The random number object to use for generating terrain

    Returns
    -------
    FloatArray, FloatArray
        arrays of x and y values of terrain
    """
    # create proper sized arrays (x array is the same as the flat terrain case)
    terrain_x, terrain_y = _flat_terrain()

    # function_x is only used for the calculation of terrain_y
    function_x = np.linspace(-np.pi, np.pi, 200, endpoint=False)

    y_val = TERRAIN_HEIGHT
    # next_change_step is to enable perioidic modifications of the terrain.
    # When i == change_step, a change is applied to the terrain.
    # For this model, the change is only to skip applying the altitude
    # function for a step. This creates minor roughness in the terrain.
    # In the orginal Gym environment, it also could change the type of
    # the terrain.
    next_change_step = TERRAIN_STARTPAD
    for i_step, x_val in enumerate(function_x):
        if i_step == next_change_step:
            # this is a "change step". The "change" here is to skip
            # updating the y value for this step. So, doing nothing.
            # Now, update the index to the next change step.
            next_change_step += random.integers(TERRAIN_GRASS // 2, TERRAIN_GRASS)
        else:
            # this is not a 'change step', so do the standard altitude
            # calculation if we are past the starting padding
            if i_step > TERRAIN_STARTPAD:
                # mypy doesn't like the altitude_fn function pointer
                y_val = TERRAIN_HEIGHT + env_params.altitude_fn((x_val,))[0]  # type: ignore[no-untyped-call]
                # scale the values vertically so they start at the baseline
                # after the padding steps.
                if i_step == TERRAIN_STARTPAD + 1:
                    # mypy doesn't like the altitude_fn function pointer
                    y_norm = env_params.altitude_fn((x_val,))[0]  # type: ignore[no-untyped-call]
                y_val -= y_norm
        terrain_y[i_step] = y_val

    return terrain_x, terrain_y
