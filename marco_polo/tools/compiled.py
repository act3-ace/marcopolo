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


from numba import njit  # type: ignore[import]


@njit
def fast_clip(array: list[float], min: float, max: float) -> list[float]:
    """Clip all values in array to be in [min, max]

    Any vaule less than `min` will be set to `min`
    Any value more than `max` will be set to `max`

    Parameters
    ----------
    array: list[float]
        iterable of data to clip
    min: float
        minimum value
    min: float
        minimum value

    Returns
    -------
    list[float]
        data after clipping
    """
    n = len(array)
    assert max > min, "max must be greater than min"
    for i in range(n):
        if array[i] < min:
            array[i] = min
        elif array[i] > max:
            array[i] = max
    return array


@njit
def fast_sum(array: list[float]) -> float:
    """Return the sum of the given array

    Parameters
    ----------
    array: list[float]
        iterable of data to sum

    Returns
    -------
    float:
        sum of array
    """
    result = 0.0
    n = len(array)
    for i in range(n):
        result += array[i]
    return result
