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


from typing import Iterator

import numpy as np
from numpy.typing import NDArray

from marco_polo.tools.types import FloatArray, IntArray


def _comp_ranks(x: FloatArray) -> IntArray:
    """
    Returns ranks in [0, len(x))

    Note: This is different from scipy.stats.rankdata, which returns ranks in [1, len(x)].
    """

    # init empty vector for return
    ranks = np.empty(len(x), dtype=int)

    # Compute ranks of x
    #  argsort() returns indices of ranks
    #  arange() returns a range 0:x
    #  This is the fastest way to do this.
    ranks[x.argsort()] = np.arange(len(x))

    return ranks


def compute_CS_ranks(x: FloatArray) -> FloatArray:
    """Compute Centered and Scaled Ranks

    This function takes an object (presumbably the scores), centers the scores
    on 0, and normalizes between -0.5 and 0.5, inclusive.

    Parameters
    ----------
    x: 2-D Numpy array
        This is a [`optim_jobs`*`rollouts_per_optim_job`, 2] array of evaluation scores

    Returns
    -------
    y: Numpy array
        Array of ranks linearly spaces on the closed interval [-0.5, 0.5]

    Reference
    ---------
    https://stats.stackexchange.com/questions/164833/how-do-i-normalize-a-vector-of-numbers-so-they-are-between-0-and-1

    """

    # Safety check
    if x.size == 1:
        # print("stats.computer_centered_ranks:: x is of size 1")
        return np.zeros(shape=1, dtype=np.float32)

    # Compute ranks
    #  object is the same dimensions as x
    #  has values [0, len(x))
    y = _comp_ranks(x.ravel()).reshape(x.shape).astype(np.float32)

    # Normalize to (0, 1)
    #  This works because the values are ranks, 0 to len(x)
    #  so max(x) - min(x) = size - 1
    y /= x.size - 1

    # Center vector at 0
    #  The vector is already normed (0,1), so shift it by -0.5,
    #  it's now centered at 0, with range (-0.5, 0.5)
    y -= 0.5

    return y


def compute_weighted_sum(
    weights: FloatArray, vec_generator: Iterator[FloatArray], theta_size: int
) -> tuple[NDArray[np.float32], int]:
    """Calculates a weighted sum

    This function calculates a weighted sum using the weights input list as the value for
    each vector from vec_generator. There is no safety if vec_generator is longer
    than weights.

    Parameters
    ----------
    weights: FloatArray
        Array of numeric weights
    vec_generator: Iterator[FloatArray]
        Iterator of numpy arrays, where the generator is the same length as the weights
    theta_size: int
        Size of arrays created by vec_generator

    Returns
    -------
    total: NDArray[np.float32]
        Weighted sum of vectors, length theta_size
    num_items_summed: int
        Count of the items combined. This should be the same length as weights, but
        is defined as the number of items from vec_generator
    """

    # setup return objects
    #  use the summation as a counter
    total = np.zeros(theta_size, dtype=np.float32)  # TODO: why float32 specifically??
    num_items = 0

    # loop through noise generator
    for vec in vec_generator:
        # grab weight and multiply by noise
        total += weights[num_items] * vec
        # increment counter
        num_items += 1

    # return total vector and number of estimates combined
    return total, num_items
