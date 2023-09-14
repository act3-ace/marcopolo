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


"""ARS Observation Filter"""

# Code in this file is copied and adapted from
# https://github.com/modestyachts/ARS/blob/master/code/filter.py
# https://github.com/iamsuvhro/Augmented-Random-Search/blob/master/ars.py
# https://github.com/sourcecode369/Augmented-Random-Search-/blob/master/Augmented%20Random%20Search/ars.py

# So, none of these implementations are entirely correct.
# Modestyachts implements the derivation from https://www.johndcook.com/blog/skewness_kurtosis/
#  These look to be Chan's formulae for 3rd/4th central moments
# The other two implement the cited one from modestyachts, but mess up the variance
# Without going back to the original derivation, I'm just going to implement the
#  cited algorithm properly.


import numpy as np
import os
import json

from marco_polo.tools.iotools import NumpyDecoder, NumpyEncoder
from marco_polo.tools.types import FloatArray, PathString


################################################################################
## Filter Class
################################################################################
class Filter:
    """Observation Filter

    This class performs centering and normalization on input observations.
    This comes from the Modestyachts ARS implementation, though the algorithms
    have been rederived for correctness.
    """

    def __init__(self, num_inputs: int) -> None:
        # obs counter
        self._n = 0
        # current mean
        self._M = np.zeros(num_inputs, dtype=np.float64)
        # previous mean
        self._M_old = np.zeros(num_inputs, dtype=np.float64)
        # dunno what "S" is, possibly "sigma" 'cause it's the standard deviation
        self._S = np.zeros(num_inputs, dtype=np.float64)

        # things for normalizing obs
        self.mean = np.zeros(num_inputs, dtype=np.float64)
        self.std = np.zeros(num_inputs, dtype=np.float64)

        # update stats
        self.update_stats()

    def reset(self) -> None:
        self._n = 0
        # use subsetting to set all values
        self._M[...] = 0.0
        self._M_old[...] = 0.0
        self._S[...] = 0.0

        # update stats
        self.update_stats()

    def copy(self) -> "Filter":
        # setup new filter
        retFilt = Filter(num_inputs=self._M.size)

        # copy internals
        retFilt._n = self._n
        retFilt._M = self._M.copy()
        retFilt._M_old = self._M_old.copy()
        retFilt._S = self._S.copy()

        # update stats
        retFilt.update_stats()

        # return
        return retFilt

    def checkpoint(self, filename: PathString, newname: str) -> None:
        # split filename
        folder = os.path.dirname(filename)
        # replace file name and save
        with open(os.path.join(folder, newname), mode="w", encoding="utf-8") as f:
            json.dump(self.__dict__, f, cls=NumpyEncoder)

    def reload(self, filename: PathString, newname: str) -> None:
        # split filename
        folder = os.path.dirname(filename)
        # replace file name and save
        with open(os.path.join(folder, newname), mode="r", encoding="utf-8") as f:
            self.__dict__ = json.load(f, cls=NumpyDecoder)

    def observe(self, x: FloatArray) -> None:
        """
        Welford's basic online algorithm

        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
        """
        # increment number of samples
        self._n += 1

        # update
        # mean
        #  at this point, "M" and "M_old" are the same, so we can use "+=" operator
        self._M += (x - self._M_old) / self._n
        # "S"
        #  this requires "M" and "M_old", which is why we have 2
        self._S += (x - self._M_old) * (x - self._M)
        # old mean
        #  at the next iteration, the current will be "old"
        self._M_old = self._M.copy()

    def update_stats(self) -> None:
        # std deviation
        #  modestyachts use _M as the default std
        #   maybe to avoid zeros?
        self.std = np.sqrt(self._S / (self._n - 1)) if (self._n > 1) else self._M.copy()

        # modestyachts safety check
        # Set values for std less than 1e-7 to +inf to avoid
        # dividing by zero. State elements with zero variance
        # are set to zero as a result.
        self.std[self.std < 1e-7] = float("inf")

        # mean
        self.mean = self._M

    def normalize(self, x: FloatArray) -> FloatArray:
        return (x - self.mean) / self.std

    def sync(self, other: "Filter") -> None:
        """
        Syncs fields from other filter

        Updates the internal state to match the supplied, "other", filter
        """
        self._n = other._n
        self._M = other._M.copy()
        self._M_old = other._M_old.copy()
        self._S = other._S.copy()

        # update stats
        self.update_stats()

    def update(self, other: "Filter") -> None:
        """
        Chan's general solution of Welford's Algorithm

        This is actually a blend, 'cause Chan's mean calculation is unstable.
        I can't find anything about stability issues with S.
        Also can't tell if it's a population variance or a sample variance.
        (I think it's population and it's biased, but not positive).

        http://i.stanford.edu/pub/cstr/reports/cs/tr/79/773/CS-TR-79-773.pdf
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        """
        # grab number of samples
        n_self = self._n
        n_other = other._n

        # difference of means
        delta = self._M - other._M

        # update total observation count
        self._n = n_self + n_other

        # update mean
        self._M = ((n_self * self._M) + (n_other * other._M)) / self._n

        # update S
        #  I have numerical stability questions about this
        self._S = self._S + other._S + delta * delta * n_self * n_other / self._n

    def __repr__(self) -> str:
        return f"(n={self._n}, mean_M={self._M.mean()}, mean_S={self._S.mean()})"
