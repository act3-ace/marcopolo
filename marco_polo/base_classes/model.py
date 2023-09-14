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


"""Base model interface"""

from typing import Any, Union

import numpy as np

from marco_polo.tools.types import ActivationType, FloatArray


class BaseModel:
    """Base interface for a model"""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Calling BaseModel is not implemented")

    def checkpoint(self, folder: str) -> None:
        """Save model to disk"""
        raise NotImplementedError("checkpoint not implemented for BaseModel")

    def reload(self, folder: str) -> None:
        """Load model from disk"""
        raise NotImplementedError("reload not implemented for BaseModel")

    def get_action(
        self, x: ActivationType, t: float = 0, mean_mode: bool = False
    ) -> Union[np.int_, FloatArray]:
        raise NotImplementedError("get_action not implemented for BaseModel")
