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


"""PettingZoo Parameter Class """

from typing import Any  # for type hinting
from collections.abc import Callable  # for type hinting
from argparse import Namespace  # for type hinting
import logging

from pettingzoo.classic import connect_four_v3  # type: ignore[import]

from marco_polo.envs._base.env_params import EnvParams

logging.getLogger("pettingzoo.utils.env_logger").setLevel(logging.WARNING)


# logger = logging.getLogger(__name__)


################################################################################
## Auxiliary Functions
################################################################################
def get_env_param_class(args: Namespace) -> Callable[..., Any]:
    """Returns the class to use, based on input arguments

    Parameters
    ----------
    args: argparse.Namespace
        arguments that were passed to the `main()` function

    Returns
    -------
    class
        the class to use in creating env parameter objects
    """
    return PettingZooEnvParams


################################################################################
## Main Class
################################################################################
class PettingZooEnvParams(EnvParams):
    """Parameters for PettingZoo Environments"""

    def __init__(self) -> None:
        tmp = connect_four_v3.env()
        tmp.reset()
        self.agents = tmp.agents

    def __getattr__(self, name: str) -> Any:
        """Returns an attribute with ``name``, unless ``name`` starts with an underscore."""
        if name.startswith("_"):
            raise AttributeError(f"accessing private attribute '{name}' is prohibited")

        if name in self.__dict__.keys():
            return self.__dict__[name]

        return getattr(self._params, name)
