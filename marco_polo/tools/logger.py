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


import csv
import logging
from pprint import pformat

from marco_polo.tools.types import PathString  # for type hinting

logger = logging.getLogger(__name__)


class CSVLogger:
    """
    CSVLogger Class Docs

    Attributes
    ----------

    Methods
    -------

    """

    def __init__(self, fnm: PathString, col_names: list[str]) -> None:
        """ """
        logger.info("Creating data logger at {}".format(fnm))
        self.fnm = fnm
        self.col_names = col_names

        with open(fnm, mode="a", newline="", encoding="utf8") as f:
            writer = csv.writer(f, delimiter=",")
            writer.writerow(col_names)

        # hold over previous values if empty
        self.vals = {name: None for name in col_names}

    def log(self, **cols: str) -> None:
        """ """
        self.vals.update(cols)  # type: ignore  # complicated to fix
        logger.info(pformat(self.vals))

        if any(key not in self.col_names for key in self.vals):
            raise Exception("CSVLogger given invalid key")

        with open(self.fnm, mode="a", newline="", encoding="utf8") as f:
            writer = csv.writer(f, delimiter=",")
            writer.writerow([self.vals[name] for name in self.col_names])
