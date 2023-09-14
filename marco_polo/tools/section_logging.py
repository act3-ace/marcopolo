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


"""Class to log sections of runs

Using this class cleans up code elsewhere

"""
import logging  # for type hinting
import time
from typing import Any


class SectionLogger:
    """Class tracks time and prints banners

    Example usage:
    section = SectionLogger(logger, sectin_name="NAME", prefix="step 1")
    ...
    section.print_status_banner("update msg")
    section.print_raw("some message")
    section.print_time()
    section.print_end_banner()

    This will print (via the logger's info stream):
    step 1 - ############### NAME Start ###############
    step 1 - ############### NAME update msg ###############
    step 1 - some message
    step 1 - NAME Step Time: 12.4 seconds
    step 1 - ############### NAME Complete ###############

    Alternate usage with context manager (gives same output):
    with SectionLogger(logger, sectin_name="NAME", prefix="step 1") as section:
        ...
        section.print_status_banner("update msg")
        section.print_raw("some message")
        section.print_time()

    """

    def __init__(
        self,
        logger: logging.Logger,
        section_name: str,
        prefix: str = "",
        print_start: bool = True,
    ) -> None:
        """Create the object and print the start banner

        Parameters
        ----------
        logger: logging.Logger
            the logging object to use for output
        section_name: str
            name of the section to print in banners
        prefix: str, default = ""
            prefix to prepend to each output line
            default is no prefix
        print_start: bool, default = True
            Whether to print the start banner when object is created
            default is True
        """
        self.start_time = time.time()
        self.logger = logger
        self.section_name = section_name
        self.prefix = prefix
        # if there is a prefix, add a spacer after it
        if self.prefix:
            self.prefix += " - "
        if print_start:
            self.print_start_banner()

    def print_start_banner(self) -> None:
        """Print the start banner."""
        self.logger.info(
            f"{self.prefix}############### {self.section_name} Start ###############"
        )

    def print_time(self) -> None:
        """Print the current time elapsed"""
        delta_time = time.time() - self.start_time
        self.logger.info(
            f"{self.prefix}{self.section_name} Step Time: {delta_time:.2f} seconds"
        )

    def print_end_banner(self) -> None:
        """Print the end banner."""
        self.logger.info(
            f"{self.prefix}############### {self.section_name} Complete ###############\n"
        )

    def print_status_banner(self, status: str) -> None:
        """Print a status banner."""
        self.logger.info(
            f"{self.prefix}############### {self.section_name} {status} ###############"
        )

    def print_raw(self, line: str) -> None:
        """Print a line with the prefix."""
        self.logger.info(f"{self.prefix}{line}")

    def __enter__(self) -> "SectionLogger":
        """Enter context, return object."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Exit context, print end banner."""
        self.print_end_banner()
