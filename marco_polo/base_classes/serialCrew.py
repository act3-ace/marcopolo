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


"""Serial Pool for Serial Multiprocessing Manager"""

from marco_polo.tools.extraneous import SingleProcessPool


class SerialCrew:  # pylint: disable=too-few-public-methods
    """Creates a 'crew' of one the does everything directly"""

    def get_crew(self) -> SingleProcessPool:
        """Return a Dummy Pool with only a single process

        Returns
        -------
        SingleProcessPool
            'pool' of one worker
        """
        return SingleProcessPool()
