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


"""types used in MarcoPolo"""
from typing import NewType, Union
from typing_extensions import LiteralString
from os import PathLike

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float_]
ActivationType = FloatArray

IntArray = NDArray[np.int_]

RenderFrame = NDArray[np.uint8]

# using NewType can help find logical errors within MarcoPolo
# however, the type will conflict with typing in pettingzoo
# If that's a problem, comment out that definition and uncomment
# the one after it
Role = NewType("Role", str)
# Role = str

AgentId = NewType("AgentId", str)
EnvId = NewType("EnvId", str)
TeamId = NewType("TeamId", str)

# PathString = NewType(
#     "PathString", Union[str, PathLike[str], bytes, PathLike[bytes], LiteralString]
# )
# PathString = Union[str, PathLike[str], bytes, PathLike[bytes], LiteralString]
PathString = str
