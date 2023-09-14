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


"""Constants used by both the BipedalWalker and its renderers"""


SCALE = 30.0  # affects how fast-paced the game is, forces should be adjusted as well

# Output viewport (width and height) in pixels
VIEWPORT_W = 600
VIEWPORT_H = 400

TERRAIN_STEP = 14 / SCALE  # length of a step in scaled units
TERRAIN_LENGTH = 200  # length of terrain course, in steps
TERRAIN_HEIGHT = VIEWPORT_H / SCALE / 4  # base terrain height in scaled units
TERRAIN_STARTPAD = 20  # padding at start of course, in steps

TERRAIN_GRASS = 10  # how long the grass spots are, in steps
