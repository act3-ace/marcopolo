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


"""Bipedal walker environment

This is simple 4-joints walker robot environment.

There are two versions:

- Normal, with slightly uneven terrain.

- Hardcore with ladders, stumps, pitfalls.

Reward is given for moving forward, total 300+ points up to the far end. If the robot
falls, it gets -100. Applying motor torque costs a small amount of points, more optimal
agent will get better score.

Heuristic is provided for testing, it's also useful to get demonstrations to
learn from. To run heuristic:

python gym/envs/box2d/bipedal_walker.py

State consists of hull angle speed, angular velocity, horizontal speed, vertical speed,
position of joints and joints angular speed, legs contact with ground, and 10 lidar
rangefinder measurements to help to deal with the hardcore version. There's no
coordinates in the state vector. Lidar is less useful in normal version, but it works.

To solve the game you need to get 300 points in 1600 time steps.

To solve hardcore version you need 300 points in 2000 time steps.

Created by Oleg Klimov. Licensed on the same terms as the rest of OpenAI Gym.
"""


import argparse  # for typing
import copy
import math
from collections import namedtuple
from collections.abc import Callable
from typing import Any, cast, Optional, Type, Union

import Box2D  # type: ignore[import]
import gymnasium as gym
import numpy as np
from Box2D.b2 import (  # type: ignore[import]
    contactListener,
    edgeShape,
    fixtureDef,
    polygonShape,
    revoluteJointDef,
)
from gymnasium import spaces
from gymnasium.utils import seeding
from pettingzoo.utils.env import ParallelEnv  # type: ignore[import]

from marco_polo.envs.BipedalWalker.bpw_constants import (
    SCALE,
    TERRAIN_GRASS,
    TERRAIN_HEIGHT,
    TERRAIN_STARTPAD,
    TERRAIN_STEP,
    VIEWPORT_H,
    VIEWPORT_W,
)
from marco_polo.envs.BipedalWalker.bpw_renderer import (
    BipedalWalkerRenderer,
    ColorType,
    PointType,
    PolygonType,
)
from marco_polo.envs.BipedalWalker.bpw_renderer_updated import (
    BipedalWalkerRendererEnhanced,
)
from marco_polo.envs.BipedalWalker.cppn import CppnEnvParams
from marco_polo.envs.BipedalWalker.terrain_generation import generate_terrain_coords
from marco_polo.tools.compiled import fast_clip, fast_sum
from marco_polo.tools.types import RenderFrame, Role


FPS = 50

MOTORS_TORQUE = 80
SPEED_HIP = 4
SPEED_KNEE = 6
LIDAR_RANGE = 160 / SCALE

INITIAL_RANDOM = 5

HULL_POLY = [(-30, +9), (+6, +9), (+34, +1), (+34, -8), (-30, -8)]
LEG_DOWN = -8 / SCALE
LEG_W, LEG_H = 8 / SCALE, 34 / SCALE

FRICTION = 2.5

HULL_FD = fixtureDef(
    shape=polygonShape(vertices=[(x / SCALE, y / SCALE) for x, y in HULL_POLY]),
    density=5.0,
    friction=0.1,
    categoryBits=0x0020,
    maskBits=0x001,  # collide only with ground
    restitution=0.0,
)  # 0.99 bouncy

LEG_FD = fixtureDef(
    shape=polygonShape(box=(LEG_W / 2, LEG_H / 2)),
    density=1.0,
    restitution=0.0,
    categoryBits=0x0020,
    maskBits=0x001,
)

LOWER_FD = fixtureDef(
    shape=polygonShape(box=(0.8 * LEG_W / 2, LEG_H / 2)),
    density=1.0,
    restitution=0.0,
    categoryBits=0x0020,
    maskBits=0x001,
)

# statics for the step function
HKHK_Vec = np.array(object=[SPEED_HIP, SPEED_KNEE, SPEED_HIP, SPEED_KNEE])
X_SCALE = 0.3 * (VIEWPORT_W / SCALE) / FPS
Y_SCALE = 0.3 * (VIEWPORT_H / SCALE) / FPS
SCROLL_SCALE = VIEWPORT_W / SCALE / 5
SHAPING_SCALE = 130 / SCALE
LIDAR_SCALE = 1.5 / 10.0

LIDAR_FULL = [
    (math.sin(i * LIDAR_SCALE) * LIDAR_RANGE, math.cos(i * LIDAR_SCALE) * LIDAR_RANGE)
    for i in range(10)
]

FakeLidar = namedtuple("FakeLidar", ["point1", "point2"])


def get_env_class(args: argparse.Namespace) -> Callable[..., Any]:
    """Returns the class to use, based on input arguments

    This function selects between BipedalWalkerCustom and
    BipedalWalkerCustomEnhanced based on the requested class

    The enhanced version is used if the input config contains:
    env_params:
      class: "BipedalWalkerCustomEnhanced"

    The regular version is used if the input config contains:
    env_params:
      class: "BipedalWalkerCustom"

    if anyother value for "class" is given, an error is raised.

    If env_params.class is omitted, BipedalWalkerCustom is used.

    Parameters
    ----------
    args: argparse.Namespace
        arguments that were passed to the `main()` function

    Returns
    -------
    class
        the class to use in creating env objects
    """
    if "env_params" in args:
        if "class" in args.env_params:
            class_str = args.env_params["class"]
            if class_str == "BipedalWalkerCustomEnhanced":
                return BipedalWalkerCustomEnhanced
            if class_str == "BipedalWalkerCustom":
                return BipedalWalkerCustom

            raise NotImplementedError(f"Unknown walker class: {class_str}")

    # Either no "env_params" or no "env_params.class".
    # That's fine, just return the default class
    return BipedalWalkerCustom


class ContactDetector(contactListener):  # type: ignore[misc] # contactListener is Any
    """Contact detector for Box2D physics."""

    def __init__(self, env: ParallelEnv) -> None:
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact: Any) -> None:  # pylint: disable=invalid-name
        """Handle contact starting."""
        if (
            self.env.hull == contact.fixtureA.body  # pylint: disable=consider-using-in
            or self.env.hull == contact.fixtureB.body
        ):
            self.env.game_over = True
        for leg in [self.env.legs[1], self.env.legs[3]]:
            if leg in [contact.fixtureA.body, contact.fixtureB.body]:
                leg.ground_contact = 1.0

    def EndContact(self, contact: Any) -> None:  # pylint: disable=invalid-name
        """Handle contact ending."""
        for leg in [self.env.legs[1], self.env.legs[3]]:
            if leg in [contact.fixtureA.body, contact.fixtureB.body]:
                leg.ground_contact = 1.0


class LidarCallback(Box2D.b2.rayCastCallback):  # type: ignore  # pylint: disable=too-few-public-methods
    """Callback class for lidar raycasting"""

    def ReportFixture(  # type: ignore
        self, fixture, point, _, fraction
    ):  # pylint: disable=invalid-name
        """Callback for Box2D"""
        if (fixture.filterData.categoryBits & 1) == 0:
            return 1
        self.point2 = point  # pylint: disable=attribute-defined-outside-init
        self.fraction = fraction  # pylint: disable=attribute-defined-outside-init
        return 0


class BipedalWalkerCustom(ParallelEnv):  # type: ignore[misc] # ParallelEnv is Any
    """BipedalWalkerCustom augmented gym environment

    Attributes
    ----------
    env_params: cppn.CppnEnvParams
        Parameters for this env, including mutation info
    self.renderer: BipedalWalkerRenderer
        Object to render visuals
    self.world: Box2D.b2World
        2D Physics world object
    self.terrain: list
        List of Box2D.b2World.StaticBody objects, ie objects with no dynamics
    self.hull: Box2D.b2World.CreateDynamicBody
        DynamicBody object for the walker, experiences physics like gravity
    self.prev_shaping: float
        If not None, subtract previous step reward off of current step reward each
        step resulting in reward that shows (pos or neg) improvement each step
    self.fd_polygon: Box2D.b2.fixtureDef
        Abstract fixture, I think this corresponds to the terrain the walker uses
        in the case where we're not using the cppn to create the terrain.
    self.fd_edge: Box2D.b2.fixtureDef
        Abstract fixture, I think this corresponds to the terrain the walker uses
        in the case where we are using the cppn to create the terrain.
    self.action_space = spaces.Box
        Gym action space
    """

    # multi-agent compatibility
    role_name = cast(Role, "default")
    possible_agents = [role_name]

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": FPS}

    def __repr__(self) -> str:
        return f"{self.__dict__}\nenv\n{self.__dict__['np_random'].get_state()}"

    def __init__(self) -> None:
        self.agents: list[Role] = []
        self.env_params: Optional[CppnEnvParams] = None
        self.seed()

        self.world = Box2D.b2World()
        self.terrain: list[Box2D.b2Body] = []
        self.terrain_poly: list[tuple[PolygonType, ColorType]] = []

        # components of walker
        self.hull: Optional[Box2D.b2Body] = None
        self.joints: list[Box2D.b2Joint] = []
        self.legs: list[Box2D.b2Body] = []

        self.current_step = 0  # the current step number in a simulation run

        self.prev_shaping = 0
        self.fd_polygon = fixtureDef(
            shape=polygonShape(vertices=[(0, 0), (1, 0), (1, -1), (0, -1)]),
            friction=FRICTION,
        )

        self.fd_edge = fixtureDef(
            shape=edgeShape(vertices=[(0, 0), (1, 1)]),
            friction=FRICTION,
            categoryBits=0x0001,
        )

        self.max_episode_steps = 2000  # default (may be changed in augment())

        # internal tracking of reward, only for rendering
        self.reward_to_now = 0.0  # sum of reward up to now
        self.final_reward: Optional[float] = None  # at completion of run

        self.terrain_len = 0  # will be updated in reset
        self.end_point = 0.0  # will be updated in reset
        self.finish = False

        self.scroll = 0.0

        self.render_mode = "rgb_array"
        self.renderer = self._RENDER_CLASS(self, mode=self.render_mode)
        self.reset()

        self.obs_buffer = np.zeros(24)  # to avoid creating new arrays constantly

    def augment(self, params: CppnEnvParams) -> None:
        """Add cppn environment parameters.

        Parameters
        ----------
        params: CppnEnvParams
            the cppn data to add
        """
        self.env_params = params
        self.max_episode_steps = params.get("max_episode_steps", self.max_episode_steps)

    def seed(self, seed: Optional[int] = None) -> None:
        """Set the random number seed.

        Parameters
        ----------
        seed: int, optional
            the random number seed
        """
        self.np_random, seed = seeding.np_random(seed)

    def _destroy(self) -> None:
        if not self.terrain:
            return
        self.world.contactListener = None
        for terrain in self.terrain:
            self.world.DestroyBody(terrain)
        self.terrain = []
        self.world.DestroyBody(self.hull)
        self.hull = None
        for leg in self.legs:
            self.world.DestroyBody(leg)
        self.legs = []
        self.joints = []
        self.world = None

    def generate_terrain(self) -> None:
        """Generate terrain for environment.

        To be useful, the env_params should be set first via `augment()`
        """
        terrain_x, terrain_y = generate_terrain_coords(self.env_params, self.np_random)

        # use the x, y values to generate the terrain pieces
        self.terrain = []
        self.terrain_poly = []
        for i_step in range(len(terrain_x) - 1):
            poly: list[PointType] = [
                (terrain_x[i_step], terrain_y[i_step]),
                (terrain_x[i_step + 1], terrain_y[i_step + 1]),
            ]
            self.fd_edge.shape.vertices = poly
            terrain = self.world.CreateStaticBody(fixtures=self.fd_edge)
            if i_step % 2 == 0:
                color = (77, 255, 77)
            else:
                color = (77, 204, 77)
            terrain.color1 = color
            terrain.color2 = color
            self.terrain.append(terrain)

            # Make the polygons for rendering. These are scaled to pixel sizes
            color = (102, 153, 77)
            render_polygon = [(point[0] * SCALE, point[1] * SCALE) for point in poly]
            # these points are the bottom of the polygon
            render_polygon += [(render_polygon[1][0], 0), (render_polygon[0][0], 0)]
            self.terrain_poly.append((render_polygon, color))
        # ???: it's not clear why this is reversed
        self.terrain.reverse()

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ) -> tuple[dict[Role, Any], dict[Role, dict[str, Any]]]:
        """Reset the env for a new run."""
        if seed is not None:
            self.seed(seed)
        self._destroy()
        self.world = Box2D.b2World()
        self.world.contactListener_bug_workaround = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_bug_workaround
        self.game_over = False
        self.prev_shaping = 0
        self.scroll = 0.0
        self.agents = [self.possible_agents[0]]

        # The step() function increments this each time, including at the end of
        # this function. Setting this to -1 means it will correctly start at zero
        # for the actual simulation run.
        self.current_step = -1

        self.generate_terrain()
        assert self.terrain is not None
        self.terrain_len = len(self.terrain) + 1
        self.end_point = (self.terrain_len - TERRAIN_GRASS) * TERRAIN_STEP

        init_x = TERRAIN_STEP * TERRAIN_STARTPAD / 2
        init_y = TERRAIN_HEIGHT + 2 * LEG_H
        self.hull = self.world.CreateDynamicBody(
            position=(init_x, init_y), fixtures=HULL_FD
        )
        self.hull.color1 = (127, 102, 230)
        self.hull.color2 = (77, 77, 127)
        self.hull.ApplyForceToCenter(
            (self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM), 0), True
        )

        self.legs = []
        self.joints = []
        for i in [-1, +1]:
            leg = self.world.CreateDynamicBody(
                position=(init_x, init_y - LEG_H / 2 - LEG_DOWN),
                angle=(i * 0.05),
                fixtures=LEG_FD,
            )
            leg.color1 = (153 - i * 26, 77 - i * 26, 127 - i * 26)
            leg.color2 = (102 - i * 26, 51 - i * 26, 77 - i * 26)
            rjd = revoluteJointDef(
                bodyA=self.hull,
                bodyB=leg,
                localAnchorA=(0, LEG_DOWN),
                localAnchorB=(0, LEG_H / 2),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=MOTORS_TORQUE,
                motorSpeed=i,
                lowerAngle=-0.8,
                upperAngle=1.1,
            )
            self.legs.append(leg)
            self.joints.append(self.world.CreateJoint(rjd))

            lower = self.world.CreateDynamicBody(
                position=(init_x, init_y - LEG_H * 3 / 2 - LEG_DOWN),
                angle=(i * 0.05),
                fixtures=LOWER_FD,
            )
            lower.color1 = (152 - i * 26, 77 - i * 26, 127 - i * 26)
            lower.color2 = (102 - i * 26, 51 - i * 26, 77 - i * 26)
            rjd = revoluteJointDef(
                bodyA=leg,
                bodyB=lower,
                localAnchorA=(0, -LEG_H / 2),
                localAnchorB=(0, LEG_H / 2),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=MOTORS_TORQUE,
                motorSpeed=1,
                lowerAngle=-1.6,
                upperAngle=-0.1,
            )
            lower.ground_contact = False
            self.legs.append(lower)
            self.joints.append(self.world.CreateJoint(rjd))

        self.drawlist = self.terrain + self.legs + [self.hull]

        self.lidar = [LidarCallback() for _ in range(10)]

        self.renderer.reset()
        self.obs_buffer = np.zeros(24)

        self.reward_to_now = 0.0
        self.final_reward = None  # None because we don't know this yet
        self.finish = False

        initial_obs, _, _, _, info = self.step({self.role_name: np.zeros(4)})

        return initial_obs, info

    def step(
        self, actions: dict[str, Any]
    ) -> tuple[
        dict[Role, Any],
        dict[Role, float],
        dict[Role, bool],
        dict[Role, bool],
        dict[Role, dict[str, Any]],
    ]:
        self.current_step += 1
        truncated = {self.role_name: False}
        terminated = {self.role_name: False}
        action = actions[self.role_name]

        # Uncomment the next line to receive a bit of stability help
        # self.hull.ApplyForceToCenter((0, 20), True)
        control_speed = False  # Should be easier as well

        motor_torques = MOTORS_TORQUE * np.clip(
            a=np.abs(action), a_min=0, a_max=1, dtype=float
        )
        reward = -0.00035 * np.sum(a=motor_torques)

        if control_speed:
            motor_speeds = HKHK_Vec * np.clip(a=action, a_min=-1, a_max=1, dtype=float)
            for i in range(4):
                self.joints[i].motorSpeed = motor_speeds[i]

        else:
            motor_speeds = HKHK_Vec * np.sign(action, dtype=float)
            for i in range(4):
                self.joints[i].motorSpeed = motor_speeds[i]
                self.joints[i].maxMotorTorque = motor_torques[i]

        # self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        self.world.Step(0.02, 180, 60)

        assert self.hull is not None
        pos = self.hull.position
        vel = self.hull.linearVelocity

        lidar_frac = []
        for i in range(10):
            self.lidar[i].fraction = 1.0
            self.lidar[i].point1 = pos
            self.lidar[i].point2 = (
                pos[0] + LIDAR_FULL[i][0],
                pos[1] - LIDAR_FULL[i][1],
            )
            self.world.RayCast(
                self.lidar[i], self.lidar[i].point1, self.lidar[i].point2
            )
            lidar_frac.append(self.lidar[i].fraction)

        state = [
            # Normal angles up to 0.5 here, but sure more is possible.
            self.hull.angle,
            2.0 * self.hull.angularVelocity / FPS,
            # Normalized to get -1..1 range
            vel.x * X_SCALE,
            vel.y * Y_SCALE,
            # This will give 1.1 on high up, but it's still OK
            # There should be spikes on hitting the ground, that's normal too
            self.joints[0].angle,
            self.joints[0].speed / SPEED_HIP,
            self.joints[1].angle + 1.0,
            self.joints[1].speed / SPEED_KNEE,
            self.legs[1].ground_contact,
            self.joints[2].angle,
            self.joints[2].speed / SPEED_HIP,
            self.joints[3].angle + 1.0,
            self.joints[3].speed / SPEED_KNEE,
            self.legs[3].ground_contact,
        ]

        state.extend(lidar_frac)

        self.scroll = pos.x - SCROLL_SCALE

        # moving forward is a way to receive reward (normalized to get 300 on completion)
        shaping = pos[0] * SHAPING_SCALE
        # keep head straight, other than that and falling, any behavior is unpunished
        shaping -= 5.0 * abs(state[0])

        reward += shaping - self.prev_shaping
        self.prev_shaping = shaping

        self.reward_to_now += reward

        if self.game_over or pos[0] < 0:
            reward = -100
            terminated = {self.role_name: True}
            self.reward_to_now = reward
            self.final_reward = self.reward_to_now
            self.agents = []

        if pos[0] > self.end_point:
            terminated = {self.role_name: True}
            self.final_reward = self.reward_to_now
            self.finish = True
            self.agents = []

        if self.current_step == self.max_episode_steps:
            truncated = {self.role_name: True}
            self.agents = []

        # reshape for multiagent compatibility
        observations = {self.role_name: np.array(state)}
        rewards = {self.role_name: reward}
        info = {self.role_name: {"finish": self.finish}}

        return observations, rewards, terminated, truncated, info

    def _step_noviz(
        self, actions: dict[str, Any]
    ) -> tuple[
        dict[Role, Any],
        dict[Role, float],
        dict[Role, bool],
        dict[Role, bool],
        dict[Role, dict[str, Any]],
    ]:
        """Step the environment without visualization support

        This may have a slight performance gain over step() as it
        does not calculate/store data used for visualization.
        It can otherwise be used just like step()
        """
        self.current_step += 1
        truncated = {self.role_name: False}
        terminated = {self.role_name: False}
        action = actions[self.role_name]

        # cache these lookups
        assert self.hull is not None
        hull = self.hull
        joints = self.joints
        world = self.world
        state = self.obs_buffer

        control_speed = False  # Should be easier as well

        motor_torques = MOTORS_TORQUE * fast_clip(
            np.abs(action).astype(float), min=0, max=1
        )
        reward = -0.00035 * fast_sum(motor_torques)

        if control_speed:
            motor_speeds = HKHK_Vec * fast_clip(action.astype(float), min=-1, max=1)
            for i in range(4):
                joints[i].motorSpeed = motor_speeds[i]
        else:
            motor_speeds = HKHK_Vec * np.sign(action, dtype=float)
            for i in range(4):
                joints[i].motorSpeed = motor_speeds[i]
                joints[i].maxMotorTorque = motor_torques[i]

        # self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        world.Step(0.02, 180, 60)

        pos = hull.position
        vel = hull.linearVelocity

        # Normal angles up to 0.5 here, but sure more is possible.
        state[0] = hull.angle
        state[1] = 2.0 * hull.angularVelocity / FPS
        # Normalized to get -1..1 range
        state[2] = vel.x * X_SCALE
        state[3] = vel.y * Y_SCALE
        # This will give 1.1 on high up, but it's still OK
        # (and there should be spikes on hiting the ground, that's normal too)
        state[4] = joints[0].angle
        state[5] = joints[0].speed / SPEED_HIP
        state[6] = joints[1].angle + 1.0
        state[7] = joints[1].speed / SPEED_KNEE
        state[8] = self.legs[1].ground_contact
        state[9] = joints[2].angle
        state[10] = joints[2].speed / SPEED_HIP
        state[11] = joints[3].angle + 1.0
        state[12] = joints[3].speed / SPEED_KNEE
        state[13] = self.legs[3].ground_contact

        lidar = self.lidar[0]
        for i in range(10):
            lidar.fraction = 1.0
            point2 = (pos[0] + LIDAR_FULL[i][0], pos[1] - LIDAR_FULL[i][1])
            world.RayCast(lidar, pos, point2)
            state[14 + i] = lidar.fraction

        # moving forward is a way to receive reward (normalized to get 300 on completion)
        shaping = pos[0] * SHAPING_SCALE
        # keep head straight, other than that and falling, any behavior is unpunished
        shaping -= 5.0 * abs(state[0])

        reward += shaping - self.prev_shaping
        self.prev_shaping = shaping

        self.finish = False
        if self.game_over or pos[0] < 0:
            reward = -100
            terminated = {self.role_name: True}
        if pos[0] > self.end_point:
            terminated = {self.role_name: True}
            self.finish = True

        if self.current_step == self.max_episode_steps:
            truncated = {self.role_name: True}
            self.agents = []

        # reshape for multiagent compatibility
        observations = {self.role_name: state}
        rewards = {self.role_name: reward}
        info = {self.role_name: {"finish": self.finish}}

        return observations, rewards, terminated, truncated, info

    def observation_space(self, agent: Role) -> gym.spaces.Box:
        """Return observation space for the agent

        Parameters
        ----------
        agent: Role
            The agent to get the observation space for

        Returns
        -------
        gym.spaces.Space
            The observation space for the agent
        """
        return self.observation_spaces[agent]

    def action_space(self, agent: Role) -> gym.spaces.Box:
        """Return action space for the agent

        Parameters
        ----------
        agent: Role
            The agent to get the action space for

        Returns
        -------
        gym.spaces.Space
            The action space for the agent
        """
        return self.action_spaces[agent]

    _RENDER_CLASS: Type[BipedalWalkerRenderer] = BipedalWalkerRenderer

    def render(self, *args: Any, **kwargs: Any) -> Union[None, RenderFrame]:
        """Render a frame of the environment run."""
        return self.renderer.render(*args, **kwargs)

    def render_all(self, *args: Any, **kwargs: Any) -> list[RenderFrame]:
        """Render all frames of the environment run."""
        return self.renderer.render_all(*args, **kwargs)

    def bundle_step_data(self) -> Any:
        """Return rendering data for the current situation

        This is used to save the current state when doing batch rendering.

        Returns
        -------
        tuple[scroll, lidars, draw_list_data, step]
            data for the frames.
            scroll is the x scroll position of the state
            lidars is a list of lidar-like objects storing the lida points
            draw_list_data is a list of object drawing data
            step is the simulation step (not the frame step) for this frame
        """
        lidars = [FakeLidar([l.point1[0], l.point1[1]], l.point2) for l in self.lidar]
        draw_list_data = []
        for obj in self.drawlist:
            fixtures = []
            for fixture in obj.fixtures:
                trans = fixture.body.transform
                if isinstance(fixture.shape, Box2D.b2.circleShape):
                    translation = trans * fixture.shape.pos
                    fixtures.append(
                        [
                            "circle",
                            fixture.shape.radius,
                            copy.deepcopy(translation),
                            obj.color1,
                            obj.color2,
                        ]
                    )
                else:
                    path = [trans * v for v in fixture.shape.vertices]
                    path.append(path[0])
                    fixtures.append(["other", path[:], obj.color1, obj.color2])
            draw_list_data.append(fixtures)
        return (
            self.scroll,
            lidars,
            draw_list_data,
            self.current_step,
            self.reward_to_now,
        )

    # these are taken from the BPW in gymnasium
    observation_spaces = {
        role_name: spaces.Box(
            low=np.array(
                [
                    -math.pi,
                    -5.0,
                    -5.0,
                    -5.0,
                    -math.pi,
                    -5.0,
                    -math.pi,
                    -5.0,
                    -0.0,
                    -math.pi,
                    -5.0,
                    -math.pi,
                    -5.0,
                    -0.0,
                ]
                + [-1.0] * 10
            ).astype(np.float32),
            high=np.array(
                [
                    math.pi,
                    5.0,
                    5.0,
                    5.0,
                    math.pi,
                    5.0,
                    math.pi,
                    5.0,
                    5.0,
                    math.pi,
                    5.0,
                    math.pi,
                    5.0,
                    5.0,
                ]
                + [1.0] * 10
            ).astype(np.float32),
        )
    }

    action_spaces = {
        role_name: spaces.Box(
            np.array([-1, -1, -1, -1]).astype(np.float32),
            np.array([1, 1, 1, 1]).astype(np.float32),
        )
    }


class BipedalWalkerCustomEnhanced(BipedalWalkerCustom):
    """Same as BipedalWalkerCustom, except using the enhanced renderer"""

    _RENDER_CLASS = BipedalWalkerRendererEnhanced
