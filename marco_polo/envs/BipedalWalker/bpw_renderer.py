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


"""Default rendering class for bipedal walker"""
import math
from typing import Any, cast, Optional, Union

# This is here to discard the ill-advised 'Hello' message that pygame prints
# by default on import, plus some deprecation warnings.
import contextlib

# redirecting stdout gets rid of pygame spam,
# redirecting stderr gets rid of warning for deprecated package management
# from pygame and/or dependencies
with contextlib.redirect_stdout(None), contextlib.redirect_stderr(None):
    import pygame

import imageio
from Box2D.b2 import circleShape  # type: ignore[import]

import numpy as np

from marco_polo.envs.BipedalWalker.bpw_constants import (
    SCALE,
    TERRAIN_HEIGHT,
    TERRAIN_LENGTH,
    TERRAIN_STEP,
    VIEWPORT_H,
    VIEWPORT_W,
)
from marco_polo.tools.types import RenderFrame


class Viewport:
    """Data for a viewport definintion (x and y box limits)"""

    def __init__(
        self, x_min: float, x_max: float, y_min: float, y_max: float, scale: float
    ) -> None:
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.scale = scale

    @property
    def pixel_x_offset(self) -> float:
        """offset x value, in pixels. i.e. x_min scaled to pixels"""
        return self.x_min * self.scale

    @property
    def pixel_x_min(self) -> float:
        """minimum x value, in pixels."""
        return 0

    @property
    def pixel_x_max(self) -> float:
        """maximum x value, in pixels."""
        return self.scale * self.get_size()[0]

    @property
    def pixel_y_min(self) -> float:
        """minimum y value, in pixels."""
        return 0

    @property
    def pixel_y_max(self) -> float:
        """maximum y value, in pixels."""
        return self.scale * self.get_size()[1]

    def __repr__(self) -> str:
        return f"Viewport(xmin={self.x_min}, xmax={self.x_max}, y_min={self.y_min}, y_max={self.y_max}, scale={self.scale})"

    def get_size(self) -> tuple[float, float]:
        """Return the size (width and height) of the viewport

        Returns
        -------
        tuple[float, float]
            width, height
        """
        return self.x_max - self.x_min, self.y_max - self.y_min

    def get_pixel_size(self) -> tuple[float, float]:
        """Return the size (width and height) of the viewport, in pixels

        Returns
        -------
        tuple[float, float]
            width, height
        """
        return self.pixel_x_max - self.pixel_x_min, self.pixel_y_max - self.pixel_y_min


ColorType = tuple[int, int, int]
PointType = tuple[float, float]
PolygonType = list[PointType]


def shift_point(point: PointType, delta_x: float, delta_y: float = 0) -> PointType:
    """Return a point that is shifted by the offsets

    Parameters
    ----------
    point: PointType
        The point to shift
    delta_x: float
        Change in X value
    delta_y: float, default = 0
        Change in Y value

    Returns
    -------
    PointType i.e. (float, float)
        The shifted point
    """
    return point[0] + delta_x, point[1] + delta_y


class BipedalWalkerRenderer:
    """Default renderer for Bipedal Walker class

    This is a pygame rewrite of the original rendering. It is intended to match
    the original output as much as possible.

    Attributes
    ----------
    self.parent: BipedalWalkerCustom
        Owner of this renderer
    self.cloud_poly: list[tuple[PolygonType, float, float]]
        Clouds to render
    self.lidar_render: int
        Step in render (used for deciding when to show lidar)
    """

    def __init__(self, parent: Any, mode: str = "rgb_array") -> None:
        """Create the renderer with the given parent

        The parent is the simulation environment. This renderer
        will pull the scroll position, the terrain, the walker
        object, and the lidar information from the parent. In
        addition, it will use the parent's random number generator.

        Parameters
        ----------
        parent: BipedalWalkerCustom
            the environment this class will render
        mode: str, default = "rgb_array"
            The rending mode. If the value is "rgb_array", this will
            return the frame as data. Anything else will not.
        """
        self.parent = parent
        self.render_mode = mode
        if self.render_mode != "rgb_array":
            raise NotImplementedError("only 'rgb_array' is supported as a render mode")

        self.scroll = self.parent.scroll
        self.surface = cast(pygame.Surface, None)  # this fixes a bunch of hassles
        self.cloud_poly: list[tuple[PolygonType, float, float]] = []
        self.lidar_render = 0
        self.scale_facter = SCALE
        self.reset()

    def scale_point(self, point: PointType) -> PointType:
        return point[0] * self.scale_facter, point[1] * self.scale_facter

    def draw_polygon(self, poly: PolygonType, color: ColorType) -> None:
        pygame.draw.polygon(self.surface, color=color, points=poly)

    def draw_lines(
        self, poly: PolygonType, color: ColorType, closed: bool, width: int
    ) -> None:
        pygame.draw.lines(
            self.surface, color=color, closed=closed, points=poly, width=width
        )

    def render_whole_env(
        self, filename: str, generate_terrain: bool = True, generate_clouds: bool = True
    ) -> None:
        """Save image of entire environment at once (without walker)

        Instead of an animated gif, as in the main rendering, this
        creates a single wide image showing all the terrain at once.

        The terrain and clouds are optionally generated if requested.
        This may or may not be needed, depending on how/where this is
        called.

        Parameters
        ----------
        filename: str
            filename to write file to. Will write to the current
            directory unless the name contains another path
        generate_terrain: bool, default = True
            whether to (re-)generate terrain prior to rendering.
            defaults to True
        generate_clouds: bool, default = True
            whether to (re-)generate clouds prior to rendering.
            defaults to True

        Side-effects
        ------------
        A file is written to disk.
        Additionally, if generate_terrain and/or generate_clouds are
        True, this will (re-)generate those items in the environment.
        That may change those values and will advance the random
        sequence.
        """
        # The surface used for this function is a different size than
        # the main surface, so back up the existing one.
        # It will be restored at the end of the function.
        backup_surface = self.surface

        width = round((TERRAIN_STEP * TERRAIN_LENGTH) * SCALE)
        viewport = Viewport(0, width, 0, VIEWPORT_H, SCALE)
        self.surface = pygame.Surface((width, VIEWPORT_H))

        if generate_terrain:
            self.parent.generate_terrain()
        if generate_clouds:
            self._generate_clouds()
        self._draw_sky(viewport)

        # cloud function has hard coded view limits
        cloud_color = (255, 255, 255)
        for poly, _, _ in self.cloud_poly:
            self.draw_polygon(poly, cloud_color)

        self._draw_terrain(viewport, self.parent.terrain_poly)
        self._draw_flag(viewport)

        data = self.get_image_data()
        imageio.imwrite(filename, data)
        self.surface = backup_surface

    def get_image_data(self) -> RenderFrame:
        data = pygame.surfarray.array3d(self.surface)
        data = np.rot90(data, axes=(0, 1))
        return data

    def _generate_clouds(self) -> None:
        """Generate clouds to be displayed in the rendering"""
        self.cloud_poly = []
        for _ in range(TERRAIN_LENGTH // 20):
            x_base = self.parent.np_random.uniform(0, TERRAIN_LENGTH) * TERRAIN_STEP
            y_base = VIEWPORT_H / SCALE * 3 / 4
            poly: PolygonType = [
                self.scale_point(
                    (
                        x_base
                        + 15 * TERRAIN_STEP * math.sin(3.14 * 2 * a / 5)
                        + self.parent.np_random.uniform(0, 5 * TERRAIN_STEP),
                        y_base
                        + 5 * TERRAIN_STEP * math.cos(3.14 * 2 * a / 5)
                        + self.parent.np_random.uniform(0, 5 * TERRAIN_STEP),
                    )
                )
                for a in range(5)
            ]
            x_vals = [p[0] for p in poly]
            x_min: float = min(x_vals)
            x_max: float = max(x_vals)
            self.cloud_poly.append((poly, x_min, x_max))

    def reset(self) -> None:
        """Reset the renderer

        Side-Effects
        ------------
        This will reset the lidar viewing sequence and regenerate
        the clouds.
        """
        self.lidar_render = 0
        # self._set_terrain_number()
        self._generate_clouds()

    def render(self, *args: Any, **kwargs: Any) -> Union[None, RenderFrame]:
        """Render the scene"""
        return self._render(*args, **kwargs)

    def _draw_sky(self, viewport: Viewport) -> None:
        """Draw the sky background"""
        width, height = viewport.get_size()
        x_min, x_max = self.scale_point((0, width))
        y_min, y_max = self.scale_point((0, height))
        poly = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
        sky_color = (230, 230, 255)
        self.draw_polygon(poly, sky_color)

    def _draw_clouds(
        self, viewport: Viewport, clouds: list[tuple[PolygonType, float, float]]
    ) -> None:
        """Draw the clouds"""
        half_scroll = 0.5 * self.scroll

        # the x position needs to be offset by this much for proper tracking
        x_adjust = half_scroll - viewport.pixel_x_offset
        cloud_color = (255, 255, 255)
        for poly, x_min, x_max in clouds:
            if x_max < half_scroll:
                continue
            if x_min > half_scroll + VIEWPORT_W:
                continue
            new_poly = [shift_point(p, delta_x=x_adjust) for p in poly]
            self.draw_polygon(new_poly, cloud_color)

    def _draw_terrain(self, viewport: Viewport, terrain: Any) -> None:
        """Draw the ground terrain"""
        for poly, color in terrain:
            if poly[1][0] < viewport.pixel_x_offset:
                continue
            if poly[0][0] > viewport.x_max * SCALE:
                continue
            new_poly = [shift_point(p, delta_x=-viewport.pixel_x_offset) for p in poly]
            self.draw_polygon(new_poly, color)

    def _draw_lidar(self, viewport: Viewport, lidar: Any) -> None:
        """Draw the lidar line, if appropriate"""
        self.lidar_render = (self.lidar_render + 1) % 50
        i = self.lidar_render
        if i < 2 * len(lidar):
            l = lidar[i] if i < len(lidar) else lidar[len(lidar) - i - 1]
            start_pos = self.scale_point(shift_point(l.point1, delta_x=-viewport.x_min))
            end_pos = self.scale_point(shift_point(l.point2, delta_x=-viewport.x_min))
            self.draw_lines(
                [start_pos, end_pos], color=(255, 0, 0), closed=False, width=1
            )

    def _draw_objects(self, viewport: Viewport) -> None:
        """Draw the objects of the scene"""
        for obj in self.parent.drawlist:
            for fixture in obj.fixtures:
                trans = fixture.body.transform
                if isinstance(fixture.shape, circleShape):
                    raise NotImplementedError("circle rendering not supported")
                path = [trans * v for v in fixture.shape.vertices]
                shifted_path = [
                    shift_point(point, delta_x=-viewport.x_min) for point in path
                ]
                scaled_path = [self.scale_point(point) for point in shifted_path]
                if len(path) > 2:
                    self.draw_polygon(scaled_path, obj.color1)
                else:
                    # paths of len two would be a thin line, which would be
                    # overwritten by the line drawing below, so they are ignored
                    pass
                self.draw_lines(
                    poly=scaled_path, color=obj.color2, closed=True, width=2
                )

    def _draw_flag(self, viewport: Viewport) -> None:
        """Draw the flag at the start of the course"""
        flag_y1 = TERRAIN_HEIGHT * SCALE
        flag_y2 = flag_y1 + 50
        flag_x = (TERRAIN_STEP * 3) * SCALE - viewport.pixel_x_offset

        pole = [(flag_x, flag_y1), (flag_x, flag_y2)]
        self.draw_lines(poly=pole, color=(0, 0, 0), closed=False, width=2)

        flag = [(flag_x, flag_y2), (flag_x, flag_y2 - 10), (flag_x + 25, flag_y2 - 5)]
        self.draw_polygon(poly=flag, color=(230, 51, 0))
        self.draw_lines(poly=flag, color=(0, 0, 0), closed=True, width=2)

    def _close_renderer(self) -> None:
        """Close and destroy the surface, if it exists"""
        if self.surface is not None:
            self.surface = cast(pygame.Surface, None)

    def _draw_scene(self, viewport: Viewport, kwargs: dict[str, Any]) -> None:
        """Draw the objects in the scene"""
        self._draw_sky(viewport)
        self._draw_clouds(viewport, self.cloud_poly)
        self._draw_terrain(viewport, self.parent.terrain_poly)
        self._draw_lidar(viewport, self.parent.lidar)
        self._draw_objects(viewport)
        self._draw_flag(viewport)

    def _render(self, close: bool = False, **kwargs: Any) -> Union[None, RenderFrame]:
        """Render the current scene

        Parameters
        ----------
        close: bool, default = False
            Whether to close the surface. If True, the surface is closed and
            destroyed.

        Returns
        -------
        If close is True, None is returned.
        Otherwise, if self.render_mode is "rgb_array", the frame will be returned
        as a byte array.
        In any other case, the status of the viewing window will be returned.
        This assumes a view window that may be open, but not visible.
        """
        if close:
            self._close_renderer()
            return None

        self.scroll = self.parent.scroll
        x_min = self.scroll
        x_max = x_min + VIEWPORT_W / SCALE
        y_min = 0
        y_max = VIEWPORT_H / SCALE
        viewport = Viewport(x_min, x_max, y_min, y_max, SCALE)

        if self.surface is None:
            self.surface = pygame.Surface((VIEWPORT_W, VIEWPORT_H))

        self._draw_scene(viewport, kwargs)

        return self.get_image_data()

    def render_all(
        self, data: list[Any], result: Optional[tuple[int, bool]]
    ) -> list[RenderFrame]:
        """Render all frames of the environment."""
        raise NotImplementedError("render_all not supported by base renderer")
