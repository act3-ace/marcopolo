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


"""Updated rendering class for bipedal walker"""

from typing import Any, Optional

import pygame

from marco_polo.envs.BipedalWalker.bpw_constants import (
    SCALE,
    TERRAIN_HEIGHT,
    TERRAIN_STARTPAD,
    TERRAIN_STEP,
    VIEWPORT_H,
    VIEWPORT_W,
)
from marco_polo.envs.BipedalWalker.bpw_renderer import (
    BipedalWalkerRenderer,
    PolygonType,
    shift_point,
    Viewport,
)


class BipedalWalkerRendererEnhanced(BipedalWalkerRenderer):
    """Updated renderer for BipedalWalkerCustom

    This will function as a drop in replacement for the original
    renderer. However, some optional features require minor changes
    to the calling code.

    This has the following changes over the orginal renderer:
    * scales the vertical position to keep the walker in frame.
    * draws a path, showing the full terrain with a moving viewport.
    * adds an end flag marker.
    * add a frame step progress bar showing how far the run has progressed
      in simulation steps.
    * displays the current reward.

    Lastly, this class supports render_all(), which renders all of the
    frames in one call. When using this, the renderer will make these
    additional changes:
    * padding frames will be added at the end to of the video to pause
      it briefly at the end when looping the video.
    * the outcome status of the run will be shown as a color coding on
      the progress bar.
    * The final reward will be shown throughout, if known
    """

    def __init__(
        self, parent: Any, n_padding_frames: int = 10, mode: str = "rgb_array"
    ) -> None:
        """Initialize the object

        Parameters
        ----------
        parent: BipedalWalkerCustom
            The environment to render
        n_padding_frames: int, default = 10
            The number of frames to pad the end of the video with
        mode: str, default = "rgb_array"
            The rending mode. If the value is "rgb_array", this will
            return the frame as data. Anything else will not.
        """
        super().__init__(parent, mode=mode)
        self._n_padding_frames = n_padding_frames
        self._colors = {
            "black": (0, 0, 0),
            "white": (255, 255, 255),
            "gray": (127, 127, 127),
        }
        pygame.font.init()
        self.font = pygame.font.SysFont(None, 24)  # type: ignore  # None is default font

    def render_all(
        self, data: list[Any], result: Optional[tuple[int, bool]]
    ) -> list[Any]:
        """Render all of the frames at once

        Given the data for all the frames, this will return a list of
        all frames for the simulation. See class docstring for deatils
        or the changes made to rendering in this case

        Parameters
        ----------
        data: list[(float, list[Any], list[Any], int, float)]
            list of data for all frames. Each frame will have the following
            in a tuple: (scroll, lidars, draw_list_data, step, reward), where
            scroll is the x scroll position of the frame
            lidars is a list of lidar-like objects
            draw_list_data is a list of objectdrawing data
            step is the simulation step (not the frame step) for this frame
            reward is the current reward
        result tuple[int, bool] or None:
            The end result of the simulation run. This is a tuple
            (end_step, pass_fail), where
            end_step is the simulation step where this run ends
            pass_fail is bool for whether the run passed

        Return
        ------
        list[frames]
            An ordered list of all frames for the renderering, possibly
            including padded frames at the end. (see class docstring)
        """
        frames = []
        # generate each image frame
        for scroll, lidars, draw_list_data, step, reward in data:
            frame = self.render(
                scroll=scroll,
                render_data=[lidars, draw_list_data, step, reward],
                result=result,
            )
            frames.append(frame)
        # pad the end with duplicate frames as needed
        for _ in range(self._n_padding_frames):
            frames.append(frames[-1])
        return frames

    def _draw_scene(self, viewport: Viewport, kwargs: dict[str, Any]) -> None:
        """Draw the objects in the scene

        Parameters
        ----------
        viewport: Viewport
            the current viewport
        kwargs: dict[str, Any]
            optional additional arguments. The following are supported:
            * 'scroll': float -> replaces the viewpoint scroll with the
              given value
            * 'result': [last_step: int, pass_fail: bool] -> renders the
              status of the run on the progress bar. last_step is the
              index of the step where the run will end, pass_fail is
              whether the run made it to the end of the course.
            * 'render_data': [lidars: data, draw_list_data: data, step: int] ->
              data to render in place of the current data. This is
              used when batch rendering from cached data. lidars
              is a list of FakeLidar objects contining the lidar
              points. draw_list_data is a list of object data to draw.
              See _draw_cached_objects for specifications.
        """
        # if there is a passed scroll value, overwrite the existing value
        if "scroll" in kwargs:
            self.scroll = kwargs["scroll"]
            viewport.x_min = self.scroll
            viewport.x_max = viewport.x_min + VIEWPORT_W / SCALE

        step = self.parent.current_step

        # calculate the vertical shift and apply it to vertical extents
        y_shift = self._calc_y_shift()
        viewport.y_min = viewport.y_min - y_shift
        viewport.y_max = viewport.y_max - y_shift

        # shift the clouds so they stay pinned to the top
        shifted_clouds = self._shift_clouds(y_shift)
        super()._draw_sky(viewport)
        super()._draw_clouds(viewport, shifted_clouds)
        super()._draw_terrain(viewport, self.parent.terrain_poly)

        # render_all_data contains the data for the updated renderer
        if "render_data" in kwargs:
            lidars, draw_list_data, step, reward = kwargs["render_data"]
            self._handle_lidar(viewport, lidars, kwargs)
            self._draw_cached_objects(viewport, draw_list_data)
        else:
            self._handle_lidar(viewport, self.parent.lidar, kwargs)
            super()._draw_objects(viewport)

        super()._draw_flag(viewport)  # starting flag

        self._draw_end_flag(viewport, y_shift)
        self._draw_step_progress_bar(viewport, step, kwargs.get("result", []))
        self._draw_path(viewport, y_shift)

        if "render_data" in kwargs:
            # reward was set earlier
            total_reward = self.parent.final_reward
            if reward is not None:
                self._draw_reward(
                    viewport, current_reward=reward, total_reward=total_reward
                )
        else:
            reward = self.parent.reward_to_now
            total_reward = self.parent.final_reward
            if reward is not None:
                self._draw_reward(
                    viewport, current_reward=reward, total_reward=total_reward
                )

    def _handle_lidar(
        self, viewport: Viewport, lidars: list[Any], kwargs: dict[str, Any]
    ) -> None:
        if kwargs.get("all_lidar", False):
            self._draw_lidar(viewport, lidars)
        else:
            super()._draw_lidar(viewport, lidars)

    def _draw_lidar(self, viewport: Viewport, lidar: list[Any]) -> None:
        for lidar_ in lidar:
            points = [
                self.scale_point(shift_point(point, delta_x=-viewport.x_min))
                for point in [lidar_.point1, lidar_.point2]
            ]
            self.draw_lines(points, color=(255, 0, 0), closed=False, width=1)

    def _draw_end_flag(self, viewport: Viewport, y_shift: float) -> None:
        """Draw a checkered flag at the end of the course

        Parameters
        ----------
        viewport: Viewport
            the current viewport
        y_shift: float
            the y shift factor for the current view
        """
        # if the flag is out of the viewport, don't draw it.
        flag_x = TERRAIN_STEP * (self.parent.terrain_len - 1) - viewport.x_min
        if flag_x > self.scroll + VIEWPORT_W / SCALE:
            return

        flag_y_top = TERRAIN_HEIGHT + 50 / SCALE

        # flag pole
        pole = [
            self.scale_point((flag_x, viewport.pixel_y_min)),
            self.scale_point((flag_x, flag_y_top)),
        ]
        self.draw_lines(poly=pole, color=self._colors["black"], closed=False, width=2)

        #  Flag coordinates:
        #
        #  *                 <-  y1
        #  |     *           <-  y2
        #  *-----*------*    <-  y3
        #  |     *           <-  y4
        #  *                 <-  y5
        #  ^     ^      ^
        #  |     |      |
        #  x1    x2     x3
        #
        x_vals = [flag_x + i / SCALE for i in [0, 10, 25]]
        y_vals = [flag_y_top - i / SCALE for i in (0, 2, 5, 8, 10)]

        # flag and checkers for flag
        f_flag = [
            self.scale_point((x_vals[0], y_vals[0])),
            self.scale_point((x_vals[0], y_vals[4])),
            self.scale_point((x_vals[2], y_vals[2])),
            self.scale_point((x_vals[0], y_vals[0])),
        ]
        f_check1 = [
            self.scale_point((x_vals[0], y_vals[0])),
            self.scale_point((x_vals[0], y_vals[2])),
            self.scale_point((x_vals[1], y_vals[2])),
            self.scale_point((x_vals[1], y_vals[1])),
        ]
        f_check2 = [
            self.scale_point((x_vals[0], y_vals[2])),
            self.scale_point((x_vals[0], y_vals[4])),
            self.scale_point((x_vals[1], y_vals[3])),
            self.scale_point((x_vals[1], y_vals[2])),
        ]
        f_check3 = [
            self.scale_point((x_vals[1], y_vals[1])),
            self.scale_point((x_vals[1], y_vals[2])),
            self.scale_point((x_vals[2], y_vals[2])),
        ]
        f_check4 = [
            self.scale_point((x_vals[1], y_vals[3])),
            self.scale_point((x_vals[1], y_vals[2])),
            self.scale_point((x_vals[2], y_vals[2])),
        ]

        self.draw_polygon(f_check1, color=self._colors["black"])
        self.draw_polygon(f_check2, color=self._colors["white"])
        self.draw_polygon(f_check3, color=self._colors["white"])
        self.draw_polygon(f_check4, color=self._colors["black"])
        self.draw_lines(poly=f_flag, color=self._colors["gray"], closed=True, width=2)

    def _shift_clouds(self, y_shift: float) -> list[tuple[PolygonType, float, float]]:
        """Return the clouds shifted down by y_shift

        Parameters
        ----------
        y_shift: float
            the y shift factor for the current view

        Returns
        -------
        list[tuple[PolygonType, float, float]]
            A list of the clouds, with each cloud given as a list
            of points and the x_min and x_max value of the cloud.
        """
        clouds_new = [
            (
                [shift_point(point, delta_x=0, delta_y=-y_shift) for point in poly],
                x1,
                x2,
            )
            for poly, x1, x2 in self.cloud_poly
        ]
        return clouds_new

    def _calc_y_shift(self) -> float:
        """Return the shift factor for the y direction

        Returns
        -------
        float
            the y shift factor for the current view
        """
        # find the terrain segment that the walker is on. The height is
        # the average of that segment's height.
        # calculate shift by forcing the shifted height to be in a
        # target range
        x_target = self.scroll + TERRAIN_STEP * TERRAIN_STARTPAD / 2
        y_target = 0
        for poly, _ in self.parent.terrain_poly:
            if poly[0][0] <= x_target <= poly[1][0]:
                y_target = 0.5 * (poly[0][1] + poly[1][1])
                break

        if y_target < 2.0:
            return 2.0 - y_target
        if y_target > 8.0:
            return 8.0 - y_target
        return 0.0  # within view box, no shift needed

    def _draw_cached_objects(
        self, viewport: Viewport, obj_data: list[tuple[Any]]
    ) -> None:
        """Draw the objects of the scene

        This differs from the orginal _draw_objects method by using
        cached data instead of current data.

        Parameters
        ----------
        viewport: Viewport
            the current viewport
        obj_data: list[object data]
            The cached data is an array of objects, where each object
            is a list of fixtures. The format of the fixture depends
            on the type of fixture.
            For a circle: "circle", radius, transform, color1, color2
            For others: "other", path, color1, color2
        """
        for obj in obj_data:
            for fixture in obj:
                if fixture[0] == "circle":
                    raise NotImplementedError("circle rendering not supported")
                _, path, color1, color2 = fixture
                path = [
                    self.scale_point(shift_point(point, delta_x=-viewport.x_min))
                    for point in path
                ]
                if len(path) > 3:
                    self.draw_polygon(path[:-1], color=color1)
                self.draw_lines(path, color=color2, closed=False, width=2)

    def _draw_step_progress_bar(
        self, viewport: Viewport, step: int, result: tuple[int, bool]
    ) -> None:
        """Draw a bar to indicate frame progress

        Parameters
        ----------
        vieport: Viewport
            the current viewport
        step: int
            the current frame step index, in range [0, parent.max_episode_steps]
        result: tuple[int, bool]
            the result of the simulation run. The tuple includes the
            step when the run ends and a bool indicating whether is
            passed or failed.
        """
        percent_done = step / self.parent.max_episode_steps

        # so bar doesn't take full width
        width, height = viewport.get_pixel_size()
        x_min = 0.03 * width
        x_max = 0.97 * width
        y_min = 0.92 * height
        y_max = 0.97 * height

        progress = 0.94 * width * percent_done + x_min
        progress_color = (0, 0, 204)
        bg_color = (204, 204, 204)

        progress_box = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
        self.draw_polygon(poly=progress_box, color=bg_color)

        # fill progress bar line
        progress_bar = [
            (x_min, y_min),
            (progress, y_min),
            (progress, y_max),
            (x_min, y_max),
        ]
        self.draw_polygon(poly=progress_bar, color=progress_color)

        # Draw the final result in the progress bar
        # if the run succeeds, color green from the sucess point to the end
        # if the run fails, color red from the sucess point to the end
        if result:
            color_pass_fail = {True: (179, 255, 179), False: (255, 179, 179)}
            end_step, pass_fail = result
            end_percent = end_step / self.parent.max_episode_steps
            end_point = 0.94 * width * end_percent + x_min
            result_bar = [
                (end_point, y_min),
                (x_max, y_min),
                (x_max, y_max),
                (end_point, y_max),
            ]
            self.draw_polygon(poly=result_bar, color=color_pass_fail[pass_fail])

        # draw border around progress bar
        self.draw_lines(
            poly=progress_box, color=self._colors["black"], closed=True, width=2
        )

    def _draw_path(self, viewport: Viewport, y_shift: float) -> None:
        """Draw the total path with a viewport indicator

        Parameters
        ----------
        viewport: Viewport
            the current viewport
        y_shift: float
            the y shift factor for the current view
        """
        x_min, x_max = viewport.x_min, viewport.x_max
        _, p_height = viewport.get_pixel_size()
        delta_x = x_max - x_min
        y_shift -= 1

        terrain_len = len(self.parent.terrain) + 1
        percent_length = viewport.x_min / (TERRAIN_STEP * terrain_len)
        mini_height = 0.2 * p_height
        mini_width = mini_height * VIEWPORT_W / VIEWPORT_H
        mini_x_min = percent_length * VIEWPORT_W
        mini_y_min = p_height * 0.00

        x_scale = delta_x / (TERRAIN_STEP * terrain_len)
        new_poly = [poly[0] for poly, _ in self.parent.terrain_poly]
        new_poly.append(self.parent.terrain_poly[-1][0][1])  # last point
        heights = [p[1] for p in new_poly]
        height_min, height_max = min(heights), max(heights)
        y_zero = new_poly[0][1]
        y_base = mini_height / 2 - y_zero
        diff = max(abs(height_max - y_zero), abs(height_min - y_zero))
        if diff <= mini_height / 2:  # the plot fits within the range
            y_scale = 1
        else:
            y_scale = 2 / diff

        # draw the terrain line
        for points in zip(new_poly[:-1], new_poly[1:]):
            point1, point2 = [
                (xx * x_scale, yy * y_scale + y_base) for xx, yy in points
            ]
            self.draw_lines(
                [point1, point2], color=self._colors["black"], closed=False, width=1
            )
        # draw viewpoint indicator
        mini_viewport = Viewport(
            mini_x_min,
            mini_x_min + mini_width,
            mini_y_min,
            mini_y_min + mini_height,
            SCALE,
        )
        color_dark_gray = (77, 77, 77)
        color_dark_red = (204, 77, 77)
        box = [
            (mini_viewport.x_min, mini_viewport.y_min),
            (mini_viewport.x_max, mini_viewport.y_min),
            (mini_viewport.x_max, mini_viewport.y_max),
            (mini_viewport.x_min, mini_viewport.y_max),
        ]
        self.draw_lines(poly=box, color=color_dark_gray, closed=True, width=1)

        # draw bar on mini viewer to indicate where walker is
        point1 = (mini_x_min + 0.25 * mini_width, mini_y_min)
        point2 = (mini_x_min + 0.25 * mini_width, mini_y_min + mini_height)
        self.draw_lines(
            poly=[point1, point2], color=color_dark_red, closed=False, width=1
        )

    def _draw_reward(
        self,
        viewport: Viewport,
        current_reward: float,
        total_reward: Optional[float] = None,
    ) -> None:
        y_font = 0.875 * viewport.get_pixel_size()[1]
        x_font_start1 = 0.03 * viewport.get_pixel_size()[0]
        x_font_end2 = 0.97 * viewport.get_pixel_size()[0]  # end point of total reward

        # current reward
        text = self.font.render(f"Reward: {current_reward:.2f}", True, (0, 0, 0))
        text = pygame.transform.rotate(text, 180)
        text = pygame.transform.flip(text, True, False)
        self.surface.blit(text, [x_font_start1, y_font])

        if total_reward is not None:
            text = self.font.render(
                f"Final reward: {total_reward:.2f}", True, (0, 0, 0)
            )
            text = pygame.transform.rotate(text, 180)
            text = pygame.transform.flip(text, True, False)
            text_width = text.get_width()  # use to find where to put text
            self.surface.blit(text, [x_font_end2 - text_width, y_font])
