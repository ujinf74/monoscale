import math
from typing import Iterable, Tuple

import cv2
import numpy as np


class LogOddsGrid:
    def __init__(
        self,
        resolution: float,
        width: int,
        height: int,
        origin_x: float,
        origin_y: float,
        free_update: float,
        occupied_update: float,
        occupied_probability: float = 0.65,
        free_probability: float = 0.35,
        floor: float = -4.0,
    ):
        self.resolution = float(resolution)
        self.width = int(width)
        self.height = int(height)
        self.origin_x = float(origin_x)
        self.origin_y = float(origin_y)
        self.free_update = float(free_update)
        self.occupied_update = float(occupied_update)
        # Where a cell stops being unknown. Raising the occupied threshold
        # buys precision at the cost of recall, and vice versa.
        self.occupied_probability = float(occupied_probability)
        self.free_probability = float(free_probability)
        # How much free history a cell may bank. At -4 the bank is seven
        # occupied votes deep: a cell driven past and carved free a hundred
        # times flips from eight misplaced close-range votes, which is how
        # ground already mapped free turned into obstacles as the car passed.
        self.floor = float(floor)
        self.log_odds = np.zeros((self.height, self.width), dtype=np.float32)
        self.observed = np.zeros((self.height, self.width), dtype=bool)

    def world_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        return (
            int(math.floor((x - self.origin_x) / self.resolution)),
            int(math.floor((y - self.origin_y) / self.resolution)),
        )

    def inside(self, cell: Tuple[int, int]) -> bool:
        x, y = cell
        return 0 <= x < self.width and 0 <= y < self.height

    @staticmethod
    def ray_cells(start: Tuple[int, int], end: Tuple[int, int]) -> Iterable[Tuple[int, int]]:
        x0, y0 = start
        x1, y1 = end
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        error = dx + dy
        while True:
            yield x0, y0
            if x0 == x1 and y0 == y1:
                break
            twice = 2 * error
            if twice >= dy:
                error += dy
                x0 += sx
            if twice <= dx:
                error += dx
                y0 += sy

    def integrate_point(self, point_xy) -> None:
        """Add occupied evidence to one cell, carving nothing on the way in.

        A triangulated obstacle is only as trustworthy as its depth, and depth
        from a 10 cm baseline is not trustworthy enough to declare everything
        in front of it empty. Carving from these rays erased real obstacles.
        """
        cell = self.world_to_cell(float(point_xy[0]), float(point_xy[1]))
        if not self.inside(cell):
            return
        x, y = cell
        self.log_odds[y, x] = min(4.0, self.log_odds[y, x] + self.occupied_update)
        self.observed[y, x] = True

    def integrate_ray(self, origin_xy, endpoint_xy, occupied: bool) -> None:
        start = self.world_to_cell(float(origin_xy[0]), float(origin_xy[1]))
        end = self.world_to_cell(float(endpoint_xy[0]), float(endpoint_xy[1]))
        cells = list(self.ray_cells(start, end))
        if not cells:
            return
        for x, y in cells[:-1] if occupied else cells:
            if self.inside((x, y)):
                self.log_odds[y, x] = max(
                    self.floor, self.log_odds[y, x] - self.free_update
                )
                self.observed[y, x] = True
        if occupied and self.inside(cells[-1]):
            x, y = cells[-1]
            self.log_odds[y, x] = min(
                4.0, self.log_odds[y, x] + self.occupied_update
            )
            self.observed[y, x] = True

    def message_values(self, inflation_radius: float) -> np.ndarray:
        probability = 1.0 / (1.0 + np.exp(-self.log_odds))
        values = np.full(self.log_odds.shape, -1, dtype=np.int8)
        values[self.observed & (probability < self.free_probability)] = 0
        values[
            self.observed
            & (probability >= self.free_probability)
            & (probability <= self.occupied_probability)
        ] = 50
        occupied = self.observed & (probability > self.occupied_probability)
        radius_cells = int(math.ceil(max(inflation_radius, 0.0) / self.resolution))
        if radius_cells > 0 and np.any(occupied):
            size = 2 * radius_cells + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
            occupied = cv2.dilate(occupied.astype(np.uint8), kernel).astype(bool)
        values[occupied] = 100
        return values
