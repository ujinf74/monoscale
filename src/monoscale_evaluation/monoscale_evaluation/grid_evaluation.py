"""Scores one occupancy grid against another, resampling between them.

The vision grid is a fixed global map and the LiDAR reference grid is a local
one that travels with the vehicle, at a different resolution. Both messages
carry their own origin, so the comparison walks the reference cells, converts
each centre to map coordinates, and samples the vision grid there.
"""

from dataclasses import dataclass
import math
from typing import Dict, Optional

import numpy as np

UNKNOWN = -1
OCCUPIED_THRESHOLD = 65
FREE_THRESHOLD = 35


@dataclass(frozen=True)
class Grid:
    values: np.ndarray  # (height, width), -1 unknown else 0..100
    resolution: float
    origin_x: float
    origin_y: float
    origin_yaw: float = 0.0

    @property
    def height(self) -> int:
        return self.values.shape[0]

    @property
    def width(self) -> int:
        return self.values.shape[1]

    def cell_centres(self):
        """Map coordinates of every cell centre, as two (height, width) arrays."""
        xs = (np.arange(self.width) + 0.5) * self.resolution
        ys = (np.arange(self.height) + 0.5) * self.resolution
        grid_x, grid_y = np.meshgrid(xs, ys)
        c = math.cos(self.origin_yaw)
        s = math.sin(self.origin_yaw)
        return (
            self.origin_x + c * grid_x - s * grid_y,
            self.origin_y + s * grid_x + c * grid_y,
        )

    def sample(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Nearest cell value at map coordinates, UNKNOWN outside the grid."""
        c = math.cos(-self.origin_yaw)
        s = math.sin(-self.origin_yaw)
        dx = np.asarray(x) - self.origin_x
        dy = np.asarray(y) - self.origin_y
        local_x = c * dx - s * dy
        local_y = s * dx + c * dy
        columns = np.floor(local_x / self.resolution).astype(np.int64)
        rows = np.floor(local_y / self.resolution).astype(np.int64)
        inside = (
            (columns >= 0)
            & (columns < self.width)
            & (rows >= 0)
            & (rows < self.height)
        )
        out = np.full(columns.shape, UNKNOWN, dtype=np.int16)
        out[inside] = self.values[rows[inside], columns[inside]]
        return out


def compare_grids(
    candidate: Grid,
    reference: Grid,
    max_range_m: float = 0.0,
    reference_origin: Optional[tuple] = None,
) -> Dict[str, float]:
    """Score `candidate` where `reference` has an opinion.

    Cells the reference calls unknown are excluded outright: a LiDAR grid does
    not see behind walls either, and charging vision for that would say more
    about occlusion than about the mapper.
    """
    x, y = reference.cell_centres()
    reference_values = reference.values
    considered = reference_values != UNKNOWN
    if max_range_m > 0.0 and reference_origin is not None:
        distance = np.hypot(x - reference_origin[0], y - reference_origin[1])
        considered &= distance <= max_range_m
    if not np.any(considered):
        return {'compared_cells': 0}

    sampled = candidate.sample(x, y)
    reference_occupied = considered & (reference_values >= OCCUPIED_THRESHOLD)
    reference_free = considered & (reference_values <= FREE_THRESHOLD)
    candidate_occupied = considered & (sampled >= OCCUPIED_THRESHOLD)
    candidate_free = considered & (sampled <= FREE_THRESHOLD) & (sampled != UNKNOWN)
    candidate_unknown = considered & (sampled == UNKNOWN)

    true_positive = int(np.count_nonzero(reference_occupied & candidate_occupied))
    false_positive = int(np.count_nonzero(candidate_occupied & ~reference_occupied))
    false_negative = int(np.count_nonzero(reference_occupied & ~candidate_occupied))
    union = true_positive + false_positive + false_negative

    return {
        'compared_cells': int(np.count_nonzero(considered)),
        'reference_occupied_cells': int(np.count_nonzero(reference_occupied)),
        'candidate_occupied_cells': int(np.count_nonzero(candidate_occupied)),
        'occupied_precision': _ratio(true_positive, true_positive + false_positive),
        'occupied_recall': _ratio(true_positive, true_positive + false_negative),
        'occupied_iou': _ratio(true_positive, union),
        # The same argument the docstring makes for the reference applies to
        # the candidate. A cell vision never observed is not a cell it got
        # wrong, and counting it as a miss measures where the cameras were
        # pointed rather than what the mapper did with what it saw. Reported
        # beside the plain figure rather than replacing it, because on its own
        # it is trivially gamed: a mapper that observes nothing scores 1.000.
        # Read it with explored_fraction, which says how much it looked at.
        'occupied_recall_observed': _ratio(
            true_positive,
            int(np.count_nonzero(reference_occupied & ~candidate_unknown)),
        ),
        'free_recall': _ratio(
            int(np.count_nonzero(reference_free & candidate_free)),
            int(np.count_nonzero(reference_free)),
        ),
        'explored_fraction': _ratio(
            int(np.count_nonzero(considered & ~candidate_unknown)),
            int(np.count_nonzero(considered)),
        ),
    }


class ReferenceAccumulator:
    """Fuses the travelling reference grid into one map-frame grid.

    A single snapshot scores whatever happens to be under the vehicle at that
    instant, which swings wildly. Accumulating answers the question that
    matters instead: of everything the LiDAR saw near the path, how much did
    the cameras recover?
    """

    def __init__(self, template: Grid, max_range_m: float = 0.0):
        self.resolution = template.resolution
        self.origin_x = template.origin_x
        self.origin_y = template.origin_y
        self.max_range = max_range_m
        self.occupied = np.zeros(template.values.shape, dtype=bool)
        self.free = np.zeros(template.values.shape, dtype=bool)

    def add(self, grid: Grid, viewpoint=None) -> None:
        x, y = grid.cell_centres()
        known = grid.values != UNKNOWN
        if self.max_range > 0.0 and viewpoint is not None:
            known &= np.hypot(x - viewpoint[0], y - viewpoint[1]) <= self.max_range
        columns = np.floor((x - self.origin_x) / self.resolution).astype(np.int64)
        rows = np.floor((y - self.origin_y) / self.resolution).astype(np.int64)
        inside = (
            known
            & (columns >= 0)
            & (columns < self.occupied.shape[1])
            & (rows >= 0)
            & (rows < self.occupied.shape[0])
        )
        occupied = inside & (grid.values >= OCCUPIED_THRESHOLD)
        free = inside & (grid.values <= FREE_THRESHOLD)
        self.occupied[rows[occupied], columns[occupied]] = True
        self.free[rows[free], columns[free]] = True

    def as_grid(self) -> Grid:
        values = np.full(self.occupied.shape, UNKNOWN, dtype=np.int16)
        values[self.free] = 0
        # An obstacle seen once outranks free space seen many times.
        values[self.occupied] = 100
        return Grid(values, self.resolution, self.origin_x, self.origin_y)


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator > 0 else float('nan')


def grid_from_message(msg) -> Grid:
    """Build a Grid from nav_msgs/OccupancyGrid."""
    values = np.asarray(msg.data, dtype=np.int16).reshape(
        msg.info.height, msg.info.width
    )
    orientation = msg.info.origin.orientation
    yaw = math.atan2(
        2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
        1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z),
    )
    return Grid(
        values,
        msg.info.resolution,
        msg.info.origin.position.x,
        msg.info.origin.position.y,
        yaw,
    )


def format_grid_summary(summary: Dict[str, float]) -> str:
    if not summary.get('compared_cells'):
        return 'no overlapping cells yet'
    return (
        'grid: cells={compared_cells} ref_occ={reference_occupied_cells} '
        'vis_occ={candidate_occupied_cells} precision={occupied_precision:.3f} '
        'recall={occupied_recall:.3f} iou={occupied_iou:.3f} '
        'free_recall={free_recall:.3f} explored={explored_fraction:.3f}'
    ).format(**summary)
