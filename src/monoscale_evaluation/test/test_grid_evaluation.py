import math

import numpy as np
import pytest

from monoscale_evaluation.grid_evaluation import compare_grids
from monoscale_evaluation.grid_evaluation import Grid
from monoscale_evaluation.grid_evaluation import UNKNOWN


def _grid(values, resolution=1.0, origin_x=0.0, origin_y=0.0, yaw=0.0):
    return Grid(np.asarray(values, dtype=np.int16), resolution, origin_x, origin_y, yaw)


def test_sampling_maps_world_points_onto_cells():
    grid = _grid([[0, 100], [50, UNKNOWN]], resolution=2.0, origin_x=10.0, origin_y=-4.0)

    values = grid.sample(np.array([11.0, 13.0, 11.0, 13.0]), np.array([-3.0, -3.0, -1.0, -1.0]))

    assert values.tolist() == [0, 100, 50, UNKNOWN]


def test_sampling_outside_the_grid_is_unknown():
    grid = _grid([[0, 100]], resolution=1.0)

    assert grid.sample(np.array([-5.0, 50.0]), np.array([0.0, 0.0])).tolist() == [
        UNKNOWN,
        UNKNOWN,
    ]


def test_rotated_origin_is_respected():
    grid = _grid([[0, 100]], resolution=1.0, yaw=math.pi / 2.0)

    # The grid's +x axis now points along world +y.
    assert grid.sample(np.array([-0.5]), np.array([1.5])).tolist() == [100]


def test_identical_grids_score_perfectly():
    values = [[0, 100, 0], [0, 100, 0], [0, 0, 0]]
    summary = compare_grids(_grid(values), _grid(values))

    assert summary['occupied_precision'] == pytest.approx(1.0)
    assert summary['occupied_recall'] == pytest.approx(1.0)
    assert summary['occupied_iou'] == pytest.approx(1.0)
    assert summary['free_recall'] == pytest.approx(1.0)
    assert summary['explored_fraction'] == pytest.approx(1.0)


def test_missed_obstacles_lower_recall_but_not_precision():
    reference = _grid([[100, 100], [0, 0]])
    candidate = _grid([[100, 0], [0, 0]])

    summary = compare_grids(candidate, reference)

    assert summary['occupied_recall'] == pytest.approx(0.5)
    assert summary['occupied_precision'] == pytest.approx(1.0)
    assert summary['occupied_iou'] == pytest.approx(0.5)


def test_reference_unknown_cells_are_not_scored():
    reference = _grid([[UNKNOWN, 100], [UNKNOWN, 0]])
    candidate = _grid([[100, 100], [100, 0]])

    summary = compare_grids(candidate, reference)

    assert summary['compared_cells'] == 2
    # The two candidate obstacles in the unknown column must not count against it.
    assert summary['occupied_precision'] == pytest.approx(1.0)


def test_unexplored_candidate_shows_up_as_low_explored_fraction():
    reference = _grid([[100, 0], [0, 0]])
    candidate = _grid([[UNKNOWN, UNKNOWN], [UNKNOWN, UNKNOWN]])

    summary = compare_grids(candidate, reference)

    assert summary['explored_fraction'] == pytest.approx(0.0)
    assert summary['occupied_recall'] == pytest.approx(0.0)


def test_range_limit_excludes_distant_cells():
    reference = _grid([[100, 100, 100, 100]], resolution=1.0)
    candidate = _grid([[100, 100, 0, 0]], resolution=1.0)

    summary = compare_grids(
        candidate, reference, max_range_m=2.0, reference_origin=(0.0, 0.5)
    )

    assert summary['compared_cells'] == 2
    assert summary['occupied_recall'] == pytest.approx(1.0)


def test_differing_resolutions_are_resampled():
    reference = _grid([[100, 100], [0, 0]], resolution=1.0)
    candidate = _grid(
        [[100, 100, 100, 100], [100, 100, 100, 100], [0, 0, 0, 0], [0, 0, 0, 0]],
        resolution=0.5,
    )

    summary = compare_grids(candidate, reference)

    assert summary['occupied_iou'] == pytest.approx(1.0)
    assert summary['free_recall'] == pytest.approx(1.0)


def test_accumulator_keeps_everything_the_reference_ever_saw():
    from monoscale_evaluation.grid_evaluation import ReferenceAccumulator

    template = _grid([[UNKNOWN] * 4] * 4, resolution=1.0)
    accumulator = ReferenceAccumulator(template)

    accumulator.add(_grid([[100, 0]], resolution=1.0, origin_x=0.0, origin_y=0.0))
    accumulator.add(_grid([[0, 100]], resolution=1.0, origin_x=0.0, origin_y=2.0))

    values = accumulator.as_grid().values
    assert values[0].tolist() == [100, 0, UNKNOWN, UNKNOWN]
    assert values[2].tolist() == [0, 100, UNKNOWN, UNKNOWN]
    assert (values[1] == UNKNOWN).all()


def test_accumulated_obstacle_outranks_later_free_space():
    from monoscale_evaluation.grid_evaluation import ReferenceAccumulator

    accumulator = ReferenceAccumulator(_grid([[UNKNOWN, UNKNOWN]], resolution=1.0))
    accumulator.add(_grid([[100, UNKNOWN]], resolution=1.0))
    accumulator.add(_grid([[0, 0]], resolution=1.0))

    assert accumulator.as_grid().values[0].tolist() == [100, 0]


def test_accumulator_range_limit_drops_far_cells():
    from monoscale_evaluation.grid_evaluation import ReferenceAccumulator

    accumulator = ReferenceAccumulator(
        _grid([[UNKNOWN] * 4], resolution=1.0), max_range_m=2.0
    )

    accumulator.add(_grid([[100, 100, 100, 100]], resolution=1.0), viewpoint=(0.0, 0.5))

    assert accumulator.as_grid().values[0].tolist() == [100, 100, UNKNOWN, UNKNOWN]
