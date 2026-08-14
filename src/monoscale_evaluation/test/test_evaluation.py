import math

import pytest

from monoscale_evaluation.evaluation import compose_pose
from monoscale_evaluation.evaluation import interpolate_pose
from monoscale_evaluation.evaluation import Pose2
from monoscale_evaluation.evaluation import relative_pose
from monoscale_evaluation.evaluation import shift_along_heading
from monoscale_evaluation.evaluation import TrajectoryMetrics


def test_interpolation_takes_the_short_way_around_pi():
    before = Pose2(0.0, 0.0, 0.0, 3.0)
    after = Pose2(1.0, 2.0, 4.0, -3.0)

    middle = interpolate_pose(before, after, 0.5)

    assert middle.x == pytest.approx(1.0)
    assert middle.y == pytest.approx(2.0)
    assert abs(middle.yaw) == pytest.approx(math.pi, abs=1e-6)


def test_relative_and_compose_are_inverse():
    origin = Pose2(0.0, 3.0, -1.0, 0.7)
    pose = Pose2(0.0, 5.0, 2.0, -0.4)

    delta = relative_pose(origin, pose)
    restored = compose_pose((origin.x, origin.y, origin.yaw), delta)

    assert restored[0] == pytest.approx(pose.x)
    assert restored[1] == pytest.approx(pose.y)
    assert restored[2] == pytest.approx(pose.yaw)


def test_vehicle_center_shifts_back_to_the_rear_axle():
    x, y = shift_along_heading(10.0, 5.0, math.pi / 2.0, -1.35)

    assert x == pytest.approx(10.0)
    assert y == pytest.approx(3.65)


def test_perfect_estimate_scores_zero_error():
    metrics = TrajectoryMetrics(rpe_delta_m=1.0)
    for step in range(51):
        pose = (0.2 * step, 0.0, 0.0)
        metrics.add(0.1 * step, pose, pose)

    summary = metrics.summary()

    assert summary['samples'] == 51
    assert summary['truth_distance_m'] == pytest.approx(10.0)
    assert summary['scale_ratio'] == pytest.approx(1.0)
    assert summary['position_rmse_m'] == pytest.approx(0.0)
    assert summary['relative_translation_percent'] == pytest.approx(0.0)
    assert summary['relative_segments'] == 10


def test_scale_error_shows_up_as_drift_and_relative_error():
    metrics = TrajectoryMetrics(rpe_delta_m=1.0)
    for step in range(51):
        truth = (0.2 * step, 0.0, 0.0)
        estimate = (0.2 * step * 1.1, 0.0, 0.0)
        metrics.add(0.1 * step, estimate, truth)

    summary = metrics.summary()

    assert summary['scale_ratio'] == pytest.approx(1.1)
    assert summary['final_position_error_m'] == pytest.approx(1.0)
    assert summary['drift_percent'] == pytest.approx(10.0)
    assert summary['relative_translation_percent'] == pytest.approx(10.0, rel=1e-3)


def test_heading_error_is_reported_in_degrees():
    metrics = TrajectoryMetrics(rpe_delta_m=1.0)
    for step in range(11):
        truth = (0.5 * step, 0.0, 0.0)
        metrics.add(0.1 * step, (0.5 * step, 0.0, math.radians(2.0)), truth)

    summary = metrics.summary()

    assert summary['yaw_rmse_deg'] == pytest.approx(2.0)
    assert summary['yaw_max_deg'] == pytest.approx(2.0)
