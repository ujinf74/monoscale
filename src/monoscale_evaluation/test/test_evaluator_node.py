"""Node-level checks for pose bookkeeping and the bridge clock workaround."""

import math

import pytest
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.parameter import Parameter

from monoscale_evaluation.odometry_evaluator import OdometryEvaluator

WHEEL_BASE = 2.7


@pytest.fixture
def evaluator_factory():
    rclpy.init()
    created = []

    def build(**overrides):
        parameters = [
            Parameter('wheel_base', Parameter.Type.DOUBLE, WHEEL_BASE),
            Parameter('report_period_sec', Parameter.Type.DOUBLE, 1000.0),
        ]
        parameters += [
            Parameter(name, value=value) for name, value in overrides.items()
        ]
        node = OdometryEvaluator(parameter_overrides=parameters)
        created.append(node)
        return node

    yield build
    for node in created:
        node.destroy_node()
    rclpy.shutdown()


def _stamp(msg, stamp):
    msg.header.stamp.sec = int(stamp)
    msg.header.stamp.nanosec = int(round((stamp - int(stamp)) * 1e9))


def _estimate(stamp, x, y=0.0, yaw=0.0):
    msg = Odometry()
    _stamp(msg, stamp)
    msg.pose.pose.position.x = x
    msg.pose.pose.position.y = y
    msg.pose.pose.orientation.z = math.sin(0.5 * yaw)
    msg.pose.pose.orientation.w = math.cos(0.5 * yaw)
    return msg


def _truth(stamp, x, y=0.0, yaw=0.0):
    msg = PoseWithCovarianceStamped()
    _stamp(msg, stamp)
    msg.pose.pose.position.x = x
    msg.pose.pose.position.y = y
    msg.pose.pose.orientation.z = math.sin(0.5 * yaw)
    msg.pose.pose.orientation.w = math.cos(0.5 * yaw)
    return msg


def _feed(node, count, truth_offset=0.0, estimate_scale=1.0, truth_shift=0.0):
    for step in range(count):
        # The simulator clock is well ahead of the bridge process clock, which
        # is what the mismatch cases below exploit.
        stamp = 500.0 + 0.05 * step
        distance = 0.1 * step
        node._on_estimate(_estimate(stamp, distance * estimate_scale))
        # The truth reports the vehicle centre, half a wheelbase ahead.
        node._on_truth(
            _truth(stamp + truth_offset, distance + 0.5 * WHEEL_BASE + truth_shift)
        )


def test_vehicle_center_offset_is_removed_before_scoring(evaluator_factory):
    node = evaluator_factory()

    _feed(node, 40)

    summary = node.metrics.summary()
    assert summary['samples'] > 30
    assert summary['position_rmse_m'] == pytest.approx(0.0, abs=1e-6)


def test_scale_error_survives_the_frame_bookkeeping(evaluator_factory):
    node = evaluator_factory()

    _feed(node, 40, estimate_scale=1.05)

    summary = node.metrics.summary()
    assert summary['scale_ratio'] == pytest.approx(1.05, rel=1e-3)


def test_mismatched_bridge_clocks_are_latched_and_removed(evaluator_factory):
    node = evaluator_factory()

    _feed(node, 40, truth_offset=-409.0)

    assert node.clock_offset == pytest.approx(409.0, abs=1e-6)
    summary = node.metrics.summary()
    assert summary['samples'] > 30
    assert summary['position_rmse_m'] == pytest.approx(0.0, abs=1e-6)


def test_offset_is_not_latched_when_disabled(evaluator_factory):
    node = evaluator_factory(clock_offset_mode='none')

    _feed(node, 40, truth_offset=-409.0)

    assert node.clock_offset == 0.0
    assert node.metrics.summary().get('samples', 0) == 0


def test_small_transport_delay_is_not_treated_as_a_clock_offset(evaluator_factory):
    node = evaluator_factory()

    _feed(node, 40, truth_offset=-0.01)

    assert node.clock_offset == 0.0
    assert node.metrics.summary()['samples'] > 30


def test_scoring_stops_once_the_window_distance_is_reached(evaluator_factory):
    node = evaluator_factory(evaluation_distance_m=2.0)

    # 100 samples covering 9.9 m, well past the 2 m window.
    _feed(node, 100)
    summary = node.metrics.summary()

    assert node.window_closed
    assert summary['truth_distance_m'] == pytest.approx(2.0, abs=0.11)


def test_an_unset_window_scores_the_whole_run(evaluator_factory):
    node = evaluator_factory()

    _feed(node, 100)

    assert not node.window_closed
    assert node.metrics.summary()['truth_distance_m'] == pytest.approx(9.9, abs=0.11)


def test_a_standing_still_tail_cannot_reach_past_the_window(evaluator_factory):
    """The tail that moved the number: error grows while distance does not."""
    bounded = evaluator_factory(evaluation_distance_m=2.0)
    unbounded = evaluator_factory()
    for node in (bounded, unbounded):
        _feed(node, 30)
        # Parked, with the estimate quietly walking away from the truth.
        for step in range(200):
            stamp = 500.0 + 0.05 * (30 + step)
            node._on_estimate(_estimate(stamp, 2.9 + 0.002 * step))
            node._on_truth(_truth(stamp, 2.9 + 0.5 * WHEEL_BASE))

    assert bounded.metrics.summary()['drift_percent'] < (
        unbounded.metrics.summary()['drift_percent']
    )
