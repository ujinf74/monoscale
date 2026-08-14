import math

import numpy as np
import pytest

from monoscale_odometry.inertial import agrees_with_prediction
from monoscale_odometry.inertial import PlanarInertialPropagator

LEVEL = (0.0, 0.0, 0.0, 1.0)
GRAVITY = 9.81


def _feed(propagator, duration, accel_x, rate=100.0, start=0.0, orientation=LEVEL):
    steps = int(duration * rate)
    for index in range(steps + 1):
        stamp = start + index / rate
        propagator.add_sample(stamp, orientation, (accel_x, 0.0, GRAVITY))
    return start + steps / rate


def test_level_and_still_produces_no_motion():
    propagator = PlanarInertialPropagator()

    _feed(propagator, 1.0, 0.0)

    assert propagator.velocity == pytest.approx([0.0, 0.0], abs=1e-9)
    assert propagator.position == pytest.approx([0.0, 0.0], abs=1e-9)


def test_constant_acceleration_integrates_to_the_textbook_answer():
    propagator = PlanarInertialPropagator()
    # Start from a known standstill, as vision would report it.
    propagator.correct_velocity([0.0, 0.0])

    end = _feed(propagator, 2.0, 1.5)

    # v = at, s = at^2/2
    assert propagator.velocity[0] == pytest.approx(3.0, abs=1e-6)
    assert propagator.position[0] == pytest.approx(3.0, rel=1e-3)
    assert propagator.predicted_translation(0.0, end, 0.0)[0] == pytest.approx(
        3.0, rel=1e-3
    )


def test_prediction_is_expressed_in_the_body_frame():
    propagator = PlanarInertialPropagator()
    propagator.correct_velocity([0.0, 0.0])
    end = _feed(propagator, 1.0, 2.0)

    forward = propagator.predicted_translation(0.0, end, 0.0)
    sideways = propagator.predicted_translation(0.0, end, math.pi / 2.0)

    assert forward[0] == pytest.approx(1.0, rel=1e-3)
    assert forward[1] == pytest.approx(0.0, abs=1e-9)
    # Facing +y, the same world motion is now to the vehicle's right.
    assert sideways[0] == pytest.approx(0.0, abs=1e-6)
    assert sideways[1] == pytest.approx(-1.0, rel=1e-3)


def test_gravity_is_removed_using_the_reported_orientation():
    # Rolled 90 degrees: gravity now reads along the body y axis.
    roll = math.pi / 2.0
    orientation = (math.sin(roll / 2.0), 0.0, 0.0, math.cos(roll / 2.0))
    propagator = PlanarInertialPropagator()

    for index in range(101):
        propagator.add_sample(index / 100.0, orientation, (0.0, -GRAVITY, 0.0))

    assert propagator.velocity == pytest.approx([0.0, 0.0], abs=1e-6)


def test_vision_correction_replaces_the_drifted_velocity():
    propagator = PlanarInertialPropagator()
    propagator.correct_velocity([0.0, 0.0])
    _feed(propagator, 1.0, 1.0)
    assert propagator.velocity[0] == pytest.approx(1.0, abs=1e-6)

    propagator.correct_velocity([0.2, 0.0])

    assert propagator.velocity[0] == pytest.approx(0.2, abs=1e-9)


def test_a_long_gap_resets_the_velocity_rather_than_extrapolating():
    propagator = PlanarInertialPropagator(max_gap_sec=0.1)
    propagator.correct_velocity([0.0, 0.0])
    _feed(propagator, 1.0, 2.0)

    propagator.add_sample(5.0, LEVEL, (0.0, 0.0, GRAVITY))

    assert propagator.velocity == pytest.approx([0.0, 0.0], abs=1e-9)


def test_gate_accepts_a_reasonable_vision_answer():
    assert agrees_with_prediction(
        np.array([0.42, 0.01]), np.array([0.43, 0.0]), 0.05, 0.3
    )


def test_gate_rejects_the_collapse_to_near_zero():
    # The measured failure: vision says the vehicle barely moved while the
    # accelerometer says it covered 0.43 m.
    assert not agrees_with_prediction(
        np.array([0.05, 0.0]), np.array([0.43, 0.0]), 0.05, 0.3
    )


def test_gate_passes_everything_without_a_prediction():
    assert agrees_with_prediction(np.array([9.0, 9.0]), None, 0.05, 0.3)


def test_gate_stays_out_of_the_way_when_the_prediction_is_tiny():
    # Constant velocity means no horizontal acceleration, so a propagator that
    # has never been corrected predicts nothing. It must not veto vision.
    assert agrees_with_prediction(
        np.array([0.12, 0.0]), np.array([0.0, 0.0]), 0.08, 0.35
    )


def test_gate_still_catches_the_collapse_once_the_prediction_is_real():
    assert not agrees_with_prediction(
        np.array([0.05, 0.0]), np.array([0.43, 0.0]), 0.08, 0.35
    )


def test_prediction_is_withheld_until_vision_anchors_the_velocity():
    propagator = PlanarInertialPropagator()
    # A spawn drop, integrated before anything has been corrected.
    _feed(propagator, 0.5, 4.0)

    assert propagator.predicted_translation(0.0, 0.5, 0.0) is None

    propagator.correct_velocity([0.0, 0.0])
    _feed(propagator, 0.5, 1.0, start=0.5)

    assert propagator.predicted_translation(0.5, 1.0, 0.0) is not None


def test_add_sample_reports_the_acceleration_a_filter_should_propagate_on():
    propagator = PlanarInertialPropagator(
        median_window=1, integration_method='zoh'
    )
    propagator.correct_velocity([0.0, 0.0])
    propagator.add_sample(0.0, LEVEL, (0.0, 0.0, GRAVITY))

    acceleration, step = propagator.add_sample(0.01, LEVEL, (2.0, 0.0, GRAVITY))

    assert acceleration == pytest.approx([2.0, 0.0], abs=1e-9)
    assert step == pytest.approx(0.01)


def test_rk4_reports_the_interval_average_to_the_velocity_filter():
    propagator = PlanarInertialPropagator(
        max_gap_sec=2.0, median_window=1, integration_method='rk4'
    )
    propagator.correct_velocity([0.0, 0.0])
    propagator.add_sample(0.0, LEVEL, (0.0, 0.0, GRAVITY))

    acceleration, step = propagator.add_sample(
        1.0, LEVEL, (2.0, 0.0, GRAVITY)
    )

    assert acceleration == pytest.approx([1.0, 0.0], abs=1e-9)
    assert step == pytest.approx(1.0)


def test_spawn_acceleration_is_not_integrated_before_vision_anchor():
    propagator = PlanarInertialPropagator(median_window=1)

    _feed(propagator, 0.5, 8.0)

    assert propagator.velocity == pytest.approx([0.0, 0.0], abs=1e-9)
    assert propagator.position == pytest.approx([0.0, 0.0], abs=1e-9)


def test_physics_impulse_is_rejected_without_poisoning_velocity():
    propagator = PlanarInertialPropagator(
        max_horizontal_acceleration=12.0, median_window=3
    )
    propagator.correct_velocity([1.0, 0.0])
    propagator.add_sample(0.0, LEVEL, (0.0, 0.0, GRAVITY))
    acceleration, _ = propagator.add_sample(
        0.01, LEVEL, (900.0, 20.0, GRAVITY)
    )

    assert acceleration == pytest.approx([0.0, 0.0], abs=1e-9)
    assert propagator.velocity == pytest.approx([1.0, 0.0], abs=1e-9)
    assert propagator.rejected_samples == 1


def test_rk4_integrates_linearly_changing_acceleration():
    propagator = PlanarInertialPropagator(
        max_gap_sec=2.0, median_window=1, integration_method='rk4'
    )
    propagator.correct_velocity([0.0, 0.0])
    propagator.add_sample(0.0, LEVEL, (0.0, 0.0, GRAVITY))
    propagator.add_sample(1.0, LEVEL, (2.0, 0.0, GRAVITY))

    # a(t)=2t: v(1)=1 and x(1)=1/3.
    assert propagator.velocity[0] == pytest.approx(1.0, abs=1e-9)
    assert propagator.position[0] == pytest.approx(1.0 / 3.0, abs=1e-9)
