import math

import numpy as np
import pytest

from monoscale_odometry.fusion import PlanarVelocityFilter
from monoscale_odometry.fusion import world_velocity_from_motion


def _settled(**kwargs):
    filtered = PlanarVelocityFilter(**kwargs)
    filtered.update(np.array([0.0, 0.0]), 500)
    return filtered


def test_prediction_follows_the_accelerometer():
    filtered = _settled()

    for _ in range(100):
        filtered.predict(np.array([2.0, 0.0]), 0.01)

    assert filtered.velocity[0] == pytest.approx(2.0, abs=1e-9)


def test_uncertainty_grows_while_only_predicting():
    filtered = _settled()
    before = filtered.uncertainty()

    for _ in range(50):
        filtered.predict(np.array([0.0, 0.0]), 0.01)

    assert filtered.uncertainty() > before


def test_a_well_supported_measurement_moves_the_estimate_more():
    strong = PlanarVelocityFilter()
    weak = PlanarVelocityFilter()
    for filtered in (strong, weak):
        filtered.update(np.array([0.0, 0.0]), 300)
        for _ in range(10):
            filtered.predict(np.array([0.0, 0.0]), 0.05)

    # Same small disagreement, different support behind it.
    strong.update(np.array([0.1, 0.0]), 3000)
    weak.update(np.array([0.1, 0.0]), 10)

    assert strong.velocity[0] > weak.velocity[0]
    assert weak.velocity[0] < 0.5 * strong.velocity[0]


def test_measurement_noise_falls_as_inliers_rise():
    filtered = PlanarVelocityFilter(vision_reference_inliers=300.0, vision_noise=0.25)

    assert filtered.measurement_variance(600) < filtered.measurement_variance(300)
    assert filtered.measurement_variance(300) < filtered.measurement_variance(50)


def test_the_collapse_to_zero_is_gated_out_once_the_filter_is_confident():
    filtered = PlanarVelocityFilter(acceleration_noise=0.5)
    # Settle on 6 m/s with well supported vision.
    for _ in range(30):
        filtered.update(np.array([6.0, 0.0]), 600)
        filtered.predict(np.array([0.0, 0.0]), 0.05)

    # The failure we measured: vision claims the vehicle nearly stopped.
    accepted = filtered.update(np.array([0.3, 0.0]), 40)

    assert accepted is False
    assert filtered.velocity[0] == pytest.approx(6.0, abs=0.5)


def test_a_genuine_deceleration_still_gets_through():
    filtered = PlanarVelocityFilter(acceleration_noise=1.5)
    for _ in range(30):
        filtered.update(np.array([2.0, 0.0]), 600)
        filtered.predict(np.array([0.0, 0.0]), 0.05)
    assert filtered.velocity[0] == pytest.approx(2.0, abs=0.1)

    # Braking: the accelerometer reports it and vision agrees frame by frame.
    speed = 2.0
    for _ in range(10):
        filtered.predict(np.array([-4.0, 0.0]), 0.05)
        speed = max(speed - 4.0 * 0.05, 0.0)
        filtered.update(np.array([speed, 0.0]), 600)

    assert filtered.velocity[0] == pytest.approx(speed, abs=0.2)
    assert filtered.velocity[0] < 0.5


def test_nothing_is_gated_before_the_first_update():
    filtered = PlanarVelocityFilter()

    assert filtered.update(np.array([50.0, 0.0]), 5) is True


def test_body_translation_uses_the_fused_velocity():
    filtered = PlanarVelocityFilter()
    # Let the filter converge on 4 m/s along world +y.
    for _ in range(30):
        filtered.predict(np.array([0.0, 0.0]), 0.05)
        filtered.update(np.array([0.0, 4.0]), 600)

    translation = filtered.body_translation(0.5, math.pi / 2.0)

    # Facing +y at 4 m/s, half a second is 2 m straight ahead.
    assert translation[0] == pytest.approx(2.0, rel=0.2)
    assert translation[1] == pytest.approx(0.0, abs=0.2)


def test_motion_to_world_velocity_round_trip():
    velocity = world_velocity_from_motion(0.4, 0.0, 0.1, math.pi / 2.0)

    assert velocity[0] == pytest.approx(0.0, abs=1e-9)
    assert velocity[1] == pytest.approx(4.0)


def test_an_outlier_still_tethers_the_filter_instead_of_being_dropped():
    filtered = PlanarVelocityFilter(acceleration_noise=0.5)
    for _ in range(30):
        filtered.update(np.array([6.0, 0.0]), 600)
        filtered.predict(np.array([0.0, 0.0]), 0.05)

    before = filtered.velocity[0]
    accepted = filtered.update(np.array([0.3, 0.0]), 40)

    # Reported as an outlier, but it still pulls the estimate a little rather
    # than leaving the filter to integrate alone.
    assert accepted is False
    assert filtered.velocity[0] < before
    assert filtered.velocity[0] > 0.8 * before


def test_repeated_outliers_eventually_win_if_the_world_really_changed():
    filtered = PlanarVelocityFilter(acceleration_noise=0.5)
    for _ in range(30):
        filtered.update(np.array([6.0, 0.0]), 600)
        filtered.predict(np.array([0.0, 0.0]), 0.05)

    # About six seconds of consistent disagreement at 20 Hz.
    for _ in range(120):
        filtered.predict(np.array([0.0, 0.0]), 0.05)
        filtered.update(np.array([0.3, 0.0]), 600)

    assert filtered.velocity[0] == pytest.approx(0.3, abs=0.2)


def test_camera_disagreement_dilutes_a_solve_the_inliers_call_confident():
    trusting = PlanarVelocityFilter()
    doubting = PlanarVelocityFilter()
    for filtered in (trusting, doubting):
        filtered.update(np.array([0.0, 0.0]), 300)
        for _ in range(10):
            filtered.predict(np.array([0.0, 0.0]), 0.05)

    # Same measurement, same inlier count. Only the cross-camera evidence
    # differs, which is exactly what inliers cannot reveal.
    trusting.update(np.array([0.5, 0.0]), 800, 0.0)
    doubting.update(np.array([0.5, 0.0]), 800, 4.0)

    assert trusting.velocity[0] > doubting.velocity[0]
    assert doubting.velocity[0] < 0.3 * trusting.velocity[0]
