"""Rescaling a hop has to follow the arc the vehicle is actually on.

Two places stretch a measured motion to a different interval: the pair, whose
two cameras are stamped a little apart, and the coast that carries the pose
across a rejected solve. Both used to multiply the translation and be done
with it, which is right only while the wheel is centred. A vehicle turning at
a steady rate travels an arc and its heading turns as it goes, so the
direction of travel at the end is not the direction at the start.
"""

import math

import numpy as np
import pytest

from monoscale_odometry.geometry import (
    PlanarMotion, Pose2, motion_from_twist, rescale_motion, twist_from_motion)


def test_a_straight_hop_is_just_stretched():
    straight = PlanarMotion(0.4, 0.0, 0.0, 100, 1.0)

    doubled = rescale_motion(straight, 2.0)

    assert doubled.x == pytest.approx(0.8)
    assert doubled.y == pytest.approx(0.0)
    assert doubled.yaw == pytest.approx(0.0)


def test_the_twist_round_trips():
    for motion in (
        PlanarMotion(0.13, -0.02, 0.05, 0, 1.0),
        PlanarMotion(0.4, 0.0, 0.0, 0, 1.0),
        PlanarMotion(-0.2, 0.03, -0.3, 0, 1.0),
    ):
        vx, vy, omega = twist_from_motion(motion)
        back = motion_from_twist(vx, vy, omega)
        assert back.x == pytest.approx(motion.x, abs=1e-12)
        assert back.y == pytest.approx(motion.y, abs=1e-12)
        assert back.yaw == pytest.approx(motion.yaw, abs=1e-12)


def test_rescaling_composes_the_way_driving_does():
    """Two halves of a turn, driven one after the other, are the whole turn.

    This is the property that makes the arc the right shape and a stretched
    straight line the wrong one: halving and then composing has to land where
    the full hop lands, and it only does if the heading turns along the way.
    """
    whole = PlanarMotion(0.30, 0.0, 0.25, 0, 1.0)

    half = rescale_motion(whole, 0.5)
    composed = Pose2(0.0, 0.0, 0.0).compose(half).compose(half)

    assert composed.x == pytest.approx(whole.x, abs=1e-12)
    assert composed.y == pytest.approx(whole.y, abs=1e-12)
    assert composed.yaw == pytest.approx(whole.yaw, abs=1e-12)


def test_the_straight_line_answer_is_wrong_by_a_measurable_amount():
    """And by how much, so the reason for the change is on the record.

    A tenth of a second at 4 m/s through a 16 deg/s turn, carried across twice
    its own length: stretching the straight line puts the vehicle 11 mm off
    sideways. The pair's two cameras are only a few milliseconds apart and see
    a fraction of that, but a coast across a rejected solve spans exactly this
    sort of interval, and there are tens of them in a drive.
    """
    speed, rate, interval = 4.0, math.radians(16.0), 0.1
    hop = motion_from_twist(speed, 0.0, rate, interval)

    stretched = PlanarMotion(hop.x * 2.0, hop.y * 2.0, hop.yaw * 2.0, 0, 1.0)
    arc = rescale_motion(hop, 2.0)

    sideways = abs(arc.y - stretched.y)
    assert sideways == pytest.approx(0.0112, abs=1e-3)
    # Forward it barely matters: the arc and the chord are the same length to
    # within a part in a thousand at this angle.
    assert abs(arc.x - stretched.x) < 1e-3


def test_rescaling_to_nothing_leaves_nothing():
    motion = PlanarMotion(0.3, 0.1, 0.2, 7, 1.0)

    still = rescale_motion(motion, 0.0)

    assert still.x == pytest.approx(0.0)
    assert still.y == pytest.approx(0.0)
    assert still.yaw == pytest.approx(0.0)


def test_the_inlier_count_survives_the_rescale():
    """It is the weight the fusion uses, and it is not a property of time."""
    motion = PlanarMotion(0.3, 0.1, 0.2, 137, 0.5)

    assert rescale_motion(motion, 1.7).inliers == 137
    assert rescale_motion(motion, 1.7).scale == 0.5


def test_the_attitude_starts_level_rather_than_from_one_reading():
    """A vehicle on a road is level; one accelerometer sample is not evidence.

    Seeding from the first sample of a drive that opens under throttle starts
    the pitch 15 degrees out, and a sixty second trim cannot recover inside a
    thirty second drive. That configuration scored 18.9 m.
    """
    from monoscale_odometry.odometry_node import AttitudeFilter

    filter_ = AttitudeFilter(tau=60.0, tolerance=0.3)
    filter_.update((0.0, 0.0, 0.0), (2.7, -1.3, 9.8), 0.005)

    assert filter_.roll == 0.0
    assert filter_.pitch == 0.0
    assert filter_.started


def test_the_gyro_carries_the_attitude():
    from monoscale_odometry.odometry_node import AttitudeFilter

    filter_ = AttitudeFilter(tau=60.0, tolerance=0.3)
    filter_.update((0.0, 0.0, 0.0), (0.0, 0.0, 9.80665), 0.005)
    # A tenth of a radian a second in roll, for a second, while accelerating
    # hard enough that gravity is not to be believed.
    for _ in range(200):
        filter_.update((0.1, 0.0, 0.0), (3.0, 0.0, 9.80665), 0.005)

    assert filter_.roll == pytest.approx(0.1, rel=0.02)
    assert filter_.corrections == 0


def test_gravity_trims_the_attitude_back_when_it_is_believable():
    from monoscale_odometry.odometry_node import AttitudeFilter

    filter_ = AttitudeFilter(tau=1.0, tolerance=0.3)
    filter_.update((0.0, 0.0, 0.0), (0.0, 0.0, 9.80665), 0.005)
    filter_.roll = 0.05
    # Standing level and still, for two time constants.
    for _ in range(400):
        filter_.update((0.0, 0.0, 0.0), (0.0, 0.0, 9.80665), 0.005)

    assert abs(filter_.roll) < 0.01
    assert filter_.corrections == 400
