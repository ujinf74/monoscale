"""The C++ port has to answer exactly what the numpy did.

This is the whole safety net for the port. Both implementations stay in the
tree and the drive is scored with either, so any difference between them would
appear as a difference in the score with nothing to distinguish it from a real
change. The functions are small and deterministic, so agreement to a few
floating point ulps is the right bar, not a loose tolerance.

Skipped rather than failed when the extension is absent: the Python path is
the one that has to keep working everywhere, and a machine without a compiler
should still be able to run the tests.
"""

import math

import numpy as np
import pytest

from monoscale_odometry.geometry import CameraModel, pixels_to_ground
from monoscale_odometry.tracks import align_to_anchors

fast = pytest.importorskip('monoscale_fast')


def a_camera():
    """The front camera, as configured: 30 degrees down, 0.93 m up."""
    pitch = math.radians(-30.0)
    rotation = np.array([
        [0.0, -math.sin(-pitch), math.cos(-pitch)],
        [-1.0, 0.0, 0.0],
        [0.0, -math.cos(-pitch), -math.sin(-pitch)],
    ])
    return CameraModel(
        rotation_base_from_camera=rotation,
        translation_base_from_camera=np.array([3.5, 0.0, 0.9272]),
        k=np.array([[450.0, 0.0, 480.0], [0.0, 450.0, 270.0], [0.0, 0.0, 1.0]]),
    )


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_the_ground_projection_agrees(seed):
    rng = np.random.default_rng(seed)
    model = a_camera()
    pixels = np.column_stack((
        rng.uniform(0.0, 960.0, 400),
        rng.uniform(0.0, 540.0, 400),
    ))

    expected, expected_valid = pixels_to_ground(pixels, model, 8.0, 0.6)
    got, got_valid = fast.pixels_to_ground(
        pixels, np.linalg.inv(model.k), model.rotation_base_from_camera,
        model.translation_base_from_camera, 8.0, 0.6, None)

    assert np.array_equal(expected_valid, got_valid)
    # Only the usable ones carry a defined position; the rest are whatever
    # fell out of a division the caller was told to ignore.
    assert got[expected_valid] == pytest.approx(expected[expected_valid], abs=1e-9)


def test_the_ground_projection_agrees_under_tilt():
    rng = np.random.default_rng(7)
    model = a_camera()
    pixels = np.column_stack((
        rng.uniform(0.0, 960.0, 300),
        rng.uniform(0.0, 540.0, 300),
    ))
    roll, pitch = math.radians(2.0), math.radians(-1.0)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    tilt = np.array([
        [cp, sp * sr, sp * cr],
        [0.0, cr, -sr],
        [-sp, cp * sr, cp * cr],
    ])

    expected, expected_valid = pixels_to_ground(pixels, model, 8.0, 0.6, tilt)
    got, got_valid = fast.pixels_to_ground(
        pixels, np.linalg.inv(model.k), model.rotation_base_from_camera,
        model.translation_base_from_camera, 8.0, 0.6, tilt)

    assert np.array_equal(expected_valid, got_valid)
    assert got[expected_valid] == pytest.approx(expected[expected_valid], abs=1e-9)


def a_drive(rng, count, yaw, translation, noise, outliers=0):
    body = np.column_stack((
        rng.uniform(0.6, 8.0, count),
        rng.uniform(-4.0, 4.0, count),
    ))
    c, s = math.cos(yaw), math.sin(yaw)
    world = (np.array([[c, -s], [s, c]]) @ body.T).T + translation
    world += rng.normal(scale=noise, size=world.shape)
    if outliers:
        which = rng.choice(count, size=outliers, replace=False)
        world[which] += rng.normal(scale=2.0, size=(outliers, 2))
    return body, world


@pytest.mark.parametrize('seed', [0, 1, 2, 3])
@pytest.mark.parametrize('refine', [False, True])
def test_the_anchor_alignment_agrees(seed, refine):
    rng = np.random.default_rng(seed)
    body, world = a_drive(
        rng, 400, yaw=0.01, translation=np.array([2.0, -0.5]),
        noise=0.02, outliers=40)
    weights = rng.uniform(1.0, 20.0, len(body))

    expected = align_to_anchors(body, world, weights, 0.0, 0.1, 20, refine)
    got = fast.align_to_anchors(body, world, weights, 0.0, 0.1, 20, refine)

    assert (expected is None) == (got is None)
    if expected is None:
        return
    assert got[0] == pytest.approx(expected[0], abs=1e-9)
    assert np.array_equal(got[1], expected[1])
    assert got[2] == pytest.approx(expected[2], abs=1e-12)
    assert got[3] == pytest.approx(expected[3], abs=1e-12)
    assert got[4] == pytest.approx(expected[4], abs=1e-12)


def test_both_give_up_on_the_same_hopeless_case():
    rng = np.random.default_rng(11)
    body, world = a_drive(rng, 30, yaw=0.0, translation=np.array([0.0, 0.0]),
                          noise=5.0)

    assert align_to_anchors(body, world, np.ones(len(body)), 0.0, 0.01, 25) is None
    assert fast.align_to_anchors(body, world, np.ones(len(body)), 0.0, 0.01, 25) is None


def test_an_even_inlier_count_takes_the_same_median():
    """numpy's median averages the middle pair; a nth_element does not.

    The initial centre is a median, and getting it wrong by half a gap moves
    which points survive the first reweighting pass, so the two paths diverge
    from there on a case that looks unremarkable.
    """
    rng = np.random.default_rng(5)
    body, world = a_drive(rng, 200, yaw=0.0, translation=np.array([1.0, 1.0]),
                          noise=0.05)

    expected = align_to_anchors(body, world, np.ones(len(body)), 0.0, 0.2, 10)
    got = fast.align_to_anchors(body, world, np.ones(len(body)), 0.0, 0.2, 10)

    assert got[0] == pytest.approx(expected[0], abs=1e-9)
    assert np.array_equal(got[1], expected[1])
