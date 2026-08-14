"""What the heading filter has to do that a gain could not.

Two things were tried before this and measured on real drives. A hard bound
on how far the ground solve may move the heading recovered a drifting gyro at
8 m/s and ruined a 2.5 m/s drive, because a push of up to the bound on every
solve is a random walk and a slower vehicle takes more solves per metre.
Weighting the share by precision -- the ordinary scalar gain -- was no better
anywhere: 0.172 m on the drifting drive against 0.169 for believing the gyro
outright, and worse on the two undrifted ones.

The reason both failed is the same. A bias is not noise. Inflate its variance
and pay that off against a measurement and the next interval earns it straight
back. So the property to test for is not that the heading gets corrected, but
that the rate behind it is learned and stops coming back.
"""

import math

import numpy as np
import pytest

from monoscale_odometry.odometry_node import HeadingBiasFilter


def run(filter_, bias, steps, dt=0.02, sigma=0.01, seed=0):
    """Drive the filter with a heading that drifts at `bias` radians a second.

    Returns how far the heading was out at the end, after every correction the
    filter asked for has been applied to it.
    """
    rng = np.random.default_rng(seed)
    error = 0.0
    for _ in range(steps):
        error += bias * dt
        filter_.predict(dt)
        # What the ground says the heading should have been, seen through a
        # solve of finite precision.
        measured = -error + rng.normal(scale=sigma)
        error += filter_.update(measured, sigma)
    return error


def test_disabled_by_default_the_heading_is_left_alone():
    quiet = HeadingBiasFilter(0.0, 1e-5, 1e-3)

    assert not quiet.enabled
    assert quiet.update(0.5, 0.01) == 0.0
    assert run(quiet, 0.005, 200) == pytest.approx(0.005 * 0.02 * 200)


def test_a_constant_bias_is_learned_and_stops_accumulating():
    bias = 0.005
    filter_ = HeadingBiasFilter(bias_sigma=0.01, walk_sigma=1e-6, noise_sigma=1e-4)

    left = run(filter_, bias, 600)

    # The rate it settled on is the one that was there.
    assert filter_.rate == pytest.approx(-bias, rel=0.2)
    # And what the heading has left over is a fraction of a milliradian,
    # against the 60 mrad it would have accumulated untouched.
    assert abs(left) < 0.002


def test_learning_the_rate_is_what_makes_the_difference():
    """The same filter, denied its second state, cannot keep up.

    Pinning the rate to zero leaves something that can only chase the error it
    can already see, which is the shape both earlier attempts had.
    """
    bias = 0.005
    full = HeadingBiasFilter(0.01, 1e-6, 1e-4)
    rateless = HeadingBiasFilter(0.01, 1e-6, 1e-4)
    rateless.covariance[1, 1] = 0.0

    with_rate = abs(run(full, bias, 600))
    without = abs(run(rateless, bias, 600))

    assert with_rate < 0.25 * without


def test_a_quiet_instrument_is_not_talked_out_of_its_heading():
    """Nothing to find, so nothing should be done.

    This is the case the simulator provides and a vehicle never will, and the
    filter costing accuracy here is the price of it working elsewhere -- but
    the price has to stay small.
    """
    filter_ = HeadingBiasFilter(0.01, 1e-6, 1e-4)

    left = abs(run(filter_, 0.0, 600, sigma=0.01, seed=3))

    assert left < 0.002
    assert abs(filter_.rate) < 0.001


def test_the_covariance_stays_symmetric_and_positive():
    filter_ = HeadingBiasFilter(0.01, 1e-6, 1e-4)

    run(filter_, 0.005, 300)

    assert np.allclose(filter_.covariance, filter_.covariance.T)
    assert filter_.covariance[0, 0] > 0.0
    assert filter_.covariance[1, 1] > 0.0
    assert np.linalg.det(filter_.covariance) > 0.0
    assert all(math.isfinite(v) for v in filter_.covariance.reshape(-1))
