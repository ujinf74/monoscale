"""Attitude the planar solve cannot see, and the heading bias it earns.

Roll and pitch are carried by the gyro and pulled back towards gravity; yaw is
left alone because the ground registration owns it, and estimating it here as
well would rotate the ground twice.
"""

import math
from typing import Optional

import numpy as np


class AttitudeFilter:
    """Roll and pitch: carried by the gyro, pulled back towards gravity.

    The ground projection assumes the body is level, and one degree of tilt
    moves every range in the frame by two and a half to five per cent. Neither
    instrument can supply the attitude on its own. The gyro integrates cleanly
    but has nothing to hold it to, and the accelerometer measures gravity plus
    whatever the vehicle is doing and cannot tell a braking vehicle from a
    nose-down one.

    Measured against the simulator's own attitude over a drive with hard
    braking and hard steering, weighting these wrongly is far worse than not
    trying: correcting on every sample with a one second constant left pitch
    out by 4.3 degrees against a signal of 0.04. The gyro has to carry it and
    the accelerometer has to be a slow trim on top -- a sixty second constant,
    applied only while the measured acceleration is within 0.3 m/s2 of
    gravity's own -- which lands at 0.046 degrees. Tighter still, 0.02 m/s2
    and twenty seconds, reaches 0.007, and would be the setting to use on an
    instrument quiet enough to pass that gate.
    """

    def __init__(self, tau: float, tolerance: float, gravity: float = 9.80665):
        self.roll = 0.0
        self.pitch = 0.0
        self.started = False
        self.tau = tau
        self.tolerance = tolerance
        self.gravity = gravity
        self.corrections = 0

    def update(self, gyro, accel, dt: float) -> None:
        ax, ay, az = accel
        level_roll = math.atan2(ay, az)
        level_pitch = math.atan2(-ax, math.hypot(ay, az))
        if not self.started:
            # Level, not whatever the first sample happens to say. A vehicle
            # on a road is within a degree or two of level and that is a far
            # better prior than one accelerometer reading: seeding from the
            # first sample of a drive that opens under throttle starts the
            # pitch 15 degrees out, and a sixty second trim never recovers
            # inside a thirty second drive. It scored 18.9 m.
            self.roll, self.pitch = 0.0, 0.0
            self.started = True
            return
        if dt <= 0.0:
            return
        # Body rates resolved through the current attitude. At the angles a
        # road vehicle reaches these are the exact Euler rates and the small
        # angle form would do just as well.
        tan_pitch = math.tan(self.pitch)
        self.roll += dt * (
            gyro[0]
            + gyro[1] * math.sin(self.roll) * tan_pitch
            + gyro[2] * math.cos(self.roll) * tan_pitch)
        self.pitch += dt * (
            gyro[1] * math.cos(self.roll) - gyro[2] * math.sin(self.roll))
        magnitude = math.sqrt(ax * ax + ay * ay + az * az)
        if abs(magnitude - self.gravity) > self.tolerance:
            return
        self.corrections += 1
        gain = min(dt / self.tau, 1.0)
        self.roll += gain * (level_roll - self.roll)
        self.pitch += gain * (level_pitch - self.pitch)


class HeadingBiasFilter:
    """Two states: how far the heading is out, and how fast it is going out.

    The reason this exists rather than a gain. The heading the estimator is
    handed drifts because the instrument behind it has a bias, and a bias is
    not noise: inflate its variance and pay the variance off against a
    measurement, and the next interval simply earns it back. Measured, that
    is exactly what happens -- a precision weighted share of the ground
    solve's own heading left a drifting drive at 0.172 m where believing the
    gyro outright gave 0.169, and made the two undrifted drives worse. The
    bias has to be a state or it cannot be cancelled.

    So: the error and its rate, the error fed by the rate, both corrected by
    what the ground says the heading should have been. The error is injected
    into the pose and reset to zero after every update, which is what makes
    this an error state filter rather than a filter on the heading itself --
    the heading lives in the pose, where the rest of the estimator can see it.
    """

    def __init__(self, bias_sigma: float, walk_sigma: float, noise_sigma: float):
        self.error = 0.0
        self.rate = 0.0
        # No idea which way the bias points, every idea of how big it is.
        self.covariance = np.array(
            [[0.0, 0.0], [0.0, bias_sigma ** 2]], dtype=np.float64)
        self.walk = walk_sigma ** 2
        self.noise = noise_sigma ** 2
        self.enabled = bias_sigma > 0.0

    def predict(self, dt: float):
        if not self.enabled or dt <= 0.0:
            return
        self.error += self.rate * dt
        transition = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float64)
        self.covariance = transition @ self.covariance @ transition.T
        self.covariance[0, 0] += self.noise * dt
        self.covariance[1, 1] += self.walk * dt

    def update(self, residual: float, sigma: float) -> float:
        """Fold in one ground solve's opinion and return the heading offset.

        `residual` is how far that solve wanted to move the heading from
        where it was handed. Returns what to actually move it by, which is
        the error state after the update and before it is reset.
        """
        if not self.enabled or not math.isfinite(sigma) or sigma <= 0.0:
            return 0.0
        innovation = residual - self.error
        variance = self.covariance[0, 0] + sigma * sigma
        if variance <= 0.0:
            return 0.0
        gain = self.covariance[:, 0] / variance
        self.error += gain[0] * innovation
        self.rate += gain[1] * innovation
        self.covariance = (
            np.eye(2) - np.outer(gain, np.array([1.0, 0.0]))) @ self.covariance
        applied = self.error
        # Injected into the pose, so the error state starts again from zero
        # while what it taught us about the rate stays.
        self.error = 0.0
        return applied



def _merge_slip_origins(kept, fresh):
    """One table, sorted by identity so the next step can bisect it."""
    ids = np.concatenate((kept[0], fresh[0]))
    order = np.argsort(ids, kind='stable')
    return tuple(np.concatenate((k, f))[order] for k, f in zip(kept, fresh))
