"""Velocity fused from the accelerometer and from vision.

The previous arrangement was a switch: believe vision, or throw it away and
dead reckon. That discards the thing both sources actually have, which is a
partial and quantifiable opinion about how fast the vehicle is going.

Here the accelerometer propagates the velocity and grows its uncertainty, and
each vision solve arrives as a measurement whose noise falls as its inlier
count rises. A solve backed by five hundred agreeing ground points moves the
estimate; one backed by forty barely does. Nothing is thrown away and nothing
is trusted absolutely.

Scale still comes from the ground plane: vision reports metres per second
because its translation is metric, and the filter never invents scale of its
own.
"""

import math
from typing import Optional, Tuple

import numpy as np


class PlanarVelocityFilter:
    """A two state Kalman filter over world-frame planar velocity."""

    def __init__(
        self,
        acceleration_noise: float = 1.55,
        vision_noise: float = 0.25,
        vision_reference_inliers: float = 300.0,
        initial_variance: float = 25.0,
        innovation_gate: float = 9.0,
        outlier_inflation: float = 25.0,
    ):
        # Spectral density of the acceleration the model does not know about,
        # in m/s^2. Larger means the filter leans on vision sooner.
        self.acceleration_noise = float(acceleration_noise)
        # Velocity noise of a vision solve carrying the reference inlier count.
        self.vision_noise = float(vision_noise)
        self.vision_reference_inliers = float(vision_reference_inliers)
        self.innovation_gate = float(innovation_gate)
        # A measurement that fails the consistency check is not discarded, its
        # variance is inflated. Discarding leaves the filter on dead reckoning
        # alone, and an integral with nothing observing it walks away: it read
        # 14 m/s while the vehicle did 6.
        self.outlier_inflation = float(outlier_inflation)
        self.velocity = np.zeros(2, dtype=np.float64)
        self.covariance = np.eye(2, dtype=np.float64) * float(initial_variance)
        self.updates = 0
        self.rejected = 0

    @property
    def settled(self) -> bool:
        """Whether vision has pinned the velocity down at least once.

        Before that the covariance still carries the startup guess, and any
        prediction built on it says more about the initial variance than about
        the vehicle.
        """
        return self.updates > 0

    def predict(self, acceleration_world: np.ndarray, dt: float) -> None:
        if dt <= 0.0:
            return
        self.velocity = self.velocity + np.asarray(
            acceleration_world, dtype=np.float64
        ).reshape(2) * dt
        # Variance grows as q^2*dt, the white-noise form the docstring names,
        # not (q*dt)^2. While predict() is driven by a fixed-rate IMU the two
        # differ only by a constant and the tuning absorbs it -- which is why
        # q=12 was a sharp optimum under the old form. They stop agreeing the
        # moment the rate changes, and the vehicle's IMU runs at 100 Hz where
        # the simulation's runs at 60: the old form would have quietly cut the
        # process noise by 2.8x on the car, against 1.7x for the correct one,
        # at an operating point where 12 -> 8 costs a factor of five in ATE.
        growth = (self.acceleration_noise ** 2) * dt
        self.covariance = self.covariance + np.eye(2) * growth

    def measurement_variance(self, inliers: int) -> float:
        """Vision noise falls as the supporting inlier count rises."""
        count = max(float(inliers), 1.0)
        scale = self.vision_reference_inliers / count
        return (self.vision_noise ** 2) * scale

    def update(
        self, measured: np.ndarray, inliers: int, extra_variance: float = 0.0
    ) -> bool:
        """Fold in a vision velocity. Returns False if it was gated out.

        `extra_variance` carries evidence the solve itself cannot see. The
        front and rear cameras look at different ground and solve
        independently, so how far apart their answers are is a measure of how
        much either can be trusted, and no amount of internal agreement inside
        one of them reveals it.
        """
        measured = np.asarray(measured, dtype=np.float64).reshape(2)
        variance = self.measurement_variance(inliers) + max(float(extra_variance), 0.0)
        innovation = measured - self.velocity
        innovation_covariance = self.covariance + np.eye(2) * variance
        try:
            inverse = np.linalg.inv(innovation_covariance)
        except np.linalg.LinAlgError:
            return False

        # Normalised innovation squared, the standard consistency check. A
        # solve that disagrees with both the model and its own uncertainty is
        # more likely wrong than the vehicle is surprising.
        distance = float(innovation @ inverse @ innovation)
        accepted = True
        if self.settled and distance > self.innovation_gate:
            self.rejected += 1
            accepted = False
            innovation_covariance = self.covariance + np.eye(2) * (
                variance * self.outlier_inflation
            )
            try:
                inverse = np.linalg.inv(innovation_covariance)
            except np.linalg.LinAlgError:
                return False

        gain = self.covariance @ inverse
        self.velocity = self.velocity + gain @ innovation
        self.covariance = (np.eye(2) - gain) @ self.covariance
        self.covariance = 0.5 * (self.covariance + self.covariance.T)
        self.updates += 1
        return accepted

    def body_translation(self, dt: float, yaw: float) -> np.ndarray:
        """Travel over dt at the fused velocity, in the body frame at `yaw`."""
        delta = self.velocity * float(dt)
        c = math.cos(yaw)
        s = math.sin(yaw)
        return np.array([c * delta[0] + s * delta[1], -s * delta[0] + c * delta[1]])

    def speed(self) -> float:
        return float(np.linalg.norm(self.velocity))

    def uncertainty(self) -> float:
        return float(math.sqrt(max(np.trace(self.covariance), 0.0)))


def world_velocity_from_motion(
    motion_x: float, motion_y: float, dt: float, yaw: float
) -> Optional[Tuple[float, float]]:
    """Turn a body-frame frame-to-frame translation into world velocity."""
    if dt <= 0.0:
        return None
    c = math.cos(yaw)
    s = math.sin(yaw)
    return (
        (c * motion_x - s * motion_y) / dt,
        (s * motion_x + c * motion_y) / dt,
    )


class PlanarDisplacementFilter:
    """Fuse what each sensor measures: acceleration, and displacement between
    two image times.

    The velocity filter divides the vision translation by the interval and
    calls the quotient a velocity measurement. That carries the interval's own
    error into the measurement, inflates its noise by 1/dt^2, and treats an
    average over the interval as the value at its end -- 15 cm of modelling
    error per update at 0.1 s hops and 3 m/s. It also has no bias state, so
    any accelerometer offset integrates into velocity and only vision removes
    it.

    The fix is not simply to measure displacement instead. A first attempt
    zeroed the displacement at every hop, on the argument that an anchor you
    define is an anchor you know exactly. It reproduced vision alone almost
    to the millimetre (0.115 m against vision's 0.124, where the velocity
    filter reads 0.070) because resetting the anchor throws away every
    correlation the previous hop earned: each hop became an independent
    estimate from one measurement, and the smoothing that made the velocity
    filter worth having disappeared.

    So the anchor is a state, cloned from the current position when the hop
    opens and carried with its cross-covariance -- stochastic cloning, which
    exists for exactly this. Vision measures p - a; the correlation between
    them is what lets one hop inform the next.

    State: [position, anchor, velocity, bias], all planar, the bias in the
    body frame because that is where an instrument's offset lives.

    Carrying the repeatable part of the vision error as a state as well was
    tried and does not work: the accelerometer's own per-hop error is 54 mm
    where the offset to be found is under 1 mm, so the extra state is not
    observable and the trajectory fell apart at any walk above zero.
    """

    P, A, V, B = slice(0, 2), slice(2, 4), slice(4, 6), slice(6, 8)

    def __init__(
        self,
        acceleration_noise: float = 1.55,
        bias_walk: float = 0.05,
        vision_noise_m: float = 0.02,
        vision_reference_inliers: float = 300.0,
        initial_velocity_variance: float = 25.0,
        initial_bias_variance: float = 1.0,
        innovation_gate: float = 9.0,
        outlier_inflation: float = 25.0,
    ):
        self.acceleration_noise = float(acceleration_noise)
        self.bias_walk = float(bias_walk)
        self.vision_noise_m = float(vision_noise_m)
        self.vision_reference_inliers = float(vision_reference_inliers)
        self.innovation_gate = float(innovation_gate)
        self.outlier_inflation = float(outlier_inflation)
        self.state = np.zeros(8, dtype=np.float64)
        self.covariance = np.diag(np.array([
            0.0, 0.0, 0.0, 0.0,
            initial_velocity_variance, initial_velocity_variance,
            initial_bias_variance, initial_bias_variance,
        ], dtype=np.float64))
        self.updates = 0
        self.rejected = 0
        self.last_update = None

    @property
    def settled(self) -> bool:
        return self.updates > 0

    @property
    def velocity(self) -> np.ndarray:
        return self.state[self.V]

    @property
    def bias(self) -> np.ndarray:
        return self.state[self.B]

    def open_hop(self) -> None:
        """Clone the current position as the anchor the next hop measures from."""
        self.state[self.A] = self.state[self.P]
        self.covariance[self.A, :] = self.covariance[self.P, :]
        self.covariance[:, self.A] = self.covariance[:, self.P]
        self.covariance[self.A, self.A] = self.covariance[self.P, self.P]

    def predict(self, acceleration_world, dt: float, yaw: float = 0.0) -> None:
        """Propagate over dt. `yaw` orients the bias, which is body-fixed."""
        if dt <= 0.0:
            return
        accel = np.asarray(acceleration_world, dtype=np.float64).reshape(2)
        c, s = math.cos(yaw), math.sin(yaw)
        rotation = np.array([[c, -s], [s, c]])
        corrected = accel - rotation @ self.state[self.B]
        self.state[self.P] += self.state[self.V] * dt + 0.5 * corrected * dt * dt
        self.state[self.V] += corrected * dt

        transition = np.eye(self.state.size)
        transition[self.P, self.V] = np.eye(2) * dt
        transition[self.P, self.B] = rotation * (-0.5 * dt * dt)
        transition[self.V, self.B] = rotation * (-dt)
        self.covariance = transition @ self.covariance @ transition.T

        spectral = self.acceleration_noise ** 2
        noise = np.zeros((self.state.size, self.state.size))
        noise[self.P, self.P] = np.eye(2) * (spectral * dt ** 3 / 3.0)
        noise[self.P, self.V] = np.eye(2) * (spectral * dt ** 2 / 2.0)
        noise[self.V, self.P] = noise[self.P, self.V]
        noise[self.V, self.V] = np.eye(2) * (spectral * dt)
        noise[self.B, self.B] = np.eye(2) * (self.bias_walk ** 2 * dt)
        self.covariance = self.covariance + noise

    def measurement_variance(self, inliers: int, spread: float = 0.0) -> float:
        """How well this solve knows the displacement, in metres squared.

        The inlier count alone is a proxy: it says how many votes there were,
        not how far apart they landed. The solve already measures the second
        thing -- `spread` is the RMS residual of the inlying votes -- and the
        standard error of their mean is spread^2/N. Through a turn the votes
        can be many and still disagree, which is precisely the case the count
        cannot see and the lateral error accumulates in.

        The configured noise stays as a floor. A solve whose votes happen to
        agree closely is not thereby exact; it can still be wrong together.
        Charging for that separately -- a share of the spread that does not
        average down with the vote count -- was tried against truth and did
        not pay: it flags the right hops, the ones where a standing vehicle
        was reported as travelling up to 104 mm, but those hops turn out not
        to be what the trajectory error is made of.
        """
        count = max(float(inliers), 1.0)
        floor = (self.vision_noise_m ** 2) * (self.vision_reference_inliers / count)
        if spread > 0.0:
            return max(floor, (spread ** 2) / count)
        return floor

    def _observation(self) -> np.ndarray:
        model = np.zeros((2, self.state.size))
        model[:, self.P] = np.eye(2)
        model[:, self.A] = -np.eye(2)
        return model

    def update(self, displacement_world, inliers: int,
               extra_variance: float = 0.0, spread: float = 0.0) -> bool:
        measured = np.asarray(displacement_world, dtype=np.float64).reshape(2)
        variance = (self.measurement_variance(inliers, spread)
                    + max(float(extra_variance), 0.0))
        model = self._observation()
        innovation = measured - (self.state[self.P] - self.state[self.A])
        block = model @ self.covariance @ model.T + np.eye(2) * variance
        try:
            inverse = np.linalg.inv(block)
        except np.linalg.LinAlgError:
            return False
        accepted = True
        if self.settled and float(innovation @ inverse @ innovation) > self.innovation_gate:
            self.rejected += 1
            accepted = False
            block = model @ self.covariance @ model.T + np.eye(2) * (
                variance * self.outlier_inflation)
            try:
                inverse = np.linalg.inv(block)
            except np.linalg.LinAlgError:
                return False
        gain = self.covariance @ model.T @ inverse
        # What the filter claimed, kept for the run that checks whether it was
        # true. A filter is only as good as its covariance: if the innovation
        # is routinely larger than the S it predicts, the noise model is wrong
        # and no amount of searching over q finds the right answer, it only
        # finds the value that hurts least.
        prior = model @ self.covariance @ model.T
        self.last_update = {
            'measured': measured.copy(),
            'predicted': (self.state[self.P] - self.state[self.A]).copy(),
            'innovation': innovation.copy(),
            'variance': float(variance),
            'prior_trace': float(np.trace(prior)),
            'nis': float(innovation @ inverse @ innovation),
            'accepted': accepted,
        }
        self.state = self.state + gain @ innovation
        # Joseph form. `P - K H P` is algebraically equal to it and numerically
        # is not: it subtracts two nearly equal matrices, and a filter that
        # runs a few hundred updates on eight states drifts out of positive
        # definiteness that way. Joseph stays symmetric and positive by
        # construction at the cost of one more product.
        spread = np.eye(self.state.size) - gain @ model
        self.covariance = (
            spread @ self.covariance @ spread.T
            + gain @ (np.eye(2) * variance) @ gain.T
        )
        self.covariance = 0.5 * (self.covariance + self.covariance.T)
        self.last_update['posterior'] = (
            self.state[self.P] - self.state[self.A]).copy()
        self.updates += 1
        return accepted

    def update_zero_velocity(self, sigma: float = 0.01) -> None:
        """The vehicle is standing still. Say so.

        This is where the accelerometer bias becomes observable: at rest the
        instrument still reports something, and whatever velocity that
        something integrates into is the bias. Without it the bias state has
        nothing to separate it from real motion, which is why it sat inert
        through every tuning of its own process noise. A parking manoeuvre
        stops several times, so the information is there for the taking.
        """
        model = np.zeros((2, self.state.size))
        model[:, self.V] = np.eye(2)
        innovation = -self.state[self.V]
        block = model @ self.covariance @ model.T + np.eye(2) * (sigma ** 2)
        try:
            inverse = np.linalg.inv(block)
        except np.linalg.LinAlgError:
            return
        gain = self.covariance @ model.T @ inverse
        self.state = self.state + gain @ innovation
        spread = np.eye(self.state.size) - gain @ model
        self.covariance = (
            spread @ self.covariance @ spread.T
            + gain @ (np.eye(2) * sigma ** 2) @ gain.T
        )
        self.covariance = 0.5 * (self.covariance + self.covariance.T)

    def body_translation(self, yaw: float) -> np.ndarray:
        """The fused displacement of this hop, in the body frame at `yaw`."""
        c = math.cos(yaw)
        s = math.sin(yaw)
        d = self.state[self.P] - self.state[self.A]
        return np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])

    def speed(self) -> float:
        return float(np.linalg.norm(self.state[self.V]))
