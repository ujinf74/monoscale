"""Short-horizon translation prediction from the IMU.

The estimator solved every frame from scratch, with nothing to check the answer
against. When tracking degraded the solve could return "barely moved" and there
was no way to know it was wrong; above about 2 m/s that is exactly what
happened.

Integrating the accelerometer between two camera frames gives an independent
prediction of how far the vehicle travelled. Over the 50 ms between frames the
integration is trustworthy; over a minute it is not, which is why vision
corrects the velocity every time it produces an answer worth believing.

This deliberately does not supply scale. Scale stays with the ground plane,
which is observable at constant velocity where an accelerometer is not.
"""

from collections import deque
import math
from typing import Deque, Optional, Tuple

import numpy as np


def _rotation_from_quaternion(x: float, y: float, z: float, w: float) -> np.ndarray:
    norm = x * x + y * y + z * z + w * w
    if norm < 1e-12:
        return np.eye(3)
    s = 2.0 / norm
    return np.array(
        [
            [1.0 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
            [s * (x * y + z * w), 1.0 - s * (x * x + z * z), s * (y * z - x * w)],
            [s * (x * z - y * w), s * (y * z + x * w), 1.0 - s * (x * x + y * y)],
        ]
    )


class PlanarInertialPropagator:
    """Dead reckons horizontal motion, and takes velocity corrections."""

    def __init__(
        self,
        gravity: float = 9.81,
        max_gap_sec: float = 0.25,
        history_sec: float = 4.0,
        max_horizontal_acceleration: float = 12.0,
        median_window: int = 1,
        integration_method: str = 'rk4',
    ):
        self.gravity = float(gravity)
        self.max_gap = float(max_gap_sec)
        self.history_sec = float(history_sec)
        self.max_horizontal_acceleration = max(
            float(max_horizontal_acceleration), 0.0
        )
        self.acceleration_window: Deque[np.ndarray] = deque(
            maxlen=max(int(median_window), 1)
        )
        if integration_method not in ('zoh', 'rk4'):
            raise ValueError("integration_method must be 'zoh' or 'rk4'")
        self.integration_method = integration_method
        self.previous_acceleration: Optional[np.ndarray] = None
        self.rejected_samples = 0
        self.velocity = np.zeros(2, dtype=np.float64)
        self.position = np.zeros(2, dtype=np.float64)
        self.last_stamp: Optional[float] = None
        # An integral nobody has anchored has an unknown constant. Spawning the
        # vehicle drops it onto the road, and that transient integrates into a
        # velocity that never existed, so predictions stay unavailable until
        # vision has pinned the velocity down at least once.
        self.corrected = False
        # (stamp, position, velocity) so a delta between two camera frames can
        # be read off without re-integrating.
        self.history: Deque[Tuple[float, np.ndarray, np.ndarray]] = deque()

    def add_sample(self, stamp: float, orientation, acceleration):
        """Integrate one IMU sample. Orientation is (x, y, z, w).

        Returns the interval-effective horizontal acceleration and the step it
        covered, so a filter can propagate with the same integration method.
        """
        rotation = _rotation_from_quaternion(*orientation)
        world = rotation @ np.asarray(acceleration, dtype=np.float64)
        # The accelerometer measures proper acceleration, so at rest it reads
        # gravity pointing up. Remove it before integrating.
        world[2] -= self.gravity
        horizontal = world[:2]

        # CARLA occasionally reports physics impulses of hundreds or thousands
        # of m/s2 when the suspension contacts or the actor is corrected. Such
        # a sample is not useful vehicle motion and corrupts velocity for many
        # frames after a single 10 ms integration step. Reject it before the
        # short causal median used for the remaining contact noise.
        if (
            self.max_horizontal_acceleration > 0.0
            and float(np.linalg.norm(horizontal)) > self.max_horizontal_acceleration
        ):
            self.rejected_samples += 1
        else:
            self.acceleration_window.append(horizontal)
        filtered = (
            np.median(np.stack(self.acceleration_window), axis=0)
            if self.acceleration_window
            else np.zeros(2, dtype=np.float64)
        )

        # Do not integrate the spawn/drop transient before vision supplies the
        # unknown constant velocity. This also keeps the separate velocity
        # filter from receiving unanchored acceleration at startup.
        usable = filtered if self.corrected else np.zeros(2, dtype=np.float64)

        effective = usable
        if self.last_stamp is not None:
            dt = stamp - self.last_stamp
            if 0.0 < dt <= self.max_gap:
                if self.integration_method == 'rk4' and self.previous_acceleration is not None:
                    # RK4 for x'=v, v'=a with a linearly interpolated between
                    # the adjacent IMU samples. The midpoint acceleration is
                    # (a0+a1)/2; this is exact for that interpolation model.
                    previous = self.previous_acceleration
                    effective = 0.5 * (previous + usable)
                    self.position += self.velocity * dt + dt * dt * (
                        previous / 3.0 + usable / 6.0
                    )
                    self.velocity += 0.5 * (previous + usable) * dt
                else:
                    self.position += self.velocity * dt + 0.5 * usable * dt * dt
                    self.velocity += usable * dt
            elif dt > self.max_gap:
                # A gap this long makes the integral meaningless; start clean.
                self.velocity[:] = 0.0
                self.previous_acceleration = None
        step = 0.0 if self.last_stamp is None else stamp - self.last_stamp
        self.last_stamp = stamp
        self.previous_acceleration = usable.copy()
        self.history.append((stamp, self.position.copy(), self.velocity.copy()))
        while self.history and stamp - self.history[0][0] > self.history_sec:
            self.history.popleft()
        return effective, (step if 0.0 < step <= self.max_gap else 0.0)

    def _sample_at(self, stamp: float):
        if not self.history:
            return None
        if stamp <= self.history[0][0]:
            return self.history[0]
        previous = self.history[0]
        for entry in self.history:
            if entry[0] >= stamp:
                span = entry[0] - previous[0]
                if span <= 0.0:
                    return entry
                ratio = (stamp - previous[0]) / span
                return (
                    stamp,
                    previous[1] + ratio * (entry[1] - previous[1]),
                    previous[2] + ratio * (entry[2] - previous[2]),
                )
            previous = entry
        return self.history[-1]

    def predicted_translation(
        self, start: float, end: float, yaw: float
    ) -> Optional[np.ndarray]:
        """Translation between two times, expressed in the body frame at `yaw`."""
        if not self.corrected:
            return None
        first = self._sample_at(start)
        second = self._sample_at(end)
        if first is None or second is None or end <= start:
            return None
        delta = second[1] - first[1]
        c = math.cos(yaw)
        s = math.sin(yaw)
        return np.array([c * delta[0] + s * delta[1], -s * delta[0] + c * delta[1]])

    def correct_velocity(self, velocity_world, gain: float = 1.0) -> None:
        """Fold a vision-derived velocity into the integrated one.

        Without this the integral walks off within seconds even on a perfect
        accelerometer, because nothing observes the constant of integration.
        """
        measured = np.asarray(velocity_world, dtype=np.float64).reshape(2)
        gain = min(max(gain, 0.0), 1.0)
        self.corrected = True
        self.velocity += gain * (measured - self.velocity)
        if self.history:
            stamp, position, _ = self.history[-1]
            self.history[-1] = (stamp, position, self.velocity.copy())


def agrees_with_prediction(
    vision: np.ndarray,
    predicted: Optional[np.ndarray],
    absolute_tolerance: float,
    relative_tolerance: float,
    minimum_prediction: float = 0.05,
) -> bool:
    """Whether a vision translation is close enough to the inertial one.

    Only judges when the prediction carries information. A vehicle at constant
    velocity produces no horizontal acceleration, so a propagator that has not
    yet been told its velocity predicts nothing; gating on that would reject
    perfectly good vision, including the first solve after startup.

    Generous on purpose otherwise: this exists to catch the collapse to near
    zero motion, not to second-guess a working solve.
    """
    if predicted is None:
        return True
    predicted = np.asarray(predicted, dtype=np.float64).reshape(2)
    if float(np.linalg.norm(predicted)) < minimum_prediction:
        return True
    vision = np.asarray(vision, dtype=np.float64).reshape(2)
    tolerance = absolute_tolerance + relative_tolerance * float(
        np.linalg.norm(predicted)
    )
    return bool(np.linalg.norm(vision - predicted) <= tolerance)
