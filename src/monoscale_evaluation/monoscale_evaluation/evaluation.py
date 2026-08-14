"""Trajectory comparison used to score vision odometry against CARLA truth.

The estimator starts at the identity pose in its own frame, so the ground
truth is rigidly anchored to the estimate at the first matched sample and no
further alignment is applied. Residual error is therefore accumulated drift,
not a best-fit fitting error.
"""

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Pose2:
    stamp: float
    x: float
    y: float
    yaw: float


def wrap_pi(angle: float) -> float:
    return math.remainder(angle, 2.0 * math.pi)


def interpolate_pose(before: Pose2, after: Pose2, stamp: float) -> Pose2:
    span = after.stamp - before.stamp
    if span <= 0.0:
        return Pose2(stamp, after.x, after.y, after.yaw)
    ratio = (stamp - before.stamp) / span
    yaw_step = wrap_pi(after.yaw - before.yaw)
    return Pose2(
        stamp,
        before.x + ratio * (after.x - before.x),
        before.y + ratio * (after.y - before.y),
        wrap_pi(before.yaw + ratio * yaw_step),
    )


def relative_pose(origin: Pose2, pose: Pose2) -> Tuple[float, float, float]:
    """Express `pose` in the frame of `origin`."""
    c = math.cos(origin.yaw)
    s = math.sin(origin.yaw)
    dx = pose.x - origin.x
    dy = pose.y - origin.y
    return (c * dx + s * dy, -s * dx + c * dy, wrap_pi(pose.yaw - origin.yaw))


def compose_pose(base: Tuple[float, float, float], delta: Tuple[float, float, float]):
    c = math.cos(base[2])
    s = math.sin(base[2])
    return (
        base[0] + c * delta[0] - s * delta[1],
        base[1] + s * delta[0] + c * delta[1],
        wrap_pi(base[2] + delta[2]),
    )


def shift_along_heading(x: float, y: float, yaw: float, distance: float):
    """Move a pose along its own heading, e.g. vehicle centre to rear axle."""
    return (x + distance * math.cos(yaw), y + distance * math.sin(yaw))


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else float('nan')


class TrajectoryMetrics:
    """Accumulates absolute drift and segment-wise relative error."""

    def __init__(self, rpe_delta_m: float = 1.0):
        self.rpe_delta = max(float(rpe_delta_m), 1e-3)
        self.samples = 0
        self.first_stamp: Optional[float] = None
        self.last_stamp: Optional[float] = None
        self.squared_position_error = 0.0
        self.max_position_error = 0.0
        self.final_position_error = 0.0
        self.squared_yaw_error = 0.0
        self.max_yaw_error = 0.0
        self.truth_distance = 0.0
        self.estimate_distance = 0.0
        self.relative_translation_ratios: List[float] = []
        self.relative_yaw_errors: List[float] = []
        self.segment_truth_travel = 0.0
        self.segment_estimate_travel = 0.0
        self._previous_truth: Optional[Tuple[float, float, float]] = None
        self._previous_estimate: Optional[Tuple[float, float, float]] = None
        self._anchor_truth: Optional[Tuple[float, float, float]] = None
        self._anchor_estimate: Optional[Tuple[float, float, float]] = None
        self._anchor_distance = 0.0

    def add(
        self,
        stamp: float,
        estimate: Tuple[float, float, float],
        truth: Tuple[float, float, float],
    ) -> None:
        position_error = math.hypot(estimate[0] - truth[0], estimate[1] - truth[1])
        yaw_error = abs(wrap_pi(estimate[2] - truth[2]))

        self.samples += 1
        if self.first_stamp is None:
            self.first_stamp = stamp
        self.last_stamp = stamp
        self.squared_position_error += position_error * position_error
        self.max_position_error = max(self.max_position_error, position_error)
        self.final_position_error = position_error
        self.squared_yaw_error += yaw_error * yaw_error
        self.max_yaw_error = max(self.max_yaw_error, yaw_error)

        if self._previous_truth is not None:
            self.truth_distance += math.hypot(
                truth[0] - self._previous_truth[0], truth[1] - self._previous_truth[1]
            )
            self.estimate_distance += math.hypot(
                estimate[0] - self._previous_estimate[0],
                estimate[1] - self._previous_estimate[1],
            )
        self._previous_truth = truth
        self._previous_estimate = estimate

        if self._anchor_truth is None:
            self._anchor_truth = truth
            self._anchor_estimate = estimate
            self._anchor_distance = self.truth_distance
            return
        if self.truth_distance - self._anchor_distance < self.rpe_delta:
            return
        self._close_segment(truth, estimate)

    def _close_segment(self, truth, estimate) -> None:
        anchor_truth = Pose2(0.0, *self._anchor_truth)
        anchor_estimate = Pose2(0.0, *self._anchor_estimate)
        truth_step = relative_pose(anchor_truth, Pose2(0.0, *truth))
        estimate_step = relative_pose(anchor_estimate, Pose2(0.0, *estimate))
        travelled = math.hypot(truth_step[0], truth_step[1])
        # Straight line displacement per segment, which unlike a summed path
        # length is not inflated by per sample noise.
        self.segment_truth_travel += travelled
        self.segment_estimate_travel += math.hypot(estimate_step[0], estimate_step[1])
        if travelled > 1e-6:
            error = math.hypot(
                estimate_step[0] - truth_step[0], estimate_step[1] - truth_step[1]
            )
            self.relative_translation_ratios.append(error / travelled)
            self.relative_yaw_errors.append(
                abs(wrap_pi(estimate_step[2] - truth_step[2])) / travelled
            )
        self._anchor_truth = truth
        self._anchor_estimate = estimate
        self._anchor_distance = self.truth_distance

    def summary(self) -> Dict[str, float]:
        if self.samples == 0:
            return {'samples': 0}
        duration = (self.last_stamp or 0.0) - (self.first_stamp or 0.0)
        return {
            'samples': self.samples,
            'duration_sec': duration,
            'truth_distance_m': self.truth_distance,
            'estimate_distance_m': self.estimate_distance,
            'scale_ratio': (
                self.estimate_distance / self.truth_distance
                if self.truth_distance > 1e-6
                else float('nan')
            ),
            'segment_scale_ratio': (
                self.segment_estimate_travel / self.segment_truth_travel
                if self.segment_truth_travel > 1e-6
                else float('nan')
            ),
            'position_rmse_m': math.sqrt(self.squared_position_error / self.samples),
            'position_max_m': self.max_position_error,
            'final_position_error_m': self.final_position_error,
            'drift_percent': (
                100.0 * self.final_position_error / self.truth_distance
                if self.truth_distance > 1e-6
                else float('nan')
            ),
            'yaw_rmse_deg': math.degrees(
                math.sqrt(self.squared_yaw_error / self.samples)
            ),
            'yaw_max_deg': math.degrees(self.max_yaw_error),
            'relative_translation_percent': 100.0
            * _mean(self.relative_translation_ratios),
            'relative_yaw_deg_per_m': math.degrees(_mean(self.relative_yaw_errors)),
            'relative_segments': len(self.relative_translation_ratios),
        }


def format_summary(summary: Dict[str, float]) -> str:
    if not summary.get('samples'):
        return 'no matched samples yet'
    return (
        'n={samples} gt={truth_distance_m:.1f}m scale={scale_ratio:.3f} '
        'ate_rmse={position_rmse_m:.3f}m final={final_position_error_m:.3f}m '
        'drift={drift_percent:.2f}% yaw_rmse={yaw_rmse_deg:.2f}deg '
        'rpe={relative_translation_percent:.2f}%/{relative_yaw_deg_per_m:.3f}deg_per_m'
    ).format(**summary)
