"""Scores the vision odometry against the CARLA ground truth pose.

The CARLA bridge publishes the exact ego transform through its GNSS pseudo
sensor, so the only work here is frame bookkeeping: the truth is the vehicle
centre in the CARLA world frame while the estimate is base_link in its own
start frame.
"""

from collections import deque
import math
import os
from typing import Deque, Optional

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry, Path
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float32

from .evaluation import compose_pose
from .evaluation import format_summary
from .evaluation import interpolate_pose
from .evaluation import Pose2
from .evaluation import relative_pose
from .evaluation import shift_along_heading
from .evaluation import TrajectoryMetrics
from .evaluation import wrap_pi


def _yaw_from_quaternion(q) -> float:
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    )


def _stamp_seconds(header) -> float:
    return header.stamp.sec + header.stamp.nanosec * 1e-9


def _pose_stamped(stamp_msg, frame_id, x, y, yaw):
    pose = PoseStamped()
    pose.header.stamp = stamp_msg
    pose.header.frame_id = frame_id
    pose.pose.position.x = float(x)
    pose.pose.position.y = float(y)
    pose.pose.orientation.z = math.sin(0.5 * yaw)
    pose.pose.orientation.w = math.cos(0.5 * yaw)
    return pose


class OdometryEvaluator(Node):
    def __init__(self, **kwargs):
        super().__init__('odometry_evaluator', **kwargs)
        declare = self.declare_parameter
        declare('estimate_topic', '/localization/kinematic_state')
        declare('truth_topic', '/sensing/gnss/pose_with_covariance')
        declare('estimate_path_topic', '/vision/evaluation/estimate_path')
        declare('truth_path_topic', '/vision/evaluation/truth_path')
        declare('error_topic', '/vision/evaluation/position_error')
        declare('map_frame', 'map')
        declare('queue_depth', 50)
        declare('truth_reliability', 'best_effort')
        # The CARLA actor origin sits at the vehicle centre while the stack
        # estimates base_link at the rear axle.
        declare('truth_is_vehicle_center', True)
        # The IONIQ's, matching vehicle_info.param.yaml and the sensor kit.
        # 2.7 here would shift the truth 6 cm along the body -- the same order
        # as the whole remaining error -- for anyone who runs this node
        # without the params file.
        declare('wheel_base', 2.8192)
        declare('match_tolerance_sec', 0.05)
        # The CARLA bridge stamps the ego pose from the process game clock and
        # the cameras from the simulator clock, which are minutes apart. With
        # 'auto' the first estimate and truth seen together latch the constant
        # offset between them; 'none' keeps the raw stamps.
        declare('clock_offset_mode', 'auto')
        declare('clock_offset_threshold_sec', 1.0)
        # Close the window after this much ground truth travel. The run ends
        # on a wall clock timeout while the manoeuvre is scripted in
        # simulation time, so runs used to carry between 0.5 and 12.5 extra
        # seconds of standing still. Drift is a final error over a distance,
        # and standing still grows the error while the distance stays put, so
        # the tail alone moved the number. Ending on distance gives every run
        # the same stretch of road. 0 keeps everything.
        declare('evaluation_distance_m', 0.0)
        declare('relative_segment_m', 1.0)
        declare('report_period_sec', 2.0)
        declare('path_stride', 5)
        declare('max_path_poses', 4000)
        declare('output_directory', '')

        get = self.get_parameter
        self.map_frame = str(get('map_frame').value)
        self.center_offset = (
            0.5 * float(get('wheel_base').value)
            if bool(get('truth_is_vehicle_center').value)
            else 0.0
        )
        self.match_tolerance = max(float(get('match_tolerance_sec').value), 0.0)
        self.auto_clock_offset = str(get('clock_offset_mode').value) == 'auto'
        self.clock_offset_threshold = abs(
            float(get('clock_offset_threshold_sec').value)
        )
        self.clock_offset = 0.0
        self.clock_offset_latched = not self.auto_clock_offset
        self.path_stride = max(int(get('path_stride').value), 1)
        self.max_path_poses = max(int(get('max_path_poses').value), 2)
        self.output_directory = str(get('output_directory').value)

        self.metrics = TrajectoryMetrics(float(get('relative_segment_m').value))
        self.evaluation_distance = float(get('evaluation_distance_m').value)
        self.truth_buffer: Deque[Pose2] = deque(maxlen=600)
        self.truth_origin: Optional[Pose2] = None
        self.estimate_origin = (0.0, 0.0, 0.0)
        self.pending: Deque[tuple] = deque(maxlen=600)
        self.estimate_path = Path()
        self.truth_path = Path()
        self.estimate_path.header.frame_id = self.map_frame
        self.truth_path.header.frame_id = self.map_frame
        self.matched = 0
        self.anchor_stamp = float('nan')
        self.first_estimate_stamp = float('nan')
        self.window_closed = False
        self.dropped_stale = 0
        self.trajectory_rows = []

        reliable = QoSProfile(
            depth=int(get('queue_depth').value),
            reliability=ReliabilityPolicy.RELIABLE)
        # The truth this scores against arrived best effort until 08-14, which
        # means the yardstick itself could lose samples under load. On the
        # vehicle there is no truth topic at all, so nothing here ships.
        sensor = QoSProfile(
            depth=int(get('queue_depth').value),
            reliability=(
                ReliabilityPolicy.RELIABLE
                if str(get('truth_reliability').value) == 'reliable'
                else ReliabilityPolicy.BEST_EFFORT
            ))
        self.create_subscription(
            Odometry, str(get('estimate_topic').value), self._on_estimate, reliable
        )
        self.create_subscription(
            PoseWithCovarianceStamped,
            str(get('truth_topic').value),
            self._on_truth,
            sensor,
        )
        self.estimate_path_pub = self.create_publisher(
            Path, str(get('estimate_path_topic').value), 1
        )
        self.truth_path_pub = self.create_publisher(
            Path, str(get('truth_path_topic').value), 1
        )
        self.error_pub = self.create_publisher(
            Float32, str(get('error_topic').value), 10
        )
        period = max(float(get('report_period_sec').value), 0.5)
        self.create_timer(period, self._report)

    def _on_truth(self, msg: PoseWithCovarianceStamped):
        stamp = _stamp_seconds(msg.header)
        if stamp <= 0.0:
            return
        stamp = self._corrected_stamp(stamp)
        pose = msg.pose.pose
        yaw = _yaw_from_quaternion(pose.orientation)
        x, y = pose.position.x, pose.position.y
        if self.center_offset != 0.0:
            x, y = shift_along_heading(x, y, yaw, -self.center_offset)
        self.truth_buffer.append(Pose2(stamp, x, y, yaw))
        self._drain_pending()

    def _corrected_stamp(self, stamp: float) -> float:
        """Latch the constant offset between the two publisher clocks, once."""
        if not self.clock_offset_latched:
            if not self.pending:
                return stamp
            self.clock_offset_latched = True
            candidate = self.pending[-1][0] - stamp
            if abs(candidate) > self.clock_offset_threshold:
                self.clock_offset = candidate
                self.get_logger().warning(
                    f'ground truth stamps trail the estimate by {candidate:+.1f}s, '
                    'which is a bridge clock mismatch rather than transport delay. '
                    'Pairing the two streams by arrival instead, so the reported '
                    'error carries the transport jitter of that single pairing'
                )
        return stamp + self.clock_offset

    def _on_estimate(self, msg: Odometry):
        stamp = _stamp_seconds(msg.header)
        if stamp <= 0.0:
            return
        if self.first_estimate_stamp != self.first_estimate_stamp:  # NaN
            self.first_estimate_stamp = stamp
        pose = msg.pose.pose
        self.pending.append(
            (
                stamp,
                msg.header.stamp,
                pose.position.x,
                pose.position.y,
                _yaw_from_quaternion(pose.orientation),
            )
        )
        self._drain_pending()

    def _truth_at(self, stamp: float) -> Optional[Pose2]:
        if len(self.truth_buffer) < 2:
            return None
        if stamp < self.truth_buffer[0].stamp - self.match_tolerance:
            return None
        if stamp > self.truth_buffer[-1].stamp:
            return None
        previous = self.truth_buffer[0]
        for sample in self.truth_buffer:
            if sample.stamp >= stamp:
                return interpolate_pose(previous, sample, stamp)
            previous = sample
        return None

    def _drain_pending(self):
        while self.pending:
            stamp, stamp_msg, x, y, yaw = self.pending[0]
            if not self.truth_buffer:
                return
            if stamp > self.truth_buffer[-1].stamp:
                return
            self.pending.popleft()
            truth = self._truth_at(stamp)
            if truth is None:
                self.dropped_stale += 1
                continue
            self._accumulate(stamp, stamp_msg, (x, y, yaw), truth)

    def _accumulate(self, stamp, stamp_msg, estimate, truth: Pose2):
        if self.window_closed:
            return
        if self.truth_origin is None:
            self.truth_origin = truth
            self.estimate_origin = estimate
            self.anchor_stamp = stamp
            # Where the anchor landed decides everything downstream: the truth
            # is pinned to the estimate here and never realigned, so a run
            # that anchors after the vehicle has already moved reads that
            # displacement as drift for the rest of the trajectory. Making the
            # truth link reliable moved the first matched sample and turned
            # 0.07 m into 4 m on the same recording -- the estimate had not
            # changed at all. Record it, and say so when it looks wrong.
            self.get_logger().info(
                'Anchored ground truth to the estimate at '
                f'({estimate[0]:.2f}, {estimate[1]:.2f}, {estimate[2]:.2f}) '
                f'at t={stamp:.3f}'
            )
        truth_aligned = compose_pose(
            self.estimate_origin, relative_pose(self.truth_origin, truth)
        )
        self.metrics.add(stamp, estimate, truth_aligned)
        self.matched += 1
        if (
            self.evaluation_distance > 0.0
            and self.metrics.truth_distance >= self.evaluation_distance
        ):
            self.window_closed = True
            self.get_logger().info(
                f'evaluation window closed at {self.metrics.truth_distance:.2f} m '
                f'of ground truth travel over {self.matched} matched samples'
            )

        error = Float32()
        error.data = float(
            math.hypot(estimate[0] - truth_aligned[0], estimate[1] - truth_aligned[1])
        )
        self.error_pub.publish(error)
        self.trajectory_rows.append((stamp, estimate, truth_aligned))

        if self.matched % self.path_stride:
            return
        self._append_path(self.estimate_path, stamp_msg, estimate)
        self._append_path(self.truth_path, stamp_msg, truth_aligned)
        self.estimate_path.header.stamp = stamp_msg
        self.truth_path.header.stamp = stamp_msg
        self.estimate_path_pub.publish(self.estimate_path)
        self.truth_path_pub.publish(self.truth_path)

    def _append_path(self, path: Path, stamp_msg, pose):
        path.poses.append(_pose_stamped(stamp_msg, self.map_frame, *pose))
        if len(path.poses) > self.max_path_poses:
            del path.poses[0]

    def _report(self):
        summary = self.metrics.summary()
        if not summary.get('samples'):
            self._report_no_match()
            return
        self.get_logger().info(format_summary(summary))
        # Written every report as well as on shutdown, because a run that is
        # killed rather than stopped still has to leave usable trajectories.
        self.write_trajectories(quiet=True)

    def _report_no_match(self):
        state = (
            f'estimates queued={len(self.pending)}, '
            f'truth buffered={len(self.truth_buffer)}, '
            f'stale drops={self.dropped_stale}'
        )
        if not self.pending or not self.truth_buffer:
            self.get_logger().warning(f'no matched samples yet ({state})')
            return
        offset = self.pending[-1][0] - self.truth_buffer[-1].stamp
        if abs(offset) <= max(self.match_tolerance, 1.0):
            self.get_logger().warning(f'no matched samples yet ({state})')
            return
        # Both streams are alive but their headers disagree by far more than
        # any transport delay, so they are not on the same clock.
        self.get_logger().error(
            f'estimate and ground truth stamps differ by {offset:+.1f}s, so no '
            f'sample can be matched. The CARLA bridge is stamping these two '
            f'streams from different clocks; compare the header stamps of the '
            f'estimate and truth topics before trusting any run ({state})'
        )

    def write_trajectories(self, quiet: bool = False):
        """Write TUM-format trajectories so evo can plot and re-score the run."""
        if not self.output_directory or not self.trajectory_rows:
            return
        directory = os.path.expanduser(self.output_directory)
        os.makedirs(directory, exist_ok=True)
        estimate_file = os.path.join(directory, 'estimate.tum')
        truth_file = os.path.join(directory, 'truth.tum')
        with open(estimate_file, 'w') as estimate, open(truth_file, 'w') as truth:
            for stamp, estimate_pose, truth_pose in self.trajectory_rows:
                estimate.write(_tum_line(stamp, estimate_pose))
                truth.write(_tum_line(stamp, truth_pose))
        summary_file = os.path.join(directory, 'summary.txt')
        with open(summary_file, 'w') as handle:
            for key, value in self.metrics.summary().items():
                handle.write(f'{key}: {value}\n')
            handle.write(f'applied_clock_offset_sec: {self.clock_offset}\n')
            # The conditions the score was taken under, beside the score.
            handle.write(f'anchor_stamp: {self.anchor_stamp}\n')
            handle.write(
                'anchor_lag_sec: '
                f'{self.anchor_stamp - self.first_estimate_stamp}\n')
            handle.write(f'window_closed: {int(self.window_closed)}\n')
            handle.write(
                f'evaluation_distance_m: {self.evaluation_distance}\n')
        if not quiet:
            self.get_logger().info(f'Wrote trajectories and summary to {directory}')


def _tum_line(stamp: float, pose) -> str:
    yaw = wrap_pi(pose[2])
    return (
        f'{stamp:.6f} {pose[0]:.6f} {pose[1]:.6f} 0.000000 '
        f'0.000000 0.000000 {math.sin(0.5 * yaw):.6f} {math.cos(0.5 * yaw):.6f}\n'
    )


def main():
    rclpy.init()
    node = OdometryEvaluator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # A second interrupt arrives here during launch shutdown, and it must
        # not cost the run its trajectories.
        try:
            node.write_trajectories()
        except KeyboardInterrupt:
            node.write_trajectories()
        node.get_logger().info(format_summary(node.metrics.summary()))
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
