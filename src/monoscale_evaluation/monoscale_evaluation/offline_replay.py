"""Score the estimator against a recorded drive, without a ROS graph.

Replaying a bag through live nodes still leaves the answer depending on the
order callbacks happened to fire in, which is enough to move drift by half a
point between two runs of the same configuration. That is larger than most of
the differences worth measuring.

Here the bag is read directly and the estimator's callbacks are invoked in
recorded order, so the same bag and the same parameters give the same number
every time. Nothing about the estimator changes: this drives the same
callbacks the subscriptions would have.
"""

import argparse
import math
from typing import Dict, List, Tuple

import rclpy
import yaml
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.parameter import Parameter
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from sensor_msgs.msg import CameraInfo, Image, Imu
from std_msgs.msg import Float32MultiArray

import rosbag2_py

from monoscale_odometry.odometry_node import VisualOdometryNode
from monoscale_evaluation.evaluation import (
    Pose2,
    TrajectoryMetrics,
    compose_pose,
    format_summary,
    relative_pose,
)


def _flatten(values: Dict, prefix: str = '') -> Dict[str, object]:
    """ROS spells a nested parameter block with dots."""
    flat = {}
    for key, value in values.items():
        name = f'{prefix}{key}'
        if isinstance(value, dict):
            flat.update(_flatten(value, f'{name}.'))
        else:
            flat[name] = value
    return flat


def load_parameters(path: str, overrides: List[str]) -> List[Parameter]:
    with open(path) as handle:
        document = yaml.safe_load(handle)
    values = _flatten(document['/**']['ros__parameters'])
    values['use_sim_time'] = False
    for item in overrides:
        name, _, raw = item.partition('=')
        values[name] = yaml.safe_load(raw)
    return [Parameter(name, value=value) for name, value in values.items()]


def _stamp_seconds(header) -> float:
    return header.stamp.sec + header.stamp.nanosec * 1e-9


def _yaw(orientation) -> float:
    q = orientation
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    )


def read_bag(path: str):
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=path, storage_id='sqlite3'),
        rosbag2_py.ConverterOptions('', ''),
    )
    types = {t.name: get_message(t.type) for t in reader.get_all_topics_and_types()}
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic in types:
            yield topic, deserialize_message(data, types[topic])


def replay(bag: str, parameters: List[Parameter], truth_topic: str, wheel_base: float):
    node = VisualOdometryNode(parameter_overrides=parameters)
    estimates: List[Tuple[float, Tuple[float, float, float]]] = []
    node.odom_pub.publish = lambda msg: estimates.append(
        (
            _stamp_seconds(msg.header),
            (
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                _yaw(msg.pose.pose.orientation),
            ),
        )
    )
    node.pose_pub.publish = lambda msg: None

    get = node.get_parameter
    front_image = str(get('front_image_topic').value)
    rear_image = str(get('rear_image_topic').value)
    front_info = str(get('front_camera_info_topic').value)
    rear_info = str(get('rear_camera_info_topic').value)
    imu_topic = str(get('imu_topic').value)
    # With the C++ front end in play the bag carries tracks instead of images,
    # and the estimator is driven through the callback that reads them. Same
    # recorded order, same determinism.
    prefix = str(get('track_topic_prefix').value)
    front_tracks = f'{prefix}/front' if prefix else None
    rear_tracks = f'{prefix}/rear' if prefix else None

    truth: List[Tuple[float, Tuple[float, float, float]]] = []
    for topic, message in read_bag(bag):
        if topic == front_image and isinstance(message, Image):
            node._on_image('front', message)
        elif topic == rear_image and isinstance(message, Image):
            node._on_image('rear', message)
        elif topic == front_tracks and isinstance(message, Float32MultiArray):
            node._on_tracks('front', message)
        elif topic == rear_tracks and isinstance(message, Float32MultiArray):
            node._on_tracks('rear', message)
        elif topic == imu_topic and isinstance(message, Imu):
            node._on_imu(message)
        elif topic == front_info and isinstance(message, CameraInfo):
            node._on_camera_info('front', message)
        elif topic == rear_info and isinstance(message, CameraInfo):
            node._on_camera_info('rear', message)
        elif topic == truth_topic and isinstance(message, PoseWithCovarianceStamped):
            pose = message.pose.pose
            yaw = _yaw(pose.orientation)
            # The truth reports the vehicle centre; the stack estimates the
            # rear axle, so it is shifted back along the heading.
            truth.append(
                (
                    _stamp_seconds(message.header),
                    (
                        pose.position.x - 0.5 * wheel_base * math.cos(yaw),
                        pose.position.y - 0.5 * wheel_base * math.sin(yaw),
                        yaw,
                    ),
                )
            )
    node.destroy_node()
    return estimates, truth


def score(estimates, truth, evaluation_distance: float, segment: float, tolerance: float,
          skip_distance: float = 0.0):
    """Score a window of the run, optionally starting after a warm-up.

    Starting a run already at speed leaves the anchor map to be built from
    nothing while the ground sweeps past, and the error that costs is spent
    and recovered inside the first ten metres or so. That is a real cost and
    the default keeps it in the score. `skip_distance` moves the window past
    it, which answers the different question of what the estimator does once
    it is running -- worth reporting alongside, never instead.
    """
    metrics = TrajectoryMetrics(segment)
    if not estimates or not truth:
        return metrics
    truth_stamps = [row[0] for row in truth]
    origin = None
    index = 0
    skipped = 0.0
    previous_truth = None
    for stamp, estimate in estimates:
        while index + 1 < len(truth_stamps) and truth_stamps[index + 1] <= stamp:
            index += 1
        nearest = min(
            (index, min(index + 1, len(truth_stamps) - 1)),
            key=lambda i: abs(truth_stamps[i] - stamp),
        )
        if abs(truth_stamps[nearest] - stamp) > tolerance:
            continue
        matched = truth[nearest][1]
        if skip_distance > 0.0 and origin is None:
            if previous_truth is not None:
                skipped += math.hypot(
                    matched[0] - previous_truth[0], matched[1] - previous_truth[1]
                )
            previous_truth = matched
            if skipped < skip_distance:
                continue
        if origin is None:
            origin = (matched, estimate)
        aligned = compose_pose(
            origin[1], relative_pose(Pose2(0.0, *origin[0]), Pose2(0.0, *matched))
        )
        metrics.add(stamp, estimate, aligned)
        if 0.0 < evaluation_distance <= metrics.truth_distance:
            break
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bag')
    parser.add_argument('--params', required=True, help='estimator parameter yaml')
    parser.add_argument(
        '--set', action='append', default=[],
        help='parameter override, name=value, repeatable',
    )
    parser.add_argument('--distance', type=float, default=34.0)
    parser.add_argument('--segment', type=float, default=1.0)
    parser.add_argument('--tolerance', type=float, default=0.05)
    parser.add_argument(
        '--skip-distance', type=float, default=0.0,
        help='metres of warm-up to leave out of the scored window',
    )
    parser.add_argument('--wheel-base', type=float, default=2.8192)
    parser.add_argument('--truth-topic', default='/carla/ground_truth/pose')
    parser.add_argument('--label', default='')
    args = parser.parse_args()

    rclpy.init()
    try:
        estimates, truth = replay(
            args.bag,
            load_parameters(args.params, args.set),
            args.truth_topic,
            args.wheel_base,
        )
    finally:
        rclpy.shutdown()

    summary = score(
        estimates, truth, args.distance, args.segment, args.tolerance,
        args.skip_distance,
    ).summary()
    if args.label:
        print(f'--- {args.label}')
    print(format_summary(summary))


if __name__ == '__main__':
    main()
