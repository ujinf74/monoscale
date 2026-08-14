"""Scores the vision occupancy grid against the LiDAR reference grid."""

import os
from typing import Optional

from nav_msgs.msg import OccupancyGrid, Odometry
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from .grid_evaluation import compare_grids
from .grid_evaluation import format_grid_summary
from .grid_evaluation import grid_from_message
from .grid_evaluation import ReferenceAccumulator


class GridEvaluator(Node):
    def __init__(self, **kwargs):
        super().__init__('grid_evaluator', **kwargs)
        declare = self.declare_parameter
        declare('candidate_topic', '/vision/occupancy_grid')
        declare('reference_topic', '/reference/occupancy_grid')
        declare('pose_topic', '/localization/kinematic_state')
        # The cameras see a fraction of the LiDAR's range, so scoring the whole
        # reference footprint would mostly measure that difference.
        declare('max_range_m', 15.0)
        declare('report_period_sec', 5.0)
        declare('output_directory', '')

        get = self.get_parameter
        self.max_range = float(get('max_range_m').value)
        self.output_directory = str(get('output_directory').value)
        self.candidate: Optional[OccupancyGrid] = None
        self.reference: Optional[ReferenceAccumulator] = None
        self.position = (0.0, 0.0)
        self.latest_summary = {}

        latched = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        volatile = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(
            OccupancyGrid, str(get('candidate_topic').value), self._on_candidate, latched
        )
        self.create_subscription(
            OccupancyGrid, str(get('reference_topic').value), self._on_reference, volatile
        )
        self.create_subscription(
            Odometry, str(get('pose_topic').value), self._on_pose, volatile
        )
        self.create_timer(max(float(get('report_period_sec').value), 1.0), self._report)

    def _on_candidate(self, msg):
        self.candidate = msg

    def _on_reference(self, msg):
        if self.candidate is None:
            return
        if self.reference is None:
            # The accumulator borrows the vision grid's geometry so the two
            # can be compared cell for cell.
            self.reference = ReferenceAccumulator(
                grid_from_message(self.candidate), self.max_range
            )
        self.reference.add(grid_from_message(msg), self.position)

    def _on_pose(self, msg):
        self.position = (msg.pose.pose.position.x, msg.pose.pose.position.y)

    def _report(self):
        if self.candidate is None or self.reference is None:
            self.get_logger().warning(
                'waiting for grids '
                f'(vision={self.candidate is not None}, '
                f'reference={self.reference is not None})',
                throttle_duration_sec=10.0,
            )
            return
        self.latest_summary = compare_grids(
            grid_from_message(self.candidate), self.reference.as_grid()
        )
        self.get_logger().info(format_grid_summary(self.latest_summary))
        self.write_summary(quiet=True)

    def write_summary(self, quiet: bool = False):
        if not self.output_directory or not self.latest_summary:
            return
        directory = os.path.expanduser(self.output_directory)
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, 'grid_summary.txt')
        with open(path, 'w') as handle:
            for key, value in self.latest_summary.items():
                handle.write(f'{key}: {value}\n')
        if not quiet:
            self.get_logger().info(f'Wrote grid summary to {path}')


def main():
    rclpy.init()
    node = GridEvaluator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.write_summary()
        except KeyboardInterrupt:
            node.write_summary()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
