"""An occupancy grid built from what the odometry saw on the road.

The estimator publishes the points its solve already produced, labelled: a
feature that survived the ground registration is road, one whose projection
slid against the travel is standing on it. Both arrive in the map frame with
the camera position they were seen from, which is everything this node needs
-- no extrinsics, no transform tree, and no opinion about how the pose was
arrived at.

Keeping the grid here rather than inside the estimator is what lets the map
be re-tuned, or replaced, without re-running the odometry: the same recorded
cloud plays back into whatever accumulates it.
"""

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2

from .occupancy import LogOddsGrid

GROUND = 0.0
POINT_STEP = 24


class OccupancyGridMap(Node):
    def __init__(self):
        super().__init__('occupancy_grid_map')
        declare = self.declare_parameter
        declare('ground_points_topic', '/perception/ground_points')
        declare('occupancy_topic', '/perception/occupancy_grid_map')
        declare('map_frame', 'map')
        declare('map_resolution', 0.1)
        declare('map_width', 600)
        declare('map_height', 600)
        declare('map_origin_x', -30.0)
        declare('map_origin_y', -30.0)
        declare('free_log_odds_update', 0.45)
        declare('occupied_log_odds_update', 0.9)
        declare('occupied_probability_threshold', 0.65)
        declare('free_probability_threshold', 0.35)
        declare('obstacle_inflation_radius_m', 0.25)
        # Carving a ray per ground point costs more than it adds; every fourth
        # one sweeps the same free space.
        declare('free_ray_stride', 4)
        declare('occupancy_publish_rate_hz', 5.0)
        declare('input_queue_depth', 50)
        get = self.get_parameter

        self.map_frame = str(get('map_frame').value)
        self.grid = LogOddsGrid(
            float(get('map_resolution').value),
            int(get('map_width').value),
            int(get('map_height').value),
            float(get('map_origin_x').value),
            float(get('map_origin_y').value),
            float(get('free_log_odds_update').value),
            float(get('occupied_log_odds_update').value),
            float(get('occupied_probability_threshold').value),
            float(get('free_probability_threshold').value),
        )
        self.inflation_radius = float(get('obstacle_inflation_radius_m').value)
        self.free_ray_stride = max(int(get('free_ray_stride').value), 1)
        self.last_stamp = None
        self.clouds = 0
        self.points = 0
        self.obstacles = 0

        self.create_subscription(
            PointCloud2,
            str(get('ground_points_topic').value),
            self._on_points,
            QoSProfile(depth=max(int(get('input_queue_depth').value), 1),
                       reliability=ReliabilityPolicy.RELIABLE),
        )
        self.map_pub = self.create_publisher(
            OccupancyGrid,
            str(get('occupancy_topic').value),
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                       durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        rate = max(float(get('occupancy_publish_rate_hz').value), 0.1)
        self.create_timer(1.0 / rate, self._publish_grid)
        # Said out loud while it runs, not only on the way out: a node killed
        # by a signal it never sees reports nothing at all.
        self.create_timer(2.0, self._report)

    def _on_points(self, msg: PointCloud2):
        if msg.point_step != POINT_STEP:
            self.get_logger().error(
                f'Expected {POINT_STEP} byte points, got {msg.point_step}',
                throttle_duration_sec=5.0)
            return
        rows = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, 6)
        self.last_stamp = msg.header.stamp
        self.clouds += 1
        self.points += len(rows)
        ground = rows[:, 3] == GROUND
        # Free space is carved from where the camera stood to where the road
        # was seen; an obstacle marks its own cell and nothing else.
        for row in rows[ground][::self.free_ray_stride]:
            self.grid.integrate_ray(row[4:6], row[0:2], occupied=False)
        self.obstacles += int(np.count_nonzero(~ground))
        for row in rows[~ground]:
            self.grid.integrate_point(row[0:2])

    def _report(self):
        if self.clouds:
            self.get_logger().info(
                f'occupancy: clouds={self.clouds} points={self.points} '
                f'obstacles={self.obstacles}')

    def _publish_grid(self):
        if self.last_stamp is None:
            return
        msg = OccupancyGrid()
        msg.header.stamp = self.last_stamp
        msg.header.frame_id = self.map_frame
        msg.info.resolution = self.grid.resolution
        msg.info.width = self.grid.width
        msg.info.height = self.grid.height
        msg.info.origin.position.x = self.grid.origin_x
        msg.info.origin.position.y = self.grid.origin_y
        msg.info.origin.orientation.w = 1.0
        values = self.grid.message_values(self.inflation_radius)
        msg.data = values.reshape(-1).astype(np.int8).tolist()
        self.map_pub.publish(msg)


def main():
    rclpy.init()
    node = OccupancyGridMap()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
