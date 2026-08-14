"""Publishes the exact CARLA ego transform for evaluation.

Nothing in the vision stack subscribes to this. It exists so runs can be
scored, which is the whole reason the work moved off Isaac Sim.

Reading the transform through the CARLA API keeps the sensor kit free of a
GNSS pseudo sensor that only ever existed as a ground truth tap, so the kit
stays a faithful copy of the Isaac rig.
"""

import math
from typing import Optional

import carla
from geometry_msgs.msg import PoseWithCovarianceStamped
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy


def _quaternion_from_carla_rotation(rotation):
    """CARLA is left handed with Y to the right, ROS is right handed."""
    roll = math.radians(rotation.roll)
    pitch = -math.radians(rotation.pitch)
    yaw = -math.radians(rotation.yaw)
    cr, sr = math.cos(0.5 * roll), math.sin(0.5 * roll)
    cp, sp = math.cos(0.5 * pitch), math.sin(0.5 * pitch)
    cy, sy = math.cos(0.5 * yaw), math.sin(0.5 * yaw)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


class CarlaGroundTruth(Node):
    def __init__(self, **kwargs):
        super().__init__('carla_ground_truth', **kwargs)
        declare = self.declare_parameter
        declare('host', 'localhost')
        declare('port', 2000)
        declare('timeout_sec', 30.0)
        declare('role_name', 'ego_vehicle')
        declare('topic', '/carla/ground_truth/pose')
        declare('frame_id', 'map')
        declare('poll_rate_hz', 200.0)

        get = self.get_parameter
        self.role_name = str(get('role_name').value)
        self.frame_id = str(get('frame_id').value)

        client = carla.Client(str(get('host').value), int(get('port').value))
        client.set_timeout(float(get('timeout_sec').value))
        self.world = client.get_world()
        self.ego: Optional[carla.Actor] = None
        self.last_frame = -1

        self.publisher = self.create_publisher(
            PoseWithCovarianceStamped,
            str(get('topic').value),
            QoSProfile(depth=50, reliability=ReliabilityPolicy.BEST_EFFORT),
        )
        rate = max(float(get('poll_rate_hz').value), 1.0)
        self.create_timer(1.0 / rate, self._publish)

        # Without sim time the stamps come off the wall clock while every
        # sensor is on the bridge's, and nothing downstream says so -- the
        # scores just come out wrong.
        if not self.get_parameter('use_sim_time').value:
            self.get_logger().error(
                'use_sim_time is false; truth stamps will not match the '
                'sensors and any score taken against them is meaningless'
            )

    def _find_ego(self) -> Optional[carla.Actor]:
        for actor in self.world.get_actors().filter('vehicle.*'):
            if actor.attributes.get('role_name') == self.role_name:
                return actor
        return None

    def _publish(self):
        if self.ego is None or not self.ego.is_alive:
            self.ego = self._find_ego()
            if self.ego is None:
                self.get_logger().warning(
                    f'no vehicle with role_name "{self.role_name}" yet',
                    throttle_duration_sec=5.0,
                )
                return
            self.get_logger().info(f'tracking ego actor {self.ego.id}')

        snapshot = self.world.get_snapshot()
        # The bridge drives the clock; only publish once per simulation frame.
        if snapshot.frame == self.last_frame:
            return
        self.last_frame = snapshot.frame

        transform = self.ego.get_transform()
        message = PoseWithCovarianceStamped()
        message.header.frame_id = self.frame_id
        # The snapshot's elapsed_seconds counts from the server's episode, but
        # the bridge stamps every sensor and /clock with GameTime, which starts
        # when the bridge does. Those origins differ by however long the server
        # was up first -- 6.9 to 7.3 s across the recorded bags. Scored against
        # that, straight runs merely shift, but turning ones come out with a
        # 58 deg heading error. use_sim_time puts this on the bridge's clock.
        message.header.stamp = self.get_clock().now().to_msg()
        message.pose.pose.position.x = transform.location.x
        message.pose.pose.position.y = -transform.location.y
        message.pose.pose.position.z = transform.location.z
        x, y, z, w = _quaternion_from_carla_rotation(transform.rotation)
        message.pose.pose.orientation.x = x
        message.pose.pose.orientation.y = y
        message.pose.pose.orientation.z = z
        message.pose.pose.orientation.w = w
        self.publisher.publish(message)


def main():
    rclpy.init()
    node = CarlaGroundTruth()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
