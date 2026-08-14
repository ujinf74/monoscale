"""Drives the CARLA ego through a fixed manoeuvre so runs stay comparable.

Vision odometry has to be scored on the same motion every time, which manual
driving cannot provide. The node speaks the actuation and gear contract the
CARLA bridge already subscribes to, so it needs no CARLA API access.
"""

import math
from typing import List, Optional

from autoware_vehicle_msgs.msg import GearCommand, VelocityReport
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from tier4_vehicle_msgs.msg import ActuationCommandStamped

GEAR_BY_NAME = {
    'park': GearCommand.PARK,
    'neutral': GearCommand.NEUTRAL,
    'drive': GearCommand.DRIVE,
    'reverse': GearCommand.REVERSE,
}


class Segment:
    def __init__(self, duration: float, gear: str, speed: float, steer: float):
        key = gear.strip().lower()
        if key not in GEAR_BY_NAME:
            raise ValueError(f'unknown gear "{gear}"; expected one of {sorted(GEAR_BY_NAME)}')
        self.duration = max(float(duration), 0.0)
        self.gear_name = key
        self.gear = GEAR_BY_NAME[key]
        self.speed = abs(float(speed))
        self.steer = float(steer)

    @property
    def moving(self) -> bool:
        return self.gear in (GearCommand.DRIVE, GearCommand.REVERSE) and self.speed > 0.0


class EgoPilot(Node):
    def __init__(self):
        super().__init__('ego_pilot')
        declare = self.declare_parameter
        declare('actuation_topic', '/control/command/actuation_cmd')
        declare('gear_topic', '/control/command/gear_cmd')
        declare('velocity_topic', '/vehicle/status/velocity_status')
        declare('control_rate_hz', 50.0)
        declare('speed_gain', 0.6)
        declare('brake_gain', 0.5)
        declare('max_throttle', 0.5)
        declare('start_delay_sec', 3.0)
        declare('loop', False)
        declare('segment_durations', [4.0, 7.0, 5.0, 2.0, 7.0, 3.0])
        declare(
            'segment_gears',
            ['drive', 'drive', 'drive', 'park', 'reverse', 'park'],
        )
        declare('segment_speeds', [2.0, 2.0, 2.0, 0.0, 1.5, 0.0])
        declare('segment_steers', [0.0, 0.45, -0.45, 0.0, 0.35, 0.0])

        get = self.get_parameter
        self.speed_gain = float(get('speed_gain').value)
        self.brake_gain = float(get('brake_gain').value)
        self.max_throttle = float(get('max_throttle').value)
        self.start_delay = max(float(get('start_delay_sec').value), 0.0)
        self.loop = bool(get('loop').value)
        self.segments = self._build_segments()
        self.total_duration = sum(segment.duration for segment in self.segments)

        self.speed = 0.0
        self.start_time: Optional[float] = None
        self.finished = False
        self.active_index = -1

        sensor = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(
            VelocityReport, str(get('velocity_topic').value), self._on_velocity, sensor
        )
        self.actuation_pub = self.create_publisher(
            ActuationCommandStamped, str(get('actuation_topic').value), 1
        )
        self.gear_pub = self.create_publisher(
            GearCommand, str(get('gear_topic').value), 1
        )
        rate = max(float(get('control_rate_hz').value), 1.0)
        self.create_timer(1.0 / rate, self._tick)
        self.get_logger().info(
            f'{len(self.segments)} segments, {self.total_duration:.1f}s manoeuvre, '
            f'starting after {self.start_delay:.1f}s'
        )

    def _build_segments(self) -> List[Segment]:
        get = self.get_parameter
        durations = list(get('segment_durations').value)
        gears = list(get('segment_gears').value)
        speeds = list(get('segment_speeds').value)
        steers = list(get('segment_steers').value)
        lengths = {len(durations), len(gears), len(speeds), len(steers)}
        if len(lengths) != 1:
            raise ValueError(
                'segment_durations, segment_gears, segment_speeds and segment_steers '
                'must have the same length'
            )
        return [Segment(*values) for values in zip(durations, gears, speeds, steers)]

    def _on_velocity(self, msg: VelocityReport):
        self.speed = abs(float(msg.longitudinal_velocity))

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _segment_at(self, elapsed: float) -> Optional[Segment]:
        for index, segment in enumerate(self.segments):
            if elapsed < segment.duration:
                if index != self.active_index:
                    self.active_index = index
                    self.get_logger().info(
                        f'segment {index}: {segment.gear_name} '
                        f'{segment.speed:.1f} m/s steer {segment.steer:+.2f}'
                    )
                return segment
            elapsed -= segment.duration
        return None

    def _tick(self):
        now = self._now()
        if self.start_time is None:
            self.start_time = now
        elapsed = now - self.start_time - self.start_delay
        if elapsed < 0.0:
            self._publish(GearCommand.PARK, 0.0, 0.0, 1.0)
            return

        if self.loop and self.total_duration > 0.0:
            elapsed = math.fmod(elapsed, self.total_duration)
        segment = self._segment_at(elapsed)
        if segment is None:
            if not self.finished:
                self.finished = True
                self.get_logger().info('manoeuvre complete; holding the vehicle')
            self._publish(GearCommand.PARK, 0.0, 0.0, 1.0)
            return

        if not segment.moving:
            self._publish(segment.gear, segment.steer, 0.0, 1.0)
            return
        error = segment.speed - self.speed
        throttle = max(min(self.speed_gain * error, self.max_throttle), 0.0)
        brake = max(min(-self.brake_gain * error, 1.0), 0.0)
        self._publish(segment.gear, segment.steer, throttle, brake)

    def _publish(self, gear: int, steer: float, throttle: float, brake: float):
        stamp = self.get_clock().now().to_msg()
        gear_msg = GearCommand()
        gear_msg.stamp = stamp
        gear_msg.command = gear
        self.gear_pub.publish(gear_msg)

        command = ActuationCommandStamped()
        command.header.stamp = stamp
        command.header.frame_id = 'base_link'
        command.actuation.accel_cmd = float(throttle)
        command.actuation.brake_cmd = float(brake)
        command.actuation.steer_cmd = float(max(min(steer, 1.0), -1.0))
        self.actuation_pub.publish(command)


def main():
    rclpy.init()
    node = EgoPilot()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
