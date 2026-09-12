#!/usr/bin/env python3
"""Exit once the simulator has advanced the requested number of seconds.

The recording used to be bounded by a wall-clock timeout computed from the
throttle ratio, so it always ran for the whole budget even when the drive was
long finished -- 495 s for 32 s of simulation. This watches /clock instead and
returns as soon as the sim has covered the ground, leaving the wall budget as
the safety net it was meant to be.

    wait_sim_seconds.py <sim_seconds> <wall_ceiling_seconds>
"""
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock


def main():
    target = float(sys.argv[1])
    ceiling = float(sys.argv[2])

    rclpy.init()
    node = Node('wait_sim_seconds')
    qos = QoSProfile(depth=10)
    qos.reliability = ReliabilityPolicy.BEST_EFFORT
    state = {'first': None, 'last': None}

    def on_clock(msg):
        now = msg.clock.sec + msg.clock.nanosec * 1e-9
        if state['first'] is None:
            state['first'] = now
        state['last'] = now

    node.create_subscription(Clock, '/clock', on_clock, qos)

    started = time.time()
    covered = 0.0
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.2)
        if state['first'] is not None:
            covered = state['last'] - state['first']
            if covered >= target:
                break
        if time.time() - started > ceiling:
            print(f'wall ceiling reached with {covered:.1f}s of sim time')
            break
    else:
        covered = 0.0

    wall = time.time() - started
    print(f'{covered:.1f}s of sim time in {wall:.0f}s wall '
          f'({wall / max(covered, 1e-6):.1f}x)')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
