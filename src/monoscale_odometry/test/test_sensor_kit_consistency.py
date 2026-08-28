"""Derives the camera extrinsics from the CARLA sensor kit independently of the
estimator configuration, and checks the two agree.

The kit lives in ioniq_carla_bridge and the estimator configuration lives here,
which is exactly why this is checked. The kit spawns cameras with CARLA poses
while the estimator consumes ROS base_link-from-optical matrices; if someone
edits one package and not the other, the odometry stays plausible and silently
wrong.

Nothing in the simulator publishes a transform tree, so `mount_from_tf` in the
odometry node has never had one to read and every replayed number in this stack
has been taken against the configuration below. Until that changes this test is
the only thing holding the two files together.

Skipped rather than failed where the kit cannot be found: this package builds on
a vehicle that has no CARLA bridge installed.
"""

import math
import os

import numpy as np
import pytest
import yaml

CONFIG = os.path.join(os.path.dirname(__file__), '..', 'config')
KIT_PACKAGE = 'ioniq_carla_bridge'
# Which kit link each configured camera is. The fisheye is assembled by the
# bridge from two faces that differ only in yaw; the assembled camera sits where
# the vio tap does, which is where the odometry thinks the eye is.
LINKS = {'front': 'front_camera_vio_link', 'rear': 'rear_camera_vio_link'}


def _load(name):
    with open(os.path.join(CONFIG, name), 'r') as handle:
        return yaml.safe_load(handle)


def _load_kit(name):
    """Read from the kit package, whether built, beside us, or under sim/."""
    candidates = []
    try:
        from ament_index_python.packages import get_package_share_directory

        candidates.append(os.path.join(get_package_share_directory(KIT_PACKAGE), 'config'))
    except Exception:
        pass
    root = os.path.join(os.path.dirname(__file__), '..', '..')
    candidates.append(os.path.join(root, KIT_PACKAGE, 'config'))
    candidates.append(os.path.join(root, '..', 'sim', KIT_PACKAGE, 'config'))
    candidates.append(
        os.path.join('/home/i/ros2_ws/hero-release/sim', KIT_PACKAGE, 'config'))
    for directory in candidates:
        path = os.path.join(directory, name)
        if os.path.exists(path):
            with open(path, 'r') as handle:
                return yaml.safe_load(handle)
    pytest.skip(f'{name} not found in {KIT_PACKAGE}; tried {candidates}')


def _carla_rotation_matrix(roll, pitch, yaw):
    """Columns are the sensor axes in the parent frame, in CARLA conventions."""
    c_y, s_y = math.cos(yaw), math.sin(yaw)
    c_p, s_p = math.cos(pitch), math.sin(pitch)
    c_r, s_r = math.cos(roll), math.sin(roll)
    return np.array(
        [
            [c_p * c_y, c_y * s_p * s_r - s_y * c_r, -c_y * s_p * c_r - s_y * s_r],
            [s_y * c_p, s_y * s_p * s_r + c_y * c_r, -s_y * s_p * c_r + c_y * s_r],
            [s_p, -c_p * s_r, c_p * c_r],
        ]
    )


def _carla_to_ros(vector):
    return np.array([vector[0], -vector[1], vector[2]])


def _base_from_optical(pose):
    axes = _carla_rotation_matrix(pose['roll'], pose['pitch'], pose['yaw'])
    forward, right, up = axes[:, 0], axes[:, 1], axes[:, 2]
    # ROS optical frame: x right, y down, z forward.
    return np.column_stack(
        (_carla_to_ros(right), _carla_to_ros(-up), _carla_to_ros(forward))
    )


@pytest.mark.parametrize('camera', ['front', 'rear'])
def test_camera_extrinsics_match_the_sensor_kit(camera):
    calibration = _load_kit('sensor_kit_calibration.yaml')
    kit = calibration['sensor_kit_base_link'][LINKS[camera]]
    params = _load('vision_fisheye.param.yaml')['/**']['ros__parameters'][camera]

    expected_rotation = np.asarray(
        params['rotation_base_from_camera'], dtype=np.float64
    ).reshape(3, 3)
    expected_translation = np.asarray(
        params['translation_base_from_camera'], dtype=np.float64
    )

    assert _base_from_optical(kit) == pytest.approx(expected_rotation, abs=1e-6)

    # The estimator needs the camera height above the road, which is the
    # mounting height plus however far CARLA holds the actor origin off it.
    ground_offset = calibration['actor_origin_height_m']
    # x and y carry no offset: both files are base_link and the bridge does the
    # conversion to CARLA's vehicle-centre frame when it spawns. Adding one
    # here on 08-09 drove the front camera into the engine bay.
    mounted = _carla_to_ros(np.array([kit['x'], kit['y'], kit['z']]))
    assert mounted[:2] == pytest.approx(expected_translation[:2], abs=1e-9)
    assert mounted[2] + ground_offset == pytest.approx(
        expected_translation[2], abs=1e-9
    )


def test_the_assembled_fisheye_sits_where_its_faces_do():
    """The bridge renders two pinhole faces and publishes one image. If the
    faces ever move off the tap the odometry is configured against, the eye the
    odometry projects from is not the eye that took the picture -- and nothing
    downstream sees two cameras, so nothing else would notice."""
    kit = _load_kit('sensor_kit_calibration.yaml')['sensor_kit_base_link']
    for camera, tap in LINKS.items():
        for face in ('a', 'b'):
            link = f'{camera}_camera_fish_{face}_link'
            if link not in kit:
                continue
            for axis in ('x', 'y', 'z', 'roll', 'pitch'):
                assert kit[link][axis] == pytest.approx(kit[tap][axis], abs=1e-9), (
                    f'{link}.{axis} has drifted off {tap}')


def test_the_two_cameras_are_a_vehicle_length_apart():
    """Not a measurement: the kit's own note derives the front mount as the rear
    one plus the vehicle's length, so this is arithmetic that must hold."""
    kit = _load_kit('sensor_kit_calibration.yaml')['sensor_kit_base_link']
    separation = kit[LINKS['front']]['x'] - kit[LINKS['rear']]['x']
    assert separation == pytest.approx(4.514, abs=1e-9)
