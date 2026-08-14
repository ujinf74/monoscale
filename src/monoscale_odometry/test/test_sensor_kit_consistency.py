"""Derives the camera extrinsics from the CARLA sensor kit independently of
the estimator configuration, and checks the two agree.

The kit lives in ioniq_carla_bridge and the estimator configuration lives here,
which is exactly why this is checked. The kit spawns cameras with CARLA poses
while the estimator consumes ROS base_link-from-optical matrices; if someone
edits one package and not the other, the odometry stays plausible and silently
wrong.
"""

import math
import os

import numpy as np
import pytest
import yaml

CONFIG = os.path.join(os.path.dirname(__file__), '..', 'config')
KIT_PACKAGE = 'ioniq_carla_bridge'


def _load(name):
    with open(os.path.join(CONFIG, name), 'r') as handle:
        return yaml.safe_load(handle)


def _load_kit(name):
    """Read from the kit package, whether built or straight from source."""
    candidates = []
    try:
        from ament_index_python.packages import get_package_share_directory

        candidates.append(os.path.join(get_package_share_directory(KIT_PACKAGE), 'config'))
    except Exception:
        pass
    # From source: the kit package sits beside this one in a development
    # checkout, and under sim/ in the deployment tree, where everything that
    # only exists for CARLA is kept apart from what rides on the vehicle.
    root = os.path.join(os.path.dirname(__file__), '..', '..')
    candidates.append(os.path.join(root, KIT_PACKAGE, 'config'))
    candidates.append(os.path.join(root, '..', 'sim', KIT_PACKAGE, 'config'))
    for directory in candidates:
        path = os.path.join(directory, name)
        if os.path.exists(path):
            with open(path, 'r') as handle:
                return yaml.safe_load(handle)
    raise AssertionError(f'{name} not found in {KIT_PACKAGE}; tried {candidates}')


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


@pytest.mark.parametrize(
    'kit_link,camera',
    [
        ('front_camera_vio_link', 'front'),
        ('rear_camera_vio_link', 'rear'),
    ],
)
def test_camera_extrinsics_match_the_sensor_kit(kit_link, camera):
    kit = _load_kit('sensor_kit_calibration.yaml')['sensor_kit_base_link'][kit_link]
    params = _load('vision_only.param.yaml')['/**']['ros__parameters'][camera]

    expected_rotation = np.asarray(
        params['rotation_base_from_camera'], dtype=np.float64
    ).reshape(3, 3)
    expected_translation = np.asarray(
        params['translation_base_from_camera'], dtype=np.float64
    )

    assert _base_from_optical(kit) == pytest.approx(expected_rotation, abs=1e-6)

    # The estimator needs the camera height above the road, which is the
    # mounting height plus however far CARLA holds the actor origin off it.
    calibration = _load_kit('sensor_kit_calibration.yaml')
    ground_offset = calibration['actor_origin_height_m']
    # x and y carry no offset: both files are base_link and the bridge does the
    # conversion to CARLA's vehicle-centre frame when it spawns. Adding one
    # here on 08-09 drove the front camera into the engine bay.
    mounted = _carla_to_ros(np.array([kit['x'], kit['y'], kit['z']]))
    assert mounted[:2] == pytest.approx(expected_translation[:2], abs=1e-9)
    assert mounted[2] + ground_offset == pytest.approx(
        expected_translation[2], abs=1e-9
    )


@pytest.mark.parametrize('camera', ['front', 'rear'])
def test_intrinsics_match_the_spawned_field_of_view(camera):
    # The mapping the measurement drives actually spawn. sensor_mapping_vio
    # is an older kit that still carries the 93.7 degree lens, and checking
    # against it only says the two files were edited together once.
    mapping = _load_kit('sensor_mapping_lidar_map.yaml')['sensor_mappings'][
        f'{camera}_camera_vio_link'
    ]
    params = _load('vision_only.param.yaml')['/**']['ros__parameters'][camera]
    spawned = mapping['parameters']

    # calibration_width is the width k is quoted at, not the width the camera
    # is spawned at: consumers scale k by the ratio between the two, so the
    # kit can render 640 against a 960 calibration and still be consistent.
    # What has to match is the field of view and the aspect ratio.
    width = params['calibration_width']
    height = params['calibration_height']
    focal = width / (2.0 * math.tan(math.radians(spawned['fov']) / 2.0))
    k = np.asarray(params['k'], dtype=np.float64).reshape(3, 3)

    assert (spawned['image_size_x'] / spawned['image_size_y']
            == pytest.approx(width / height))
    assert params['horizontal_fov_deg'] == pytest.approx(spawned['fov'])
    # The FOV is rounded to six decimals in the sensor kit, which lands within
    # a ten-thousandth of a pixel of the nominal focal length.
    assert k[0, 0] == pytest.approx(focal, abs=1e-3)
    assert k[1, 1] == pytest.approx(focal, abs=1e-3)
    assert k[0, 2] == pytest.approx(width / 2.0)
    assert k[1, 2] == pytest.approx(height / 2.0)


@pytest.mark.parametrize('camera', ['front', 'rear'])
def test_vio_tap_sits_exactly_where_the_primary_camera_does(camera):
    kit = _load_kit('sensor_kit_calibration.yaml')['sensor_kit_base_link']
    primary = kit[f'{camera}_camera_link']
    tap = kit[f'{camera}_camera_vio_link']

    assert tap == primary


@pytest.mark.parametrize('camera', ['front', 'rear'])
def test_vio_tap_keeps_the_primary_lens(camera):
    mappings = _load_kit('sensor_mapping_vio.yaml')['sensor_mappings']
    primary = mappings[f'{camera}_camera_link']['parameters']
    tap = mappings[f'{camera}_camera_vio_link']['parameters']

    assert tap['fov'] == pytest.approx(primary['fov'])
    # Same aspect ratio, or the field of view would differ vertically.
    assert tap['image_size_x'] / tap['image_size_y'] == pytest.approx(
        primary['image_size_x'] / primary['image_size_y']
    )


def test_wheelbase_agrees_between_the_kit_and_the_evaluator():
    mapping = _load_kit('sensor_mapping_vio.yaml')['vehicle_config']['wheelbase']
    evaluation = _load('evaluation.param.yaml')['/**']['ros__parameters']['wheel_base']

    assert mapping == pytest.approx(evaluation)
