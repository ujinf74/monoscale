"""Feeds the estimator a synthetic ground plane rendered through the shipped
CARLA calibration, so the extrinsics, intrinsics and motion estimator are
checked together rather than one at a time.
"""

import math

import cv2
import numpy as np
import pytest
import rclpy
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image, Imu

from monoscale_odometry.odometry_node import VisualOdometryNode
from monoscale_odometry.geometry import pixels_to_ground

WIDTH = 640
HEIGHT = 360
FRONT_ROTATION = [0.0, -0.5, 0.8660254, -1.0, 0.0, 0.0, 0.0, -0.8660254, -0.5]
REAR_ROTATION = [0.0, 0.5, -0.8660254, 1.0, 0.0, 0.0, 0.0, -0.8660254, -0.5]
STEP_X = 0.12
STEP_YAW = 0.0


@pytest.fixture
def estimator():
    rclpy.init()
    node = _make_node()
    yield node
    node.destroy_node()
    rclpy.shutdown()


def _make_node(**extra):
    overrides = [
        Parameter('use_camera_info', Parameter.Type.BOOL, False),
        # These exercise the solve on a single pair, which by definition has no
        # anchor map behind it yet. The warm-up that suppresses exactly that
        # case has its own tests below.
        Parameter('require_map_before_translating', Parameter.Type.BOOL, False),
        # One pair in, one solve out: the adaptive trigger needs several
        # frames of image motion before it fires, which these do not provide.
        Parameter('adaptive_solve_interval', Parameter.Type.BOOL, False),
        Parameter('publish_tf', Parameter.Type.BOOL, False),
        Parameter('processing_width', Parameter.Type.INTEGER, 0),
        Parameter('front.calibration_width', Parameter.Type.INTEGER, WIDTH),
        Parameter('front.calibration_height', Parameter.Type.INTEGER, HEIGHT),
        Parameter('rear.calibration_width', Parameter.Type.INTEGER, WIDTH),
        Parameter('rear.calibration_height', Parameter.Type.INTEGER, HEIGHT),
        Parameter('front.k', Parameter.Type.DOUBLE_ARRAY, _intrinsics().ravel().tolist()),
        Parameter('rear.k', Parameter.Type.DOUBLE_ARRAY, _intrinsics().ravel().tolist()),
        Parameter('front.rotation_base_from_camera', Parameter.Type.DOUBLE_ARRAY, FRONT_ROTATION),
        Parameter('rear.rotation_base_from_camera', Parameter.Type.DOUBLE_ARRAY, REAR_ROTATION),
        Parameter('front.translation_base_from_camera', Parameter.Type.DOUBLE_ARRAY, [3.5, 0.0, 0.89]),
        Parameter('rear.translation_base_from_camera', Parameter.Type.DOUBLE_ARRAY, [-0.82, 0.0, 1.26]),
    ]
    overrides += [Parameter(name, value=value) for name, value in extra.items()]
    return VisualOdometryNode(parameter_overrides=overrides)


def _intrinsics():
    # The CARLA blueprint FOV scaled from 1920x1080 down to the test resolution.
    return np.array(
        [[300.0, 0.0, 320.0], [0.0, 300.0, 180.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )


def _texture(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 255, size=(HEIGHT, WIDTH), dtype=np.uint8)
    return cv2.GaussianBlur(noise, (5, 5), 0)


def _ground_homography(model, step_x: float, step_yaw: float) -> np.ndarray:
    """Pixel motion of the ground plane for a base_link step of (step_x, step_yaw)."""
    source = np.array(
        [[120.0, 250.0], [520.0, 250.0], [120.0, 340.0], [520.0, 340.0]],
        dtype=np.float32,
    )
    ground, valid = pixels_to_ground(source.astype(np.float64), model, 40.0)
    assert valid.all(), 'test pixels must hit the ground plane'

    c = math.cos(step_yaw)
    s = math.sin(step_yaw)
    rotation = np.array([[c, s], [-s, c]], dtype=np.float64)
    moved = (rotation @ (ground - np.array([step_x, 0.0])).T).T

    points = np.column_stack((moved, np.zeros(len(moved))))
    camera = (model.rotation_base_from_camera.T @ (points - model.translation_base_from_camera).T).T
    projected = (model.k @ camera.T).T
    destination = (projected[:, :2] / projected[:, 2:3]).astype(np.float32)
    return cv2.getPerspectiveTransform(source, destination)


def _image(gray: np.ndarray, stamp: float) -> Image:
    msg = Image()
    msg.header.stamp.sec = int(stamp)
    msg.header.stamp.nanosec = int(round((stamp - int(stamp)) * 1e9))
    msg.height = HEIGHT
    msg.width = WIDTH
    msg.encoding = 'bgra8'
    msg.step = WIDTH * 4
    bgra = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGRA)
    msg.data = bgra.reshape(-1).tobytes()
    return msg


def _imu(stamp: float, yaw: float) -> Imu:
    msg = Imu()
    msg.header.stamp.sec = int(stamp)
    msg.header.stamp.nanosec = int(round((stamp - int(stamp)) * 1e9))
    msg.orientation.z = math.sin(0.5 * yaw)
    msg.orientation.w = math.cos(0.5 * yaw)
    return msg


def _drive_one_step(estimator, first: float, step_x: float, step_yaw: float):
    second = first + 1.0 / 30.0
    estimator._on_imu(_imu(first, 0.0))
    estimator._on_imu(_imu(second, step_yaw))

    frames = {}
    for index, name in enumerate(('front', 'rear')):
        texture = _texture(seed=7 + index)
        homography = _ground_homography(
            estimator.cameras[name].model, step_x, step_yaw
        )
        frames[name] = (
            texture,
            cv2.warpPerspective(texture, homography, (WIDTH, HEIGHT)),
        )

    for name in ('front', 'rear'):
        estimator._on_image(name, _image(frames[name][0], first))
    assert estimator.pose.x == pytest.approx(0.0)

    for name in ('front', 'rear'):
        estimator._on_image(name, _image(frames[name][1], second))


def test_forward_step_is_recovered_in_metres(estimator):
    _drive_one_step(estimator, 1.0, STEP_X, STEP_YAW)

    assert estimator.motion_failures == 0
    assert estimator.pose.x == pytest.approx(STEP_X, abs=0.01)
    assert estimator.pose.y == pytest.approx(0.0, abs=0.01)
    assert estimator.pose.yaw == pytest.approx(0.0, abs=0.01)


def test_first_vision_solve_anchors_the_inertial_propagator(estimator):
    assert not estimator.inertial.corrected

    _drive_one_step(estimator, 1.0, STEP_X, STEP_YAW)

    assert estimator.inertial.corrected
    # Whichever filter the node is running is the one that anchors it.
    fused = (estimator.displacement_filter.velocity
             if estimator.displacement_filter is not None
             else estimator.velocity_filter.velocity)
    assert estimator.inertial.velocity == pytest.approx(fused, abs=1e-9)


def test_yawing_step_is_recovered(estimator):
    step_yaw = 0.02

    _drive_one_step(estimator, 2.0, STEP_X, step_yaw)

    assert estimator.motion_failures == 0
    assert estimator.pose.x == pytest.approx(STEP_X, abs=0.02)
    assert estimator.pose.yaw == pytest.approx(step_yaw, abs=0.005)


def test_rejected_vision_still_applies_the_imu_yaw(estimator):
    step_yaw = 0.05

    # 2 m in one frame is past max_translation_per_frame_m, so the vision
    # translation is thrown away while the IMU rotation stays valid.
    _drive_one_step(estimator, 3.0, 2.0, step_yaw)

    assert estimator.motion_failures == 1
    assert estimator.yaw_only_updates == 1
    assert estimator.pose.yaw == pytest.approx(step_yaw, abs=1e-9)
    assert estimator.pose.x == pytest.approx(0.0)


def test_full_resolution_input_still_yields_the_processing_size(estimator):
    estimator.processing_width = 640
    wide = 1920
    tall = 1080
    msg = Image()
    msg.header.stamp.sec = 5
    msg.height = tall
    msg.width = wide
    msg.encoding = 'bgra8'
    msg.step = wide * 4
    msg.data = np.zeros((tall, wide, 4), dtype=np.uint8).reshape(-1).tobytes()

    estimator._on_image('front', msg)

    frame = estimator.pending_frames['front'][-1]
    assert frame.gray.shape == (360, 640)
    # The calibration is declared at 640 wide, so a 1920 wide frame scales K up
    # by three and the resample takes it back down. The intrinsics must end up
    # matching the pixels actually tracked, whatever the input resolution.
    assert estimator.cameras['front'].model.k[0, 0] == pytest.approx(
        _intrinsics()[0, 0]
    )
    assert estimator.cameras['front'].model.k[0, 2] == pytest.approx(
        _intrinsics()[0, 2]
    )


def test_imu_yaw_is_interpolated_between_samples(estimator):
    estimator.imu_max_age = 0.02
    estimator.imu_max_gap = 0.12
    estimator._on_imu(_imu(10.0, 0.0))
    estimator._on_imu(_imu(10.1, 0.2))

    # Half way between two samples 100 ms apart, far outside imu_max_age.
    assert estimator._imu_yaw_at(10.05) == pytest.approx(0.1, abs=1e-9)
    assert estimator._imu_yaw_at(10.0) == pytest.approx(0.0, abs=1e-9)


def test_acceleration_requires_its_independent_deployment_flag(estimator):
    estimator.inertial_use_acceleration = False
    estimator.inertial.correct_velocity([0.0, 0.0])
    first = _imu(40.0, 0.0)
    first.linear_acceleration.z = 9.81
    second = _imu(40.1, 0.0)
    second.linear_acceleration.x = 4.0
    second.linear_acceleration.z = 9.81

    estimator._on_imu(first)
    estimator._on_imu(second)

    assert estimator.velocity_filter.velocity == pytest.approx(
        [0.0, 0.0], abs=1e-9
    )

    estimator.inertial_use_acceleration = True
    third = _imu(40.2, 0.0)
    third.linear_acceleration.x = 4.0
    third.linear_acceleration.z = 9.81
    estimator._on_imu(third)

    assert estimator.velocity_filter.velocity[0] == pytest.approx(0.4)


def test_imu_yaw_refuses_a_gap_that_is_too_wide(estimator):
    estimator.imu_max_age = 0.02
    estimator.imu_max_gap = 0.12
    estimator._on_imu(_imu(20.0, 0.0))
    estimator._on_imu(_imu(20.5, 0.4))

    assert estimator._imu_yaw_at(20.25) is None


def test_interpolation_takes_the_short_way_around_pi(estimator):
    estimator.imu_max_age = 0.02
    estimator.imu_max_gap = 0.12
    estimator._on_imu(_imu(30.0, 3.1))
    estimator._on_imu(_imu(30.1, -3.1))

    yaw = estimator._imu_yaw_at(30.05)

    assert abs(abs(yaw) - math.pi) == pytest.approx(0.0, abs=0.05)


def test_the_first_pair_cannot_move_the_pose_before_the_map_exists():
    """A standing vehicle read 38 cm of travel out of its first few frames."""
    rclpy.init()
    node = _make_node(require_map_before_translating=True)
    try:
        _drive_one_step(node, 1.0, STEP_X, STEP_YAW)

        assert node.pose.x == pytest.approx(0.0, abs=1e-9)
        assert node.pose.y == pytest.approx(0.0, abs=1e-9)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_the_yaw_still_integrates_while_translation_is_held():
    rclpy.init()
    node = _make_node(require_map_before_translating=True)
    try:
        _drive_one_step(node, 1.0, STEP_X, 0.03)

        assert node.pose.x == pytest.approx(0.0, abs=1e-9)
        assert node.pose.yaw == pytest.approx(0.03, abs=0.005)
    finally:
        node.destroy_node()
        rclpy.shutdown()
