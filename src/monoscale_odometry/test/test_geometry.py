import math

import cv2
import numpy as np
import pytest

from monoscale_odometry.geometry import CameraModel
from monoscale_odometry.geometry import estimate_planar_motion
from monoscale_odometry.geometry import estimate_planar_motion_with_yaw
from monoscale_odometry.geometry import pixels_to_ground
from monoscale_odometry.geometry import PlanarMotion
from monoscale_odometry.geometry import triangulate_temporal_points
from monoscale_odometry.geometry import undistort_pixels


def test_pixels_to_ground_for_forward_optical_camera():
    k = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])
    rotation = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    model = CameraModel(k, rotation, np.array([0.0, 0.0, 1.0]))

    points, valid = pixels_to_ground(np.array([[50.0, 100.0]]), model, 20.0)

    assert valid.tolist() == [True]
    assert points[0] == pytest.approx([2.0, 0.0])


def test_planar_motion_maps_current_points_into_previous_frame():
    current = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, -1.0]])
    yaw = 0.1
    rotation = np.array(
        [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]
    )
    previous = (rotation @ current.T).T + np.array([0.4, -0.2])

    result = estimate_planar_motion(previous, current, 0.01, 3, 0.02)

    assert result is not None
    motion, _ = result
    assert motion.x == pytest.approx(0.4, abs=1e-5)
    assert motion.y == pytest.approx(-0.2, abs=1e-5)
    assert motion.yaw == pytest.approx(yaw, abs=1e-5)


def test_planar_motion_with_imu_yaw_rejects_translation_outliers():
    rng = np.random.default_rng(7)
    current = rng.uniform(-5.0, 5.0, size=(80, 2))
    yaw = -0.08
    rotation = np.array(
        [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]
    )
    previous = (rotation @ current.T).T + np.array([0.25, 0.04])
    previous[:15] += rng.uniform(0.5, 2.0, size=(15, 2))

    result = estimate_planar_motion_with_yaw(
        previous, current, yaw, ransac_threshold=0.05, min_inliers=40
    )

    assert result is not None
    motion, inliers = result
    assert motion.x == pytest.approx(0.25, abs=1e-6)
    assert motion.y == pytest.approx(0.04, abs=1e-6)
    assert motion.yaw == pytest.approx(yaw)
    assert np.count_nonzero(inliers) == 65


def test_temporal_triangulation_recovers_metric_obstacle_point():
    k = np.array([[300.0, 0.0, 320.0], [0.0, 300.0, 240.0], [0.0, 0.0, 1.0]])
    rotation = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    translation = np.array([1.0, 0.0, 1.0])
    model = CameraModel(k, rotation, translation)
    point_previous_base = np.array([8.0, 1.0, 1.5])
    motion = PlanarMotion(0.5, 0.0, 0.0, 100, 1.0)

    point_previous_camera = rotation.T @ (point_previous_base - translation)
    point_current_base = point_previous_base - np.array([motion.x, motion.y, 0.0])
    point_current_camera = rotation.T @ (point_current_base - translation)

    def project(point):
        pixel = k @ point
        return pixel[:2] / pixel[2]

    points, valid = triangulate_temporal_points(
        np.array([project(point_previous_camera)]),
        np.array([project(point_current_camera)]),
        model,
        motion,
        0.1,
        0.1,
        0.1,
        30.0,
    )

    assert valid.tolist() == [True]
    assert points[0] == pytest.approx(point_previous_base, abs=1e-5)


def test_equidistant_pixels_are_undistorted_before_geometry():
    k = np.array([[300.0, 0.0, 320.0], [0.0, 300.0, 240.0], [0.0, 0.0, 1.0]])
    distortion = np.array([0.1, -0.02, 0.001, 0.0])
    normalized = np.array([[[0.4, -0.2]]], dtype=np.float64)
    distorted = cv2.fisheye.distortPoints(normalized, k, distortion).reshape(-1, 2)
    model = CameraModel(k, np.eye(3), np.zeros(3), distortion, 'equidistant')

    corrected = undistort_pixels(distorted, model)

    expected = np.array([[440.0, 180.0]])
    assert corrected == pytest.approx(expected, abs=1e-6)


def test_zero_distortion_leaves_pixels_untouched():
    k = np.array([[900.0, 0.0, 960.0], [0.0, 900.0, 540.0], [0.0, 0.0, 1.0]])
    model = CameraModel(k, np.eye(3), np.zeros(3), np.zeros(5), 'plumb_bob')
    pixels = np.array([[10.0, 20.0], [1900.0, 1000.0]])

    assert undistort_pixels(pixels, model) == pytest.approx(pixels)


def test_plumb_bob_coefficients_are_still_applied():
    k = np.array([[300.0, 0.0, 320.0], [0.0, 300.0, 240.0], [0.0, 0.0, 1.0]])
    distortion = np.array([0.1, -0.02, 0.0, 0.0, 0.0])
    model = CameraModel(k, np.eye(3), np.zeros(3), distortion, 'plumb_bob')
    pixels = np.array([[440.0, 180.0]])

    corrected = undistort_pixels(pixels, model)

    assert corrected != pytest.approx(pixels)


def test_relative_motion_is_expressed_in_the_origin_frame():
    from monoscale_odometry.geometry import Pose2, relative_motion

    origin = Pose2(2.0, 1.0, math.pi / 2.0)
    pose = Pose2(2.0, 4.0, math.pi / 2.0 + 0.3)

    motion = relative_motion(origin, pose)

    # Moving along world +y while facing +y is straight ahead.
    assert motion.x == pytest.approx(3.0)
    assert motion.y == pytest.approx(0.0, abs=1e-9)
    assert motion.yaw == pytest.approx(0.3)


def test_relative_motion_round_trips_through_compose():
    from monoscale_odometry.geometry import Pose2, relative_motion

    origin = Pose2(-1.0, 5.0, -0.8)
    pose = Pose2(3.0, 2.0, 1.1)

    restored = origin.compose(relative_motion(origin, pose))

    assert restored.x == pytest.approx(pose.x)
    assert restored.y == pytest.approx(pose.y)
    assert restored.yaw == pytest.approx(pose.yaw)
