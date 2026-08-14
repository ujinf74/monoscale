from dataclasses import dataclass
import math
from typing import Iterable, Optional, Tuple

import cv2
import numpy as np

try:
    import monoscale_fast as _fast
except ImportError:  # pragma: no cover - the pure Python path is the fallback
    # Without the extension everything still runs, just slower. The two are
    # held to agreeing exactly by test_fast_equivalence, which is what makes
    # a score taken with one comparable to a score taken with the other.
    _fast = None


@dataclass(frozen=True)
class CameraModel:
    k: np.ndarray
    rotation_base_from_camera: np.ndarray
    translation_base_from_camera: np.ndarray
    distortion: Optional[np.ndarray] = None
    distortion_model: str = ''


@dataclass(frozen=True)
class PlanarMotion:
    x: float
    y: float
    yaw: float
    inliers: int
    scale: float


@dataclass(frozen=True)
class Pose2:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0

    def compose(self, motion: PlanarMotion) -> 'Pose2':
        c = math.cos(self.yaw)
        s = math.sin(self.yaw)
        return Pose2(
            self.x + c * motion.x - s * motion.y,
            self.y + s * motion.x + c * motion.y,
            wrap_pi(self.yaw + motion.yaw),
        )


def wrap_pi(angle: float) -> float:
    return math.remainder(angle, 2.0 * math.pi)


def relative_motion(origin: Pose2, pose: Pose2) -> PlanarMotion:
    """Motion from `origin` to `pose`, expressed in the frame of `origin`.

    Same convention as the frame to frame estimate, so it can be handed to
    triangulation as the transform between two views.
    """
    c = math.cos(origin.yaw)
    s = math.sin(origin.yaw)
    dx = pose.x - origin.x
    dy = pose.y - origin.y
    return PlanarMotion(
        c * dx + s * dy,
        -s * dx + c * dy,
        wrap_pi(pose.yaw - origin.yaw),
        0,
        1.0,
    )


def intrinsic_from_fov(width: int, height: int, horizontal_fov_deg: float) -> np.ndarray:
    if width <= 0 or height <= 0:
        raise ValueError('image dimensions must be positive')
    fov = math.radians(horizontal_fov_deg)
    if not 0.0 < fov < math.pi:
        raise ValueError('horizontal FOV must be between 0 and 180 degrees')
    focal = width / (2.0 * math.tan(0.5 * fov))
    return np.array(
        [[focal, 0.0, 0.5 * (width - 1)],
         [0.0, focal, 0.5 * (height - 1)],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


_FISHEYE_MODELS = ('equidistant', 'fisheye')


def undistort_pixels(pixels: np.ndarray, model: CameraModel) -> np.ndarray:
    """Pixels as this camera sees them, rewritten as pixels a pinhole would see.

    Everything downstream turns a pixel into a ray with inv(K), which is the
    pinhole inverse. A fisheye pixel put through it is simply read wrong, so it
    has to be carried across first.

    The zero-coefficient shortcut below is only valid for plumb_bob. An
    equidistant fisheye is exactly d = [0,0,0,0] -- theta_d = theta is the
    definition of the projection, not the absence of one -- and taking that for
    "no distortion" sent 139.5 degrees of fisheye through the pinhole inverse
    untouched. Odometry came out at 37.7% drift with the scale at 0.64.
    """
    pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    if pixels.size == 0:
        return pixels
    fisheye = model.distortion_model in _FISHEYE_MODELS
    distortion = (np.zeros(4) if model.distortion is None
                  else np.asarray(model.distortion, dtype=np.float64))
    # CARLA renders an ideal pinhole camera and reports plumb_bob with zero
    # coefficients, so undistortion is an identity map on every pixel. That
    # reasoning does not carry to a fisheye, which is why the test is on the
    # model rather than on the coefficients alone.
    if not fisheye and not np.any(distortion):
        return pixels
    points = pixels.reshape(-1, 1, 2)
    if fisheye:
        if len(distortion) < 4:
            distortion = np.zeros(4)
        # Normalised rays, then re-projected through K so the caller's inv(K)
        # recovers them. Straight to P=K would fold the fisheye back into a
        # pinhole picture and lose nothing, but only because it is the same
        # step written twice; doing it here keeps one place that knows the
        # lens.
        corrected = cv2.fisheye.undistortPoints(
            points, model.k, distortion[:4].reshape(-1, 1), P=model.k
        )
    else:
        corrected = cv2.undistortPoints(points, model.k, distortion, P=model.k)
    return corrected.reshape(-1, 2)


def pixels_to_ground(
    pixels: np.ndarray,
    model: CameraModel,
    max_distance: float,
    min_distance: float = 0.0,
    tilt: Optional[np.ndarray] = None,
    range_scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Ground intersections of the rays through `pixels`, and which are usable.

    `min_distance` exists because pixel motion goes as the inverse of range:
    the closest ground sweeps the image fastest and is the first thing optical
    flow loses. The front camera sits lower, so its nearest ground is nearer
    and it loses that band at a lower speed than the rear does.

    `tilt` is the vehicle's roll and pitch as a rotation from base into a
    level frame, and it is not a refinement. A camera pitched 30 degrees down
    sees the ground at a range that goes as 1/tan(30 deg + d), so one degree
    of body tilt moves every range in the frame by two and a half to five per
    cent. On a straight drive the body holds within 0.02 deg of level and
    assuming level costs nothing, which is why this went unnoticed; through a
    turn the same vehicle rolls 1.6 deg on average and 3.3 at the peak.
    Passing None keeps the level assumption.

    `range_scale` divides every range out by a factor measured for this
    camera. It is not a claim about where the camera is -- the extrinsics have
    to keep matching the sensor kit -- but about how far the ground is
    measured to have moved once the flow and this projection are both done
    with, which on both cameras is further than the vehicle actually went.
    """
    if pixels.size == 0:
        return np.empty((0, 2), np.float64), np.empty(0, dtype=bool)
    pixels = undistort_pixels(pixels, model)
    origin = model.translation_base_from_camera
    if range_scale != 1.0:
        # Every range out of here is proportional to the camera's height over
        # the plane, so the whole measurement scales with this one number.
        origin = np.array([origin[0], origin[1], origin[2] / range_scale])
    if _fast is not None:
        return _fast.pixels_to_ground(
            np.ascontiguousarray(pixels, dtype=np.float64),
            np.linalg.inv(model.k),
            np.ascontiguousarray(model.rotation_base_from_camera, dtype=np.float64),
            np.ascontiguousarray(origin, dtype=np.float64),
            float(max_distance),
            float(min_distance),
            None if tilt is None else np.ascontiguousarray(tilt, dtype=np.float64),
        )
    homogeneous = np.column_stack((pixels, np.ones(len(pixels))))
    rays_camera = (np.linalg.inv(model.k) @ homogeneous.T).T
    rays_base = (model.rotation_base_from_camera @ rays_camera.T).T
    if tilt is None:
        normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        # The ground stays level in the world while the body tilts, so in body
        # coordinates the plane's normal is the world vertical seen from here.
        normal = tilt.T @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    denominator = rays_base @ normal
    lam = np.full(len(pixels), np.nan, dtype=np.float64)
    nonparallel = np.abs(denominator) > 1e-8
    lam[nonparallel] = -(origin @ normal) / denominator[nonparallel]
    points = origin[None, :] + lam[:, None] * rays_base
    if tilt is not None:
        # Into the level frame, so the two coordinates that come out are the
        # ones the planar solve is entitled to treat as a plane.
        points = (tilt @ points.T).T
    distance = np.linalg.norm(points[:, :2] - origin[None, :2], axis=1)
    valid = (
        nonparallel
        & np.isfinite(points).all(axis=1)
        & (lam > 0.0)
        & (distance <= max_distance)
        & (distance >= min_distance)
    )
    return points[:, :2], valid


def estimate_planar_motion(
    previous_points: np.ndarray,
    current_points: np.ndarray,
    ransac_threshold: float,
    min_inliers: int,
    max_scale_error: float,
) -> Optional[Tuple[PlanarMotion, np.ndarray]]:
    previous_points = np.asarray(previous_points, dtype=np.float64).reshape(-1, 2)
    current_points = np.asarray(current_points, dtype=np.float64).reshape(-1, 2)
    if len(previous_points) < max(3, min_inliers):
        return None
    affine, mask = cv2.estimateAffinePartial2D(
        current_points,
        previous_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_threshold,
        maxIters=2000,
        confidence=0.995,
        refineIters=10,
    )
    if affine is None or mask is None:
        return None
    inlier_mask = mask.reshape(-1).astype(bool)
    inliers = int(np.count_nonzero(inlier_mask))
    if inliers < min_inliers:
        return None
    a = affine[:, :2]
    scale = math.sqrt(max(np.linalg.det(a), 0.0))
    if not math.isfinite(scale) or abs(scale - 1.0) > max_scale_error:
        return None
    rotation = a / max(scale, 1e-12)
    yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    translation = affine[:, 2]
    return (
        PlanarMotion(float(translation[0]), float(translation[1]), yaw, inliers, scale),
        inlier_mask,
    )


def estimate_planar_motion_with_yaw(
    previous_points: np.ndarray,
    current_points: np.ndarray,
    yaw: float,
    ransac_threshold: float,
    min_inliers: int,
) -> Optional[Tuple[PlanarMotion, np.ndarray]]:
    previous_points = np.asarray(previous_points, dtype=np.float64).reshape(-1, 2)
    current_points = np.asarray(current_points, dtype=np.float64).reshape(-1, 2)
    if len(previous_points) < max(2, min_inliers):
        return None
    c = math.cos(yaw)
    s = math.sin(yaw)
    rotation = np.array([[c, -s], [s, c]], dtype=np.float64)
    offsets = previous_points - (rotation @ current_points.T).T

    candidate_indices = np.linspace(
        0, len(offsets) - 1, min(len(offsets), 100), dtype=int
    )
    best_mask = None
    best_count = 0
    for index in candidate_indices:
        residual = np.linalg.norm(offsets - offsets[index], axis=1)
        mask = residual <= ransac_threshold
        count = int(np.count_nonzero(mask))
        if count > best_count:
            best_count = count
            best_mask = mask
    if best_mask is None or best_count < min_inliers:
        return None

    translation = np.median(offsets[best_mask], axis=0)
    residual = np.linalg.norm(offsets - translation, axis=1)
    inlier_mask = residual <= ransac_threshold
    inliers = int(np.count_nonzero(inlier_mask))
    if inliers < min_inliers:
        return None
    translation = np.mean(offsets[inlier_mask], axis=0)
    return (
        PlanarMotion(
            float(translation[0]), float(translation[1]), float(yaw), inliers, 1.0
        ),
        inlier_mask,
    )


def fuse_planar_motions(motions: Iterable[PlanarMotion]) -> Optional[PlanarMotion]:
    motions = list(motions)
    if not motions:
        return None
    weights = np.asarray([max(m.inliers, 1) for m in motions], dtype=np.float64)
    weights /= weights.sum()
    x = float(sum(w * m.x for w, m in zip(weights, motions)))
    y = float(sum(w * m.y for w, m in zip(weights, motions)))
    sine = sum(w * math.sin(m.yaw) for w, m in zip(weights, motions))
    cosine = sum(w * math.cos(m.yaw) for w, m in zip(weights, motions))
    yaw = math.atan2(sine, cosine)
    scale = float(sum(w * m.scale for w, m in zip(weights, motions)))
    return PlanarMotion(x, y, yaw, sum(m.inliers for m in motions), scale)


def triangulate_temporal_points(
    previous_pixels: np.ndarray,
    current_pixels: np.ndarray,
    model: CameraModel,
    motion_current_to_previous: PlanarMotion,
    min_parallax_deg: float,
    max_reprojection_error: float,
    min_distance: float,
    max_distance: float,
) -> Tuple[np.ndarray, np.ndarray]:
    previous_pixels = np.asarray(previous_pixels, dtype=np.float64).reshape(-1, 2)
    current_pixels = np.asarray(current_pixels, dtype=np.float64).reshape(-1, 2)
    if len(previous_pixels) == 0:
        return np.empty((0, 3), np.float64), np.empty(0, dtype=bool)
    previous_pixels = undistort_pixels(previous_pixels, model)
    current_pixels = undistort_pixels(current_pixels, model)

    c = math.cos(motion_current_to_previous.yaw)
    s = math.sin(motion_current_to_previous.yaw)
    r_prev_from_current_base = np.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    t_prev_from_current_base = np.array(
        [motion_current_to_previous.x, motion_current_to_previous.y, 0.0], dtype=np.float64
    )
    r_bc = model.rotation_base_from_camera
    t_bc = model.translation_base_from_camera
    r_c1_from_c2 = r_bc.T @ r_prev_from_current_base @ r_bc
    t_c1_from_c2 = r_bc.T @ (
        r_prev_from_current_base @ t_bc + t_prev_from_current_base - t_bc
    )
    r_c2_from_c1 = r_c1_from_c2.T
    t_c2_from_c1 = -r_c2_from_c1 @ t_c1_from_c2

    p1 = model.k @ np.column_stack((np.eye(3), np.zeros(3)))
    p2 = model.k @ np.column_stack((r_c2_from_c1, t_c2_from_c1))
    homogeneous = cv2.triangulatePoints(p1, p2, previous_pixels.T, current_pixels.T)
    points_c1 = (homogeneous[:3] / homogeneous[3]).T
    points_c2 = (r_c2_from_c1 @ points_c1.T).T + t_c2_from_c1

    projected1 = (model.k @ points_c1.T).T
    projected2 = (model.k @ points_c2.T).T
    projected1 = projected1[:, :2] / projected1[:, 2:3]
    projected2 = projected2[:, :2] / projected2[:, 2:3]
    error = np.maximum(
        np.linalg.norm(projected1 - previous_pixels, axis=1),
        np.linalg.norm(projected2 - current_pixels, axis=1),
    )

    ray1 = points_c1 / np.linalg.norm(points_c1, axis=1, keepdims=True)
    camera2_in_c1 = t_c1_from_c2
    ray2 = points_c1 - camera2_in_c1[None, :]
    ray2 /= np.linalg.norm(ray2, axis=1, keepdims=True)
    cosine = np.clip(np.sum(ray1 * ray2, axis=1), -1.0, 1.0)
    parallax = np.degrees(np.arccos(cosine))
    distance = np.linalg.norm(points_c1, axis=1)
    valid = (
        np.isfinite(points_c1).all(axis=1)
        & (points_c1[:, 2] > 0.0)
        & (points_c2[:, 2] > 0.0)
        & (error <= max_reprojection_error)
        & (parallax >= min_parallax_deg)
        & (distance >= min_distance)
        & (distance <= max_distance)
    )
    points_base_previous = (r_bc @ points_c1.T).T + t_bc
    import os as _os
    if _os.environ.get('TRI_DEBUG'):
        import sys as _sy
        n = len(points_c1)
        z1 = points_c1[:, 2] > 0.0
        z2 = points_c2[:, 2] > 0.0
        gates = (('총', np.ones(n, bool)),
                 ('유한', np.isfinite(points_c1).all(axis=1)),
                 ('z>0 양쪽', z1 & z2),
                 (f'재투영<={max_reprojection_error}', error <= max_reprojection_error),
                 (f'시차>={min_parallax_deg}', parallax >= min_parallax_deg),
                 (f'거리 {min_distance}~{max_distance}',
                  (distance >= min_distance) & (distance <= max_distance)),
                 ('전부', valid))
        z = points_base_previous[valid, 2] if valid.any() else np.zeros(0)
        band = lambda lo, hi: int(((z >= lo) & (z < hi)).sum())
        print(f'[tri] 통과={int(valid.sum())}/{n}  '
              f'시차탈락={n - int((parallax >= min_parallax_deg).sum())}  '
              f'높이: <0.2={band(-99, 0.2)} 0.2-0.6={band(0.2, 0.6)} '
              f'차량0.6-1.8={band(0.6, 1.8)} 1.8-2.5={band(1.8, 2.5)} '
              f'>2.5={band(2.5, 999)}',
              file=_sy.stderr, flush=True)
    return points_base_previous, valid


def project_ground_to_pixels(points_xy: np.ndarray, model: CameraModel) -> np.ndarray:
    """Project ground points, given in base_link, back into the image."""
    points_xy = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    points = np.column_stack((points_xy, np.zeros(len(points_xy))))
    camera = (
        model.rotation_base_from_camera.T
        @ (points - model.translation_base_from_camera).T
    ).T
    behind = camera[:, 2] <= 1e-6
    camera[behind, 2] = 1e-6
    projected = (model.k @ camera.T).T
    pixels = projected[:, :2] / projected[:, 2:3]
    pixels[behind] = np.nan
    return pixels


def predict_ground_pixels(
    pixels: np.ndarray, model: CameraModel, motion: PlanarMotion
) -> Optional[np.ndarray]:
    """Where ground features should land after the vehicle moves by `motion`.

    Optical flow searches around wherever the feature was last seen. Without a
    prediction the search settles short of the true displacement, and the
    consensus forms around features that appear barely to have moved. Measured
    on a straight 46 m run at 2 m/s, dropping this took drift from 3.3 % to
    21.7 %.
    """
    ground, valid = pixels_to_ground(pixels, model, math.inf)
    if not np.any(valid):
        return None
    c = math.cos(motion.yaw)
    s = math.sin(motion.yaw)
    rotation = np.array([[c, s], [-s, c]], dtype=np.float64)
    moved = (rotation @ (ground - np.array([motion.x, motion.y])).T).T
    predicted = project_ground_to_pixels(moved, model)
    predicted[~valid] = pixels[~valid]
    return np.where(np.isfinite(predicted), predicted, pixels)


def transform_points_to_world(points: np.ndarray, pose: Pose2) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    c = math.cos(pose.yaw)
    s = math.sin(pose.yaw)
    out = points.copy()
    out[:, 0] = pose.x + c * points[:, 0] - s * points[:, 1]
    out[:, 1] = pose.y + s * points[:, 0] + c * points[:, 1]
    return out

def twist_from_motion(motion: PlanarMotion) -> Tuple[float, float, float]:
    """The constant body twist that would produce this motion in unit time.

    A vehicle turning at a steady rate travels an arc, not a chord, and its
    heading turns while it does. Treating a measured hop as a straight line
    and stretching it is right only while the wheel is centred: over a 20 ms
    mismatch at 16 deg/s the direction of travel is out by a quarter of a
    degree, and over the tens of milliseconds a rejected pair has to be
    carried across it is out by more.

    This is the SE(2) logarithm. Straight motion falls out of it unchanged --
    the rotation terms go to their limits as the turn rate goes to zero --
    so there is no separate case to get wrong.
    """
    omega = motion.yaw
    if abs(omega) < 1e-9:
        return motion.x, motion.y, omega
    a = math.sin(omega) / omega
    b = (1.0 - math.cos(omega)) / omega
    denominator = a * a + b * b
    return (
        (a * motion.x + b * motion.y) / denominator,
        (-b * motion.x + a * motion.y) / denominator,
        omega,
    )


def motion_from_twist(
    vx: float, vy: float, omega: float, duration: float = 1.0
) -> PlanarMotion:
    """Where a constant body twist carries the vehicle over `duration`."""
    vx, vy, omega = vx * duration, vy * duration, omega * duration
    if abs(omega) < 1e-9:
        return PlanarMotion(vx, vy, omega, 0, 1.0)
    a = math.sin(omega) / omega
    b = (1.0 - math.cos(omega)) / omega
    return PlanarMotion(a * vx - b * vy, b * vx + a * vy, omega, 0, 1.0)


def rescale_motion(motion: PlanarMotion, ratio: float) -> PlanarMotion:
    """The same twist, held for `ratio` times as long."""
    if ratio == 1.0:
        return motion
    vx, vy, omega = twist_from_motion(motion)
    carried = motion_from_twist(vx, vy, omega, ratio)
    return PlanarMotion(
        carried.x, carried.y, carried.yaw, motion.inliers, motion.scale)
