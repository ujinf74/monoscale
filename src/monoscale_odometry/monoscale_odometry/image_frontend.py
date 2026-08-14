"""The front end the estimator uses when it is given images, not tracks.

The deployed path is the C++ tracker: it publishes finished feature tracks and
the estimator only projects and registers them. This is what ran before that
existed, and what still runs when `track_topic_prefix` is empty -- the same
detect-and-follow in Python, an order of magnitude slower, which is why the
processing width could never be raised while it was the only option.

It is kept because it is the only path that needs no second node, which makes
it useful offline and as a reference for what the tracker is supposed to
produce. It is separated so that the deployed path can be read, and ported,
without it.
"""

import math
import os
import time

import cv2
import numpy as np
from sensor_msgs.msg import Image

from .runtime import CameraRuntime, Frame
from .tracks import GroundAnchorMap
from .geometry import intrinsic_from_fov, predict_ground_pixels


def _image_to_gray(msg: Image) -> np.ndarray:
    encoding = msg.encoding.lower()
    if encoding in ('mono8', '8uc1'):
        channels = 1
    elif encoding in ('rgb8', 'bgr8', '8uc3'):
        channels = 3
    elif encoding in ('rgba8', 'bgra8', '8uc4'):
        channels = 4
    else:
        raise ValueError(f'unsupported image encoding: {msg.encoding}')

    row_bytes = msg.width * channels
    raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    image = raw[:, :row_bytes].reshape(msg.height, msg.width, channels) if channels > 1 \
        else raw[:, :msg.width]
    if channels == 1:
        return image.copy()
    if encoding.startswith('rgb') or encoding == '8uc3':
        code = cv2.COLOR_RGB2GRAY if channels == 3 else cv2.COLOR_RGBA2GRAY
    else:
        code = cv2.COLOR_BGR2GRAY if channels == 3 else cv2.COLOR_BGRA2GRAY
    return cv2.cvtColor(image, code)


# Corner detection through OpenCL, off unless CVA_USE_OPENCL is set.
#
# It is a real saving -- at 2560 wide, detection goes from 33.0 ms to 2.7 and
# the whole solve from 27.4 to 13.7 -- but not a free one. With a mask in play
# the OpenCL kernel returns a different set of corners from the CPU one, and
# since goodFeaturesToTrack orders by quality and the caller keeps the first N,
# a different order is a different set of features. Measured on an 8 s drive
# that costs 2D ATE 0.039 -> 0.084 m and drift 0.09 -> 0.22%.
#
# The CPU path already fits 33.3 Hz at 27.4 ms, so accuracy wins here. The
# switch stays for a machine where it does not fit -- an Orin, most likely.
_OPENCL = (os.environ.get('CVA_USE_OPENCL', '') not in ('', '0')
           and cv2.ocl.haveOpenCL() and cv2.ocl.useOpenCL())


def _accelerate(array):
    """Hand an array to OpenCL when asked, otherwise leave it be.

    Only worth it on the large frames: below a few hundred thousand pixels the
    transfer is a bigger share than the work saved.
    """
    if not _OPENCL or array is None or array.size < 300000:
        return array
    return cv2.UMat(array)


class ImageFrontend:
    """Mixed into the node: these methods are its methods, kept apart."""

    def _on_image(self, name: str, msg: Image):
        self.arrivals[name] += 1
        runtime = self.cameras[name]
        ingest_start = time.perf_counter()
        try:
            gray = _image_to_gray(msg)
        except ValueError as error:
            self.get_logger().error(str(error), throttle_duration_sec=5.0)
            return
        calibration = runtime.calibration_model
        if calibration.k[0, 0] <= 0.0:
            if not self.allow_fov_fallback:
                self.get_logger().error(
                    f'Waiting for valid camera calibration: {name} K is zero',
                    throttle_duration_sec=5.0,
                )
                return
            calibration = CameraModel(
                intrinsic_from_fov(msg.width, msg.height, runtime.fov_deg),
                calibration.rotation_base_from_camera,
                calibration.translation_base_from_camera,
                calibration.distortion,
                calibration.distortion_model,
            )
            runtime.calibration_model = calibration
            # The derived K describes this message's resolution, which is what
            # the intrinsics get scaled against from here on.
            runtime.calibration_width = int(msg.width)
            runtime.calibration_height = int(msg.height)
            self.get_logger().warning(
                f'{name} CameraInfo unavailable; derived intrinsics from '
                f'{runtime.fov_deg:.1f} deg horizontal FOV'
            )
        if self.processing_width > 0 and msg.width > self.processing_width:
            scale = self.processing_width / float(msg.width)
            gray = cv2.resize(
                gray,
                (self.processing_width, int(round(msg.height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        runtime.model = self._frame_model(runtime, gray.shape[1], gray.shape[0])
        runtime.sequence += 1
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if stamp <= 0.0:
            stamp = self.get_clock().now().nanoseconds * 1e-9
        self._queue_frame(
            name, Frame(stamp, msg.header.stamp, gray, runtime.sequence))
        self.ingest_seconds += time.perf_counter() - ingest_start
        self.ingest_count += 1
        self._try_process_pair()

    def _spread_over_grid(self, found: np.ndarray, shape, wanted: int) -> np.ndarray:
        """Keep a quota per image cell instead of the globally strongest.

        goodFeaturesToTrack returns corners in quality order, so a patch of
        rich texture takes the whole budget and the rest of the ground goes
        unwatched. What the registration needs is the opposite: points spread
        far apart, because the geometry is conditioned on their spread and not
        on their count. OpenVINS extracts per grid cell with a fixed quota for
        this reason (ov_core Grider_GRID); doing the same by bucketing one
        detector call costs a sort rather than a detection per cell.
        """
        if not self.detection_grid or len(found) <= wanted:
            return found[:wanted]
        columns, rows = self.detection_grid
        height, width = shape
        cell = (
            np.minimum((found[:, 1] * rows / height).astype(np.int64), rows - 1) * columns
            + np.minimum((found[:, 0] * columns / width).astype(np.int64), columns - 1)
        )
        quota = max(int(math.ceil(wanted / float(columns * rows))), 1)
        # Stable sort keeps quality order inside each cell, so the rank below
        # is "how many better corners this cell already has".
        order = np.argsort(cell, kind='stable')
        boundaries = np.flatnonzero(np.diff(cell[order])) + 1
        starts = np.concatenate(([0], boundaries))
        lengths = np.diff(np.concatenate((starts, [len(order)])))
        rank = np.arange(len(order)) - np.repeat(starts, lengths)
        keep = np.sort(order[rank < quota])
        return found[keep][:wanted]

    def _detect_new_tracks(self, runtime: CameraRuntime, gray: np.ndarray):
        """Top up the tracked set, keeping clear of features already followed."""
        held = 0 if runtime.track_pixels is None else len(runtime.track_pixels)
        wanted = self.max_features - held
        if wanted <= 0 or held > self.feature_refill_ratio * self.max_features:
            return
        mask = np.full(gray.shape, 255, dtype=np.uint8)
        if runtime.track_pixels is not None and len(runtime.track_pixels):
            # Six hundred cv2.circle calls per frame, in Python, holding the
            # interpreter lock while the other camera waits. Blanking a square
            # per feature with fancy indexing does the same job in one call;
            # The stamp stays a disc: a square blanks a third more area and
            # the features that survive spread differently enough to show up in
            # the result.
            radius = int(max(self.min_feature_distance, 1.0))
            if self._stamp_offsets is None or self._stamp_radius != radius:
                span = np.arange(-radius, radius + 1)
                dy, dx = np.meshgrid(span, span, indexing='ij')
                inside = dy * dy + dx * dx <= radius * radius
                self._stamp_offsets = (dy[inside], dx[inside])
                self._stamp_radius = radius
            dy, dx = self._stamp_offsets
            height, width = gray.shape
            centres = runtime.track_pixels.astype(np.int32)
            rows = np.clip(centres[:, 1, None] + dy[None, :], 0, height - 1)
            columns = np.clip(centres[:, 0, None] + dx[None, :], 0, width - 1)
            mask[rows.ravel(), columns.ravel()] = 0
        # Corner detection through OpenCL. It is the single most expensive
        # stage at the fisheye's working size -- 55 ms of a 2560 wide frame
        # against the 8.6 the optical flow costs -- and the transfer is 0.45 ms
        # of that, so the upload pays for itself many times over. Measured
        # identical to the CPU path: 3000 corners, maximum coordinate
        # difference 0.0000 px.
        found = cv2.goodFeaturesToTrack(
            _accelerate(gray),
            maxCorners=wanted * self.detection_oversample if self.detection_grid else wanted,
            qualityLevel=self.quality_level,
            minDistance=self.min_feature_distance,
            blockSize=7,
            mask=None if mask is None else _accelerate(mask),
        )
        if isinstance(found, cv2.UMat):
            found = found.get()
        if found is None or len(found) == 0:
            return
        found = found.reshape(-1, 2).astype(np.float64)
        found = self._spread_over_grid(found, gray.shape, wanted)
        identifiers = np.arange(
            runtime.next_track_id, runtime.next_track_id + len(found), dtype=np.int64
        )
        runtime.next_track_id += len(found)
        descriptors, described = self._describe(gray, found)
        if descriptors is not None and runtime.anchors is not None:
            ground, valid = pixels_to_ground(
                found, runtime.model, self.ground_max_distance,
                runtime.ground_min_distance,
            )
            usable = np.flatnonzero(valid & described)
            if len(usable):
                world = transform_points_to_world(
                    np.column_stack((ground[usable], np.zeros(len(usable)))), self.pose
                )
                active = set(
                    int(i) for i in (runtime.track_ids
                                     if runtime.track_ids is not None else [])
                )
                matched = runtime.anchors.reassociate(
                    world[:, :2], descriptors[usable], active,
                    self.reassociation_radius, self.reassociation_hamming,
                    self.reassociation_min_obs,
                )
                for index, identifier in matched.items():
                    identifiers[usable[index]] = identifier

        if runtime.track_pixels is None or len(runtime.track_pixels) == 0:
            runtime.track_pixels = found
            runtime.track_ids = identifiers
        else:
            runtime.track_pixels = np.vstack((runtime.track_pixels, found))
            runtime.track_ids = np.concatenate((runtime.track_ids, identifiers))

    def _advance_tracks(self, runtime: CameraRuntime) -> bool:
        """Carry the tracked set from the previous frame into the latest one."""
        previous, current = runtime.previous, runtime.latest
        if previous is None or current is None:
            return False
        if previous.gray.shape != current.gray.shape:
            runtime.previous = current
            return False
        if runtime.anchors is None:
            runtime.anchors = GroundAnchorMap(
                max_anchors=self.max_anchors,
                max_age_frames=self.anchor_max_age,
                update_gain=self.anchor_update_gain,
                max_observations=self.anchor_max_observations,
                select_by_consistency=self.anchor_selection,
            )
        if runtime.track_pixels is None or len(runtime.track_pixels) == 0:
            self._detect_new_tracks(runtime, previous.gray)
        followed = None
        if runtime.track_pixels is not None and len(runtime.track_pixels):
            _t = time.perf_counter()
            followed = self._follow_tracks(runtime, previous, current)
            self.stage['lk'] += time.perf_counter() - _t
        if followed is not None:
            before, runtime.track_pixels, runtime.track_ids = followed
            if len(before):
                shift = np.linalg.norm(runtime.track_pixels - before, axis=1)
                runtime.hop_disparity += float(np.percentile(shift, 90))
        else:
            runtime.track_pixels = None
            runtime.track_ids = None
        _t = time.perf_counter()
        self._detect_new_tracks(runtime, current.gray)
        self.stage['detect'] += time.perf_counter() - _t
        runtime.previous = current
        return followed is not None

    def _track_between(self, source: np.ndarray, target: np.ndarray):
        """Forward-backward checked LK between two arbitrary frames."""
        source_pixels = cv2.goodFeaturesToTrack(
            source,
            maxCorners=self.max_features,
            qualityLevel=self.quality_level,
            minDistance=self.min_feature_distance,
            blockSize=7,
        )
        if source_pixels is None or len(source_pixels) == 0:
            return None
        window = dict(
            winSize=(self.lk_window, self.lk_window),
            maxLevel=self.lk_levels,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        target_pixels, status, error = cv2.calcOpticalFlowPyrLK(
            source, target, source_pixels, None, **window
        )
        if target_pixels is None:
            return None
        back_pixels, back_status, _ = cv2.calcOpticalFlowPyrLK(
            target, source, target_pixels, None, **window
        )
        if back_pixels is None:
            return None
        source_pixels = source_pixels.reshape(-1, 2)
        target_pixels = target_pixels.reshape(-1, 2)
        back_pixels = back_pixels.reshape(-1, 2)
        height, width = target.shape
        valid = (
            status.reshape(-1).astype(bool)
            & back_status.reshape(-1).astype(bool)
            & np.isfinite(target_pixels).all(axis=1)
            & (error.reshape(-1) <= self.lk_error_threshold)
            & (
                np.linalg.norm(back_pixels - source_pixels, axis=1)
                <= self.lk_forward_backward_threshold
            )
            & (target_pixels[:, 0] >= 0)
            & (target_pixels[:, 0] < width)
            & (target_pixels[:, 1] >= 0)
            & (target_pixels[:, 1] < height)
        )
        if not np.any(valid):
            return None
        return source_pixels[valid], target_pixels[valid]
