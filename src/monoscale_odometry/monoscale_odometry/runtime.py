"""What a camera carries between frames, and what a solve hands back.

Kept out of the node so both front ends -- the tracker's and the image one --
can name the same things without importing each other.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Optional

import numpy as np

from .geometry import CameraModel, PlanarMotion, Pose2


@dataclass
class Frame:
    stamp: float
    stamp_msg: object
    gray: np.ndarray
    sequence: int
    # How far the ground moved on the hop that produced this frame, for frames
    # that arrive already tracked. The image path measures this itself when it
    # carries the features forward; the track path is told, and has to hold the
    # figure until the frame is actually consumed so the solve trigger sees the
    # same signal at the same moment in both.
    disparity: float = 0.0
    # Pixels and identities as of this frame, for frames that arrive already
    # tracked. They belong to the frame and not to the camera: the camera only
    # ever holds the newest set, and online the pair being processed is not
    # always the newest frame.
    pixels: Optional[np.ndarray] = None
    ids: Optional[np.ndarray] = None


@dataclass
class TrackResult:
    previous_pixels: np.ndarray
    current_pixels: np.ndarray
    previous_ground: np.ndarray
    current_ground: np.ndarray
    ground_valid: np.ndarray
    motion_inliers: np.ndarray
    motion: Optional[PlanarMotion]
    track_ids: Optional[np.ndarray] = None
    anchored_from_map: bool = False
    # RMS spread of the inlying votes: how precise this camera's answer was.
    spread: float = 0.0


GROUND = 0.0


OBSTACLE = 1.0


CAMERA_MOUNTS = {
    # Rotation base<-camera as a flat 3x3, then the mount point in base_link.
    # Only a default: a vehicle supplies its own calibration. Any name not
    # listed here starts at identity, which a real mount will overwrite.
    'front': ([0.0, -0.5, 0.8660254, -1.0, 0.0, 0.0, 0.0, -0.8660254, -0.5],
              [3.5, 0.0, 0.89]),
    'rear': ([0.0, 0.5, -0.8660254, 1.0, 0.0, 0.0, 0.0, -0.8660254, -0.5],
             [-0.82, 0.0, 1.26]),
}


@dataclass
class CameraRuntime:
    name: str
    model: CameraModel
    fov_deg: float
    # Whether the mount came off the transform tree; false means the
    # configured one is still in use.
    mounted: bool = False
    calibration_model: Optional[CameraModel] = None
    calibration_width: int = 0
    calibration_height: int = 0
    latest: Optional[Frame] = None
    previous: Optional[Frame] = None
    sequence: int = 0
    keyframe: Optional[Frame] = None
    keyframe_pose: Optional['Pose2'] = None
    # Per feature, the ground projection and camera position from when it was
    # first seen, so each earns its own baseline over as many frames as it
    # survives rather than sharing one keyframe.
    slip_origins: Optional[tuple] = None
    ground_min_distance: float = 0.0
    # Pixels currently being followed and the identity of each, so a feature
    # keeps the same name for as long as it survives.
    track_pixels: Optional[np.ndarray] = None
    track_ids: Optional[np.ndarray] = None
    # Where each surviving feature sat, and which frame that was, the last
    # time a pose was solved. Tracking runs every frame so each hop stays
    # small, but the solve compares against this instead of the frame before
    # it, so it still gets the long baseline it needs.
    range_scale: float = 1.0
    solve_ids: Optional[np.ndarray] = None
    solve_points: Optional[np.ndarray] = None
    solve_frame: Optional[Frame] = None
    # Descriptors for features detected this frame, held until the anchors
    # they belong to exist.
    pending_descriptors: Optional[tuple] = None
    # Pixel travel accumulated over the single-frame hops since the last
    # solve, where the near ground is still in view.
    hop_disparity: float = 0.0
    anchors: Optional['GroundAnchorMap'] = None
    next_track_id: int = 0
