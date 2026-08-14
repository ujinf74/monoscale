from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
import array
import math
import os
import time
from typing import Deque, Dict, List, Optional

from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry
import cv2
import numpy as np
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy)
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Header
from tf2_ros import Buffer, TransformBroadcaster, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

from .geometry import CameraModel
from .geometry import estimate_planar_motion
from .geometry import estimate_planar_motion_with_yaw
from .geometry import fuse_planar_motions
from .geometry import intrinsic_from_fov
from .geometry import motion_from_twist
from .geometry import pixels_to_ground
from .geometry import predict_ground_pixels
from .geometry import relative_motion
from .geometry import rescale_motion
from .geometry import PlanarMotion
from .geometry import Pose2
from .geometry import transform_points_to_world
from .geometry import triangulate_temporal_points
from .geometry import wrap_pi
from .fusion import PlanarVelocityFilter
from .runtime import (CAMERA_MOUNTS, CameraRuntime, Frame, GROUND,
                      OBSTACLE, TrackResult)
from .attitude import AttitudeFilter, HeadingBiasFilter
from .image_frontend import ImageFrontend
from .fusion import PlanarDisplacementFilter, world_velocity_from_motion
from .inertial import PlanarInertialPropagator
from .rigid import quaternion_matrix
from .tracks import align_to_anchors
from .tracks import fuse_by_precision
from .tracks import GroundAnchorMap








# What a published point is claimed to be. The solve already knows: a point
# that survived the ground registration is road, one whose projection slid
# against the travel is standing on it.





def _quaternion_from_yaw(yaw):
    return 0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw)




class VisualOdometryNode(ImageFrontend, Node):
    def __init__(self, **kwargs):
        super().__init__('monoscale_odometry', **kwargs)
        declare = self.declare_parameter


        declare('imu_topic', '/sensing/imu/imu_data')
        declare('use_imu_yaw', True)
        # The CARLA bridge publishes the IMU at 100 Hz against simulation time.
        declare('imu_max_age_sec', 0.02)
        # Largest gap between IMU samples still worth interpolating across.
        declare('imu_max_gap_sec', 0.12)
        # CARLA reports a valid pinhole CameraInfo derived from the blueprint FOV.
        declare('use_camera_info', True)
        declare('odometry_topic', '/localization/kinematic_state')
        declare('pose_topic', '/current_pose')
        # What the solve saw on the road, labelled and already in the map
        # frame, for whoever wants to build a map out of it. Each point
        # carries the camera position it was seen from, so the consumer can
        # carve free space along the ray without knowing where the cameras
        # are mounted or which pose the hop was solved against.
        declare('ground_points_topic', '/perception/ground_points')
        declare('map_frame', 'map')
        declare('base_frame', 'base_link')
        declare('sync_tolerance_sec', 0.02)
        # How many frames may wait for a partner before the oldest is thrown
        # away. It used to be a bare deque(maxlen=12), which discards in
        # silence: replaying one recording ten times produced between 644 and
        # 806 pairs out of 810 frames, and nothing in any log said so. The
        # transport was not losing them -- a reliable subscriber receives all
        # 810 and the tracker publishes all 810 -- they were evicted here
        # whenever a solve ran long enough for the queue to back up. Online a
        # shallow queue is right, because a stale frame is worse than no
        # frame; offline it makes every measurement irreproducible. So the
        # depth is settable and every eviction is counted and reported.
        declare('pair_queue_depth', 12)
        declare('allow_fov_fallback', False)
        declare('publish_tf', True)
        # Which cameras this node fuses. One is enough -- the filter then
        # carries `single_camera_variance` where two would have carried their
        # disagreement -- and there is no upper bound: the solve already fuses
        # a list, and the frames are aligned by their stamps rather than by
        # being a pair. The two names below are the defaults because they are
        # what the vehicle carries.
        # Where a camera sits on the vehicle. The transform tree is the one
        # place that already has to be right for anything else to work, so it
        # is asked first and the parameters below are only the answer for a
        # rig that publishes no tree at all. The frame to look up comes from
        # the camera's own CameraInfo, which is what names it.
        declare('extrinsics_from_tf', True)
        declare('tf_lookup_timeout_sec', 5.0)
        declare('camera_names', ['front', 'rear'])
        # Read straight from the node: `get` is not bound until the
        # declarations are done, and these decide what gets declared.
        self.camera_names = [
            str(name)
            for name in self.get_parameter('camera_names').value
        ] or ['front']
        for name in self.camera_names:
            rotation, translation = CAMERA_MOUNTS.get(
                name, ([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                       [0.0, 0.0, 1.0]))
            declare(f'{name}.horizontal_fov_deg', 90.0)
            declare(
                f'{name}.k',
                [900.0, 0.0, 960.0, 0.0, 900.0, 540.0, 0.0, 0.0, 1.0],
            )
            declare(f'{name}.distortion_model', 'plumb_bob')
            declare(f'{name}.d', [0.0] * 5)
            declare(f'{name}.ground_min_distance_m', 0.0)
            declare(f'{name}.calibration_width', 1920)
            declare(f'{name}.calibration_height', 1080)
            declare(f'{name}.rotation_base_from_camera', rotation)
            declare(f'{name}.translation_base_from_camera', translation)
            declare(f'{name}_image_topic',
                    f'/sensing/camera/{name}/image_raw')
            declare(f'{name}_camera_info_topic', '/camera_info')
            declare(f'ground_range_scale_{name}', 1.0)
        declare('processing_width', 640)
        declare('max_features_per_camera', 600)
        declare('quality_level', 0.01)
        declare('min_feature_distance_px', 8.0)
        declare('lk_error_threshold', 30.0)
        # Each pyramid level doubles the pixel displacement LK can follow. At
        # speed the near ground sweeps across the image faster than a shallow
        # pyramid can track, and the survivors are the slow far-field points,
        # which reads as the vehicle having moved less than it did.
        declare('lk_window_px', 21)
        declare('lk_pyramid_levels', 3)
        declare('lk_forward_backward_threshold_px', 1.0)
        declare('ground_max_distance_m', 25.0)
        # A ground point one hop away has moved fx*h*(v*dt)/d^2 pixels. Below
        # about a pixel the tracker cannot separate that from its own noise and
        # reports short, which biases the whole solve short -- so the far edge
        # of the band belongs at the distance where that figure is still worth
        # reading, and that distance goes with sqrt(speed). Holding it fixed
        # only works at the speed it was set for. 0 keeps the fixed limit.
        declare('min_ground_flow_px', 0.0)
        declare('ground_ransac_threshold_m', 0.12)
        declare('ground_min_inliers', 24)
        declare('max_scale_error', 0.08)
        declare('max_translation_per_frame_m', 1.0)
        declare('max_yaw_per_frame_rad', 0.35)
        # How many frames of arrears a single update is allowed to close. It
        # has to be finite: without a bound, a solve that has genuinely gone
        # wrong is accepted simply because enough frames were rejected first.
        declare('max_arrears_frames', 8.0)
        # Roll and pitch from the IMU, off until it earns its place. See
        # AttitudeFilter for what the two numbers cost when set wrongly.
        declare('attitude_from_imu', True)
        declare('attitude_tau_sec', 60.0)
        declare('attitude_gravity_tolerance', 0.3)
        # What the heading source is expected to do wrong, as the part's own
        # numbers rather than as tuning: how large its bias could be at
        # turn-on, how fast that bias wanders, and how noisy the rate is
        # underneath. Setting the first to 0 disables the whole thing and
        # believes the reported heading outright, which is what every
        # measurement before this assumed.
        #
        # That assumption is stronger than it looks. The heading is read from
        # the orientation field, not integrated from the gyro, so what the
        # estimator has relied on is an absolute heading that never drifts --
        # which this simulator provides exactly and a MEMS part on a vehicle
        # does not. Rerecording the same drive with a 0.3 deg/s bias, its
        # random walk and its noise folded into that field takes position
        # RMSE from 0.038 m to 0.169 and drift from 0.24% to 1.17%.
        declare('gyro_bias_sigma_rad_s', 0.0)
        declare('camera_workers', 2)
        # Hz at which queued pairs are solved, decoupled from arrival. 0 keeps
        # the old behaviour of solving inside the receive callback.
        declare('solve_timer_hz', 200.0)
        # 'velocity' takes the vision translation divided by the interval as a
        # velocity measurement; 'displacement' fuses what the two instruments
        # actually measure -- acceleration and metres -- and carries an
        # accelerometer bias state. See PlanarDisplacementFilter.
        declare('fusion_model', 'displacement')
        declare('filter_vision_noise_m', 0.005)
        declare('filter_bias_walk', 0.01)
        # How still 'still' is, in m/s. The zero-velocity update is only as
        # good as the detector behind it.
        declare('zupt_velocity_sigma', 0.01)
        # How far before its own stamp an accelerometer sample describes.
        # CARLA does not measure specific force; it fits a quadratic to the
        # sensor's last three positions and reports the second derivative
        # (InertialMeasurementUnit.cpp, ComputeAccelerometer), which is valid
        # at the middle sample -- one sensor period before the stamp it is
        # published with. Regressed against this recording's truth the
        # agreement rises from 0.883 to 0.988 when the samples are moved back
        # by exactly that one period. It buys no accuracy on its own (the
        # acceleration only shapes the prior within a hop) but the stamps are
        # then what they claim to be. A real instrument does measure specific
        # force, so this stays zero unless the recording is CARLA's.
        declare('imu_acceleration_offset_sec', 0.0)
        # Where to write per-update filter diagnostics; empty writes none.
        declare('filter_debug_csv', '')
        # Threads for the ROS executor. Every subscription gets its own
        # mutually exclusive callback group regardless, so with more than one
        # thread the front camera, the rear camera, the IMU and the solve
        # timer are served independently instead of taking turns in one
        # queue. 1 keeps the historical single-threaded behaviour.
        declare('executor_threads', 1)

        declare('gyro_bias_walk_sigma_rad_s', 1.0e-5)
        declare('gyro_noise_sigma_rad_s', 1.0e-3)
        declare('coast_on_reject', True)
        declare('twist_lowpass_tau', 0.12)
        declare('triangulation_min_parallax_deg', 1.0)
        declare('triangulation_reprojection_error_px', 2.5)
        declare('triangulation_min_distance_m', 0.5)
        declare('triangulation_max_distance_m', 30.0)
        declare('obstacle_min_height_m', 0.2)
        declare('obstacle_max_height_m', 2.5)
        # Travel a feature must accumulate before its ground projection's slip
        # is read as a height. 0 keeps the keyframe triangulation instead.
        declare('obstacle_slip_baseline_m', 0.0)
        # A feature at the camera's own height projects to infinity, so the
        # band stops short of it.
        declare('obstacle_height_margin', 0.7)
        # Mapping steps a feature may be missing from before its accumulated
        # baseline is abandoned.
        declare('obstacle_slip_patience', 30)
        declare('mapping_baseline_m', 0.35)
        declare('mapping_min_period_sec', 0.04)
        declare('feature_refill_ratio', 0.7)
        # Cells across and down for spreading detections; 0 keeps the plain
        # quality ordering. The oversample is how many candidates to ask the
        # detector for per wanted feature, since the quota throws some away.
        declare('detection_grid_columns', 0)
        declare('detection_grid_rows', 0)
        declare('detection_oversample', 3)
        # How many ground anchors to carry and for how many frames a feature
        # may go unseen before its anchor is dropped.
        declare('max_ground_anchors', 4000)
        declare('anchor_max_age_frames', 120)
        declare('anchor_update_gain', 0.0)
        declare('anchor_max_observations', 20)
        declare('anchor_select_by_consistency', True)
        declare('anchor_seed_travel_m', 0.0)
        # Solve every Nth frame instead of every one. A faster camera shortens
        # the baseline each solve gets, and the ground displacement it has to
        # measure falls with it; this separates the rate the data arrives at
        # from the baseline the estimate is built on.
        declare('frame_decimation', 1)
        # Trigger a solve on accumulated image motion instead of a frame count.
        declare('adaptive_solve_interval', True)
        declare('solve_trigger_disparity_px', 3.0)
        declare('solve_min_frames', 2)
        declare('solve_max_frames', 40)
        # The backward check runs the flow solver a second time, doubling the
        # only cost that matters. It earns that over a long baseline; over a
        # single frame the hop is small by construction and the status and
        # error the forward pass already returns say most of what it would.
        declare('forward_backward_on_hops', True)
        # Run the backward check on every Nth hop rather than all of them.
        # Dropping it entirely let bad tracks live to the solve and cost
        # 0.103 -> 0.733 m; the question is whether checking half as often
        # still catches them before they matter.
        declare('backward_check_stride', 1)
        # Take tracks from the C++ front end instead of doing optical flow
        # here. Empty keeps the original path, which is the one every measured
        # number in this package came from.
        declare('track_topic_prefix', '')
        # Camera input is best effort by default, which is right on a vehicle:
        # a late frame is worth less than the one behind it. For measurement it
        # is wrong, because the frames DDS drops under load differ from run to
        # run, and two runs over the same road then see different data. Set
        # this to reliable, with a queue deep enough to ride out a stall, when
        # comparing configurations. The publisher has to match.
        declare('input_reliability', 'best_effort')
        declare('input_queue_depth', 5)
        # KEEP_LAST or KEEP_ALL. RELIABLE does not mean lossless: it means
        # what the reader keeps is retransmitted until acknowledged, and a
        # KEEP_LAST reader that falls more than `depth` messages behind
        # overwrites its own backlog. That is where 675 of 810 track messages
        # went while every counter upstream said 809 were published. KEEP_ALL
        # makes the writer hold them instead -- correct offline, where a late
        # frame is better than a missing one, and wrong on the vehicle, where
        # it is the other way round.
        declare('input_history', 'keep_last')
        # A volatile reader gets nothing that was published before the writer
        # matched it, and matching is not instantaneous even when discovery
        # says the subscription exists. Transient local hands the backlog over
        # on match, which is how the last few dozen frames of a replay stop
        # depending on how fast the graph came up.
        declare('input_durability', 'volatile')
        # The IMU link. Best effort is right on the vehicle -- a late angular
        # rate is worse than none -- but it is not a measurement: a run that
        # loses samples here holds odometry and reports the nearest IMU sample
        # 16 s from the camera stamps, which is exactly how one replay in four
        # used to die. Reliable for benchmarking.
        declare('imu_reliability', 'best_effort')
        declare('imu_queue_depth', 200)
        # How many published poses the writer keeps for the evaluator. Ten at
        # 30 Hz is a third of a second of slack.
        declare('odometry_queue_depth', 10)
        # Hold the pose until the anchor map can answer for itself. The first
        # frames have no map, so they fall back to the two-frame solve, and on
        # a vehicle that has not moved yet that solve reads ground texture
        # noise as travel: measured, 38 cm of it inside the first 0.7 s, which
        # then rides along as a constant offset for the rest of the run.
        # Rotation is unaffected and keeps integrating.
        declare('require_map_before_translating', True)
        # Whether to stay silent until the map is ready rather than publish a
        # pose held at the origin. Set false to restore the old behaviour.
        declare('suppress_pose_until_map_ready', True)
        # A standing vehicle still produces a vision solve, because tracking a
        # texture is never exact and the residual reads as travel. Measured on
        # a live run: 60 cm accumulated over the three seconds before the
        # manoeuvre began, which then rode along as an offset for the whole
        # drive. The accelerometer knows the difference, so it decides.
        # Off. The test now measures disparity on the single-frame tracking
        # hop, where the near ground is still in view, which is the only place
        # it means anything: on the solve pair only distant features survive
        # the sixteen frames, and they barely move whatever the vehicle does.
        # Earlier versions read a moving car as parked and held the estimate at
        # a tenth of the ground covered. Corrected, it stops misfiring -- and
        # also stops showing any gain, because the 0.124 m it once appeared to
        # buy was the wrong version suppressing motion that was real. Left off
        # until there is a measurement that says otherwise.
        declare('zero_velocity_update', False)
        declare('zero_velocity_gyro_dps', 0.5)
        declare('zero_velocity_accel_mps2', 0.15)
        # An accelerometer cannot tell rest from constant velocity: both read
        # gravity and nothing else. Asked on its own it called 144 of 495
        # solves stationary on a run held at 8 m/s. Vision has to agree.
        declare('zero_velocity_speed_mps', 0.8)
        # Borrowed from OpenVINS, which asks the images rather than the
        # accelerometer whether the platform is moving. Features that do not
        # shift between two frames mean the vehicle did not move, and unlike a
        # specific force reading that statement stays true at constant
        # velocity. Pixels, averaged over the tracked set.
        declare('zero_velocity_disparity_px', 0.5)
        # Specific force below which the accelerometer is reporting nothing at
        # all rather than gravity. CARLA's IMU drops to zero while the actor
        # is static -- 1.9% of this recording's samples -- and every one of
        # those windows is a genuine standstill. On a vehicle a reading of
        # zero g means free fall or a dead sensor, so this stays off unless
        # the recording is known to behave that way.
        declare('zero_velocity_dropout_mps2', 0.0)
        # How much of the window has to be still. Relaxing this to 0.8 let
        # the check fire five or six times a run instead of one to three, and
        # the extra firings were wrong: ATE went from 0.046-0.055 to
        # 0.067-0.070 because real motion was being zeroed. The impulses that
        # veto a window are a nuisance, but a conservative zero-velocity
        # check is worth more than a complete one.
        declare('zero_velocity_quiet_fraction', 1.0)
        # Recognising a landmark that optical flow lost, so the map can outlive
        # an unbroken track. Off, because it does not work on this surface:
        # of the ground points that came back within half a metre of a stored
        # anchor, 85 % had a second candidate that looked just as much like
        # them, and taking the winner anyway cost accuracy at every threshold
        # tried (RMSE 0.124 m off, 0.247 at 40 bits, 0.170 at 64, 0.178 at 96).
        # Asphalt is self-similar at the scale of a 31 pixel patch, so a corner
        # on it carries almost no identity. Turning this on is worth revisiting
        # only against features that are actually distinctive -- painted lines,
        # kerb corners, drain covers -- which is a different detector, not a
        # different threshold.
        declare('anchor_reassociation', False)
        declare('anchor_reassociation_radius_m', 0.5)
        declare('anchor_reassociation_max_hamming', 40)
        declare('anchor_reassociation_min_observations', 3)
        # The accelerometer predicts how far the vehicle moved between frames.
        # A vision answer further off than this is the collapse to near zero
        # motion that used to pass unchallenged.
        declare('use_inertial_prediction', True)
        declare('inertial_use_acceleration', True)
        declare('inertial_gate_absolute_m', 0.08)
        declare('inertial_gate_relative', 0.35)
        declare('inertial_velocity_gain', 0.6)
        # Below this the prediction says nothing worth acting on.
        declare('inertial_gate_minimum_m', 0.05)
        declare('inertial_max_acceleration_mps2', 12.0)
        declare('inertial_acceleration_median_window', 1)
        declare('inertial_integration_method', 'rk4')
        # Kalman filter over planar velocity: how much unmodelled acceleration
        # to expect, and how noisy a vision solve carrying the reference inlier
        # count is.
        declare('filter_acceleration_noise', 1.55)
        declare('filter_vision_noise', 0.25)
        declare('filter_reference_inliers', 300.0)
        declare('filter_innovation_gate', 9.0)
        # How hard front/rear disagreement counts against a solve, and the
        # penalty when only one camera produced an answer at all.
        declare('camera_disagreement_weight', 1.0)
        declare('single_camera_variance', 0.5)
        # Weight cameras by the precision of their solve rather than by how
        # many ground points each happened to see.
        declare('fuse_cameras_by_spread', False)

        get = self.get_parameter
        self.map_frame = str(get('map_frame').value)
        self.base_frame = str(get('base_frame').value)
        self.extrinsics_from_tf = bool(get('extrinsics_from_tf').value)
        self.tf_lookup_timeout = float(get('tf_lookup_timeout_sec').value)
        self.tf_buffer = Buffer() if self.extrinsics_from_tf else None
        # On its own thread. /tf_static arrives transient-local, so a node
        # that has just come up takes the whole backlog at once, and on the
        # single-threaded executor that shares the queue with the cameras:
        # the first run of a pair came back with 177 hops of 280, every time,
        # until the listener stopped competing with them.
        self.tf_listener = (
            TransformListener(self.tf_buffer, self, spin_thread=True)
            if self.tf_buffer else None)
        self.sync_tolerance = max(float(get('sync_tolerance_sec').value), 0.0)
        self.use_imu_yaw = bool(get('use_imu_yaw').value)
        self.imu_max_age = max(float(get('imu_max_age_sec').value), 0.0)
        self.imu_max_gap = max(float(get('imu_max_gap_sec').value), 0.0)
        self.allow_fov_fallback = bool(get('allow_fov_fallback').value)
        self.processing_width = max(int(get('processing_width').value), 0)
        self.max_features = int(get('max_features_per_camera').value)
        self.quality_level = float(get('quality_level').value)
        self.min_feature_distance = float(get('min_feature_distance_px').value)
        self.lk_error_threshold = float(get('lk_error_threshold').value)
        self.lk_window = int(get('lk_window_px').value)
        self.lk_levels = int(get('lk_pyramid_levels').value)
        self.lk_forward_backward_threshold = float(
            get('lk_forward_backward_threshold_px').value
        )
        self.ground_max_distance = float(get('ground_max_distance_m').value)
        self.min_ground_flow_px = float(get('min_ground_flow_px').value)
        self.last_hop_dt = 0.0
        self.last_band = 0.0
        self.ground_ransac_threshold = float(get('ground_ransac_threshold_m').value)
        self.ground_min_inliers = int(get('ground_min_inliers').value)
        self.max_scale_error = float(get('max_scale_error').value)
        self.max_translation = float(get('max_translation_per_frame_m').value)
        self.max_yaw = float(get('max_yaw_per_frame_rad').value)
        self.max_arrears_frames = float(get('max_arrears_frames').value)
        self.heading = HeadingBiasFilter(
            float(get('gyro_bias_sigma_rad_s').value),
            float(get('gyro_bias_walk_sigma_rad_s').value),
            float(get('gyro_noise_sigma_rad_s').value),
        )
        self.yaw_gain_sum = 0.0
        self.yaw_gain_count = 0
        self.heading_observations = []
        self.yaw_correction = 0.0
        self.coast_on_reject = bool(get('coast_on_reject').value)
        self.twist_tau = max(float(get('twist_lowpass_tau').value), 0.0)
        self.min_parallax = float(get('triangulation_min_parallax_deg').value)
        self.max_reprojection_error = float(
            get('triangulation_reprojection_error_px').value
        )
        self.triangulation_min_distance = float(
            get('triangulation_min_distance_m').value
        )
        self.triangulation_max_distance = float(
            get('triangulation_max_distance_m').value
        )
        self.obstacle_min_height = float(get('obstacle_min_height_m').value)
        self.obstacle_slip_baseline = float(get('obstacle_slip_baseline_m').value)
        self.obstacle_height_margin = float(get('obstacle_height_margin').value)
        self.obstacle_slip_patience = int(get('obstacle_slip_patience').value)
        self.obstacle_max_height = float(get('obstacle_max_height_m').value)
        # Casting a Bresenham ray per ground inlier does not fit in a 20 Hz
        # budget, so only every nth inlier carves free space.
        self.mapping_baseline = float(get('mapping_baseline_m').value)
        # The occupancy grid is the deliverable, not part of the pose loop, and
        # it cost more per frame than the odometry it rode along with. A
        # parking grid does not improve by being rebuilt at the camera rate, so
        # it runs on its own clock and the estimator keeps the frame budget.
        self.mapping_min_period = float(get('mapping_min_period_sec').value)
        self.last_mapping_stamp = None
        # Re-detecting every frame rebuilds a mask and rescans the whole image
        # to replace the handful of tracks lost since last time. Waiting until
        # the set has actually thinned costs nothing while it is still full.
        self.feature_refill_ratio = float(get('feature_refill_ratio').value)
        columns = int(get('detection_grid_columns').value)
        rows = int(get('detection_grid_rows').value)
        self.detection_grid = (columns, rows) if columns > 0 and rows > 0 else None
        self.detection_oversample = max(int(get('detection_oversample').value), 1)
        self.max_anchors = int(get('max_ground_anchors').value)
        self.anchor_max_age = int(get('anchor_max_age_frames').value)
        # How fast an anchor follows its latest sighting. Sightings are written
        # in the world frame the estimate just settled on, so an anchor that
        # follows too eagerly absorbs the estimate's own error and stops being
        # able to correct it. What matters is how much the anchor integrates
        # per metre travelled, not per frame.
        self.anchor_update_gain = float(get('anchor_update_gain').value)
        # Where an anchor's weight stops growing. Reached in a third of a
        # second at 60 Hz, it stops telling a long-lived landmark apart from a
        # freshly seeded one, which is the whole point of the weighting.
        self.anchor_max_observations = int(get('anchor_max_observations').value)
        self.anchor_selection = bool(get('anchor_select_by_consistency').value)
        self.anchor_seed_travel = float(get('anchor_seed_travel_m').value)
        self.frame_decimation = max(int(get('frame_decimation').value), 1)
        self.adaptive_solve = bool(get('adaptive_solve_interval').value)
        self.solve_trigger_disparity = float(get('solve_trigger_disparity_px').value)
        self.solve_min_frames = max(int(get('solve_min_frames').value), 1)
        self.solve_max_frames = max(int(get('solve_max_frames').value), 1)
        self.check_backward = bool(get('forward_backward_on_hops').value)
        self.backward_stride = max(int(get('backward_check_stride').value), 1)
        self.track_prefix = str(get('track_topic_prefix').value)
        self.frames_since_solve = 0
        self.frames_in_last_solve = 1
        self.last_solve_dt = 0.0
        # OpenCV releases the interpreter lock inside the flow solver, so the
        # two cameras genuinely overlap rather than taking turns. Tracking is
        # the whole cost here: online it managed 6.6 of the 21.9 pairs a second
        # the bridge delivered, and everything downstream inherits that.
        # Leave OpenCV's own thread count alone. Halving it so the two
        # tracking threads do not oversubscribe the machine sounds right and
        # measures wrong: online ingest fell from 20-25 to 15.2 pairs a second.
        # One worker means the two cameras run one after the other on the
        # calling thread's behalf. Worth measuring rather than assuming: the
        # work per camera is mostly numpy and Python, which hold the
        # interpreter lock, so two workers can only overlap the fraction that
        # is inside OpenCV or the extension, and pay for the handoff on all
        # of it.
        self.trackers = ThreadPoolExecutor(
            max_workers=max(int(get('camera_workers').value), 1))
        self._stamp_offsets = None
        self._stamp_radius = -1
        self.require_map = bool(get('require_map_before_translating').value)
        self.suppress_warmup_pose = bool(
            get('suppress_pose_until_map_ready').value
        )
        self.map_ready = not self.require_map
        self.use_zupt = bool(get('zero_velocity_update').value)
        self.zupt_gyro = math.radians(float(get('zero_velocity_gyro_dps').value))
        self.zupt_accel = float(get('zero_velocity_accel_mps2').value)
        self.zupt_speed = float(get('zero_velocity_speed_mps').value)
        self.zupt_disparity = float(get('zero_velocity_disparity_px').value)
        self.zupt_dropout = float(get('zero_velocity_dropout_mps2').value)
        self.zupt_quiet_fraction = float(get('zero_velocity_quiet_fraction').value)
        self.reassociate = bool(get('anchor_reassociation').value)
        self.reassociation_radius = float(get('anchor_reassociation_radius_m').value)
        self.reassociation_hamming = int(get('anchor_reassociation_max_hamming').value)
        self.reassociation_min_obs = int(
            get('anchor_reassociation_min_observations').value
        )
        self.describer = cv2.ORB_create(nfeatures=4000) if self.reassociate else None
        self.imu_motion_samples = deque(maxlen=4000)
        # World-frame, gravity-free horizontal acceleration with its own
        # timestamp. The filter integrates from this over the interval a
        # vision measurement actually spans, instead of integrating in
        # arrival order and hoping the two windows coincide -- which they do
        # not, because a pair is processed well after the instant it is
        # stamped at.
        self.imu_window = deque(maxlen=8000)
        self.zupt_holds = 0
        self.last_disparity = None
        self.last_seed_position = None
        self.map_aligned_frames = 0
        self.use_inertial = bool(get('use_inertial_prediction').value)
        self.inertial_use_acceleration = bool(
            get('inertial_use_acceleration').value
        )
        self.inertial_gate_absolute = float(get('inertial_gate_absolute_m').value)
        self.inertial_gate_relative = float(get('inertial_gate_relative').value)
        self.inertial_gain = float(get('inertial_velocity_gain').value)
        self.inertial_gate_minimum = float(get('inertial_gate_minimum_m').value)
        self.inertial = PlanarInertialPropagator(
            max_horizontal_acceleration=float(
                get('inertial_max_acceleration_mps2').value
            ),
            median_window=int(get('inertial_acceleration_median_window').value),
            integration_method=str(get('inertial_integration_method').value),
        )
        self.fusion_model = str(get('fusion_model').value)
        self.displacement_filter = PlanarDisplacementFilter(
            acceleration_noise=float(get('filter_acceleration_noise').value),
            bias_walk=float(get('filter_bias_walk').value),
            vision_noise_m=float(get('filter_vision_noise_m').value),
            vision_reference_inliers=float(get('filter_reference_inliers').value),
            innovation_gate=float(get('filter_innovation_gate').value),
        ) if str(get('fusion_model').value) == 'displacement' else None
        self.velocity_filter = PlanarVelocityFilter(
            acceleration_noise=float(get('filter_acceleration_noise').value),
            vision_noise=float(get('filter_vision_noise').value),
            vision_reference_inliers=float(get('filter_reference_inliers').value),
            innovation_gate=float(get('filter_innovation_gate').value),
        )
        self.filter_rejections = 0
        self.disagreement_weight = float(get('camera_disagreement_weight').value)
        self.single_camera_variance = float(get('single_camera_variance').value)
        self.camera_disagreement = 0.0
        self.fuse_by_spread = bool(get('fuse_cameras_by_spread').value)

        self.cameras: Dict[str, CameraRuntime] = {}
        for name in self.camera_names:
            k = np.asarray(get(f'{name}.k').value, dtype=np.float64).reshape(3, 3)
            rotation = np.asarray(
                get(f'{name}.rotation_base_from_camera').value, dtype=np.float64
            ).reshape(3, 3)
            translation = np.asarray(
                get(f'{name}.translation_base_from_camera').value, dtype=np.float64
            )
            distortion = np.asarray(
                get(f'{name}.d').value, dtype=np.float64
            )
            self._validate_rotation(name, rotation)
            model = CameraModel(
                k,
                rotation,
                translation,
                distortion,
                str(get(f'{name}.distortion_model').value),
            )
            self.cameras[name] = CameraRuntime(
                name,
                model,
                float(get(f'{name}.horizontal_fov_deg').value),
                ground_min_distance=float(
                    get(f'{name}.ground_min_distance_m').value
                ),
                range_scale=float(get(f'ground_range_scale_{name}').value),
                calibration_model=model,
                calibration_width=int(get(f'{name}.calibration_width').value),
                calibration_height=int(get(f'{name}.calibration_height').value),
            )

        self.pose = Pose2()
        self.pending_points = []
        self.pair_queue_depth = int(get('pair_queue_depth').value)
        path = str(get('filter_debug_csv').value)
        self.filter_debug = open(path, 'w') if path else None
        if self.filter_debug is not None:
            self.filter_debug.write(
                'began,ended,measured_x,measured_y,predicted_x,predicted_y,'
                'innovation_x,innovation_y,variance,prior_trace,nis,accepted,'
                'spread,extra,disparity,band,inliers,posterior_x,posterior_y\n')
        self.solve_timer = None
        self.zupt_velocity_sigma = float(get('zupt_velocity_sigma').value)
        self.accel_time_offset = float(get('imu_acceleration_offset_sec').value)
        self.executor_threads = int(get('executor_threads').value)
        self.groups = {
            key: MutuallyExclusiveCallbackGroup()
            for key in list(self.camera_names) + ['imu', 'solve']
        }
        self.frames_evicted = 0
        self.tracks_received = {name: 0 for name in self.camera_names}
        self.camera_infos = {name: 0 for name in self.camera_names}
        self.tracks_raw = {name: 0 for name in self.camera_names}
        self.pending_frames: Dict[str, Deque[Frame]] = {
            name: deque() for name in self.camera_names
        }
        self.imu_yaw_samples = deque(maxlen=400)
        self.filtered_twist = np.zeros(3, dtype=np.float64)
        # Last accepted motion, used to predict where features land next.
        self.last_motion: Optional[PlanarMotion] = None
        self.last_stamp_msg = None
        # What actually arrived since the last status line, by source. Rates
        # taken from a bag recorded alongside a live run are not trustworthy
        # -- the recorder competes for the same CPU and drops, and the same
        # camera_info read 60.0 Hz in one run and 47 in the next. The node
        # counting its own inputs is the only figure that is not confounded.
        self.arrivals = dict.fromkeys(list(self.camera_names) + ['imu'], 0)
        self.arrival_since = None
        self.last_status_time = 0.0
        self.frames_processed = 0
        self.pairs_seen = 0
        self.ingest_seconds = 0.0
        self.ingest_count = 0
        self.process_seconds = 0.0
        self.stage = {
            k: 0.0 for k in
            ('lk', 'detect', 'pair', 'ground', 'lookup', 'align', 'solve',
             'rest', 'anchor', 'map')}
        self.motion_failures = 0
        # Why the translation was thrown away, split three ways. A pair that
        # solved and was then gated is a different problem from one that never
        # solved at all, and the count alone cannot tell them apart.
        self.fail_no_solve = 0
        self.fail_translation = 0
        self.fail_yaw = 0
        # When a translation was last accepted. The map solve returns an
        # absolute position, so what it reports as motion also carries back
        # whatever the rejected frames left behind.
        self.last_accept_stamp = None
        self.worst_rejected = 0.0
        self.worst_allowance = 0.0
        self.coasted = 0
        self.yaw_only_updates = 0
        self.imu_yaw_misses = 0

        sensor_qos = QoSProfile(
            depth=max(int(get('input_queue_depth').value), 1),
            history=(
                HistoryPolicy.KEEP_ALL
                if str(get('input_history').value) == 'keep_all'
                else HistoryPolicy.KEEP_LAST
            ),
            durability=(
                DurabilityPolicy.TRANSIENT_LOCAL
                if str(get('input_durability').value) == 'transient_local'
                else DurabilityPolicy.VOLATILE
            ),
            reliability=(
                ReliabilityPolicy.RELIABLE
                if str(get('input_reliability').value) == 'reliable'
                else ReliabilityPolicy.BEST_EFFORT
            ),
        )
        reliable_qos = QoSProfile(
            depth=max(int(get('odometry_queue_depth').value), 1),
            reliability=ReliabilityPolicy.RELIABLE)
        map_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        if self.track_prefix:
            for name in self.camera_names:
                self.create_subscription(
                    Float32MultiArray,
                    f'{self.track_prefix}/{name}',
                    lambda msg, n=name: self._on_tracks(n, msg),
                    sensor_qos,
                    callback_group=self.groups[name],
                )
        if not self.track_prefix:
            for name in self.camera_names:
                self.create_subscription(
                    Image,
                    str(get(f'{name}_image_topic').value),
                    lambda msg, n=name: self._on_image(n, msg),
                    sensor_qos,
                    callback_group=self.groups[name],
                )
        if self.use_imu_yaw:
            # Tracking a frame pair blocks this node for tens of milliseconds,
            # which is several IMU samples. A shallow queue silently drops them
            # and the yaw over that interval is then unrecoverable.
            self.create_subscription(
                Imu,
                str(get('imu_topic').value),
                self._on_imu,
                QoSProfile(
                    depth=max(int(get('imu_queue_depth').value), 1),
                    reliability=(
                        ReliabilityPolicy.RELIABLE
                        if str(get('imu_reliability').value) == 'reliable'
                        else ReliabilityPolicy.BEST_EFFORT
                    ),
                ),
                callback_group=self.groups['imu'],
            )
        self.imu_stamp = None
        self.attitude_filter = (
            AttitudeFilter(
                float(get('attitude_tau_sec').value),
                float(get('attitude_gravity_tolerance').value))
            if bool(get('attitude_from_imu').value) else None)
        self.use_camera_info = bool(get('use_camera_info').value)
        if self.use_camera_info:
            for name in self.camera_names:
                self.create_subscription(
                    CameraInfo,
                    str(get(f'{name}_camera_info_topic').value),
                    lambda msg, n=name: self._on_camera_info(n, msg),
                    sensor_qos,
                )
        self.odom_pub = self.create_publisher(
            Odometry, str(get('odometry_topic').value), reliable_qos
        )
        self.pose_pub = self.create_publisher(
            PoseStamped, str(get('pose_topic').value), reliable_qos
        )
        self.points_pub = self.create_publisher(
            PointCloud2, str(get('ground_points_topic').value), map_qos
        )
        self.tf_broadcaster = (
            TransformBroadcaster(self) if bool(get('publish_tf').value) else None
        )
        if self.tf_broadcaster is not None:
            self.create_timer(0.05, self._publish_transform)

        solve_hz = float(get('solve_timer_hz').value)
        if solve_hz > 0.0:
            self.solve_timer = self.create_timer(
                1.0 / solve_hz, self._try_process_pair,
                callback_group=self.groups['solve'])

    @staticmethod
    def _validate_rotation(name, rotation):
        error = np.linalg.norm(rotation.T @ rotation - np.eye(3))
        if error > 1e-3 or np.linalg.det(rotation) < 0.99:
            raise ValueError(f'{name} camera rotation is not a proper rotation matrix')

    def _extrinsics_from_tf(self, name: str, frame_id: str) -> bool:
        """Take the mount off the transform tree. True once it is there.

        Asked once per camera and then cached: where a camera is bolted does
        not change while the vehicle drives. A rig with no tree keeps the
        parameters, and says which it used rather than leaving it to be
        guessed from behaviour.
        """
        runtime = self.cameras[name]
        if runtime.mounted or self.tf_buffer is None or not frame_id:
            return runtime.mounted
        try:
            found = self.tf_buffer.lookup_transform(
                self.base_frame, frame_id, rclpy.time.Time())
        except (LookupException, ConnectivityException, ExtrapolationException):
            self.get_logger().warning(
                f'No transform {self.base_frame} <- {frame_id} for {name}; '
                'using the configured mount',
                throttle_duration_sec=self.tf_lookup_timeout)
            return False
        q = found.transform.rotation
        translation = np.array([
            found.transform.translation.x,
            found.transform.translation.y,
            found.transform.translation.z], dtype=np.float64)
        rotation = quaternion_matrix((q.x, q.y, q.z, q.w))
        self._validate_rotation(name, rotation)
        runtime.model = replace(
            runtime.model, rotation_base_from_camera=rotation,
            translation_base_from_camera=translation)
        if runtime.calibration_model is not None:
            runtime.calibration_model = replace(
                runtime.calibration_model, rotation_base_from_camera=rotation,
                translation_base_from_camera=translation)
        runtime.mounted = True
        self.get_logger().info(
            f'{name} mounted from TF: {self.base_frame} <- {frame_id}')
        return True

    def _on_camera_info(self, name: str, msg: CameraInfo):
        self.camera_infos[name] += 1
        self._extrinsics_from_tf(name, msg.header.frame_id)
        k = np.asarray(msg.k, dtype=np.float64).reshape(3, 3)
        if msg.width == 0 or msg.height == 0 or k[0, 0] <= 0.0 or k[1, 1] <= 0.0:
            self.get_logger().warning(
                f'Ignoring invalid {name} CameraInfo; using configured K/FOV',
                throttle_duration_sec=5.0,
            )
            return
        runtime = self.cameras[name]
        runtime.calibration_model = CameraModel(
            k,
            runtime.model.rotation_base_from_camera,
            runtime.model.translation_base_from_camera,
            np.asarray(msg.d, dtype=np.float64),
            msg.distortion_model,
        )
        runtime.calibration_width = int(msg.width)
        runtime.calibration_height = int(msg.height)

    def _frame_model(self, runtime: CameraRuntime, width: int, height: int):
        """Intrinsics for pixels measured in a frame of the given size.

        The calibration describes the camera at its own resolution while the
        estimator works on a downscaled frame. Pairing one with the other is
        not a clean scale error -- the principal point moves as well, so the
        whole image projects to a shifted patch of ground -- and it is the
        kind of mistake that leaves the scale looking almost right. Both the
        image path and the track path go through here so they cannot drift
        apart.
        """
        calibration = runtime.calibration_model
        k = calibration.k.copy()
        if runtime.calibration_width > 0 and runtime.calibration_height > 0:
            k[0, :] *= width / float(runtime.calibration_width)
            k[1, :] *= height / float(runtime.calibration_height)
        return CameraModel(
            k,
            calibration.rotation_base_from_camera,
            calibration.translation_base_from_camera,
            calibration.distortion,
            calibration.distortion_model,
        )

    def _on_tracks(self, name: str, msg: Float32MultiArray):
        """Take a hop that the C++ front end already followed.

        The layout is flat: stamp, count, the size of the frame the pixels
        were measured in, then id and both pixel positions for each surviving
        feature. Everything the estimator does with them -- ground projection,
        the anchor map, registration -- is unchanged; only the tracking has
        moved.
        """
        # Counted before any validity check, so a message rejected here can
        # be told apart from one that never arrived.
        self.tracks_raw[name] += 1
        data = np.asarray(msg.data, dtype=np.float64)
        if len(data) < 4:
            return
        stamp = float(data[0])
        count = int(data[1])
        width = int(data[2])
        height = int(data[3])
        if stamp <= 0.0 or width <= 0 or height <= 0 or len(data) < 4 + 5 * count:
            return
        self.arrivals[name] += 1
        self.tracks_received[name] += 1
        runtime = self.cameras[name]
        # The front end sends pixels in its own downscaled frame, so the
        # intrinsics have to be brought to that frame before anything is
        # projected. Doing this only in the image path is what left this path
        # at 5.92 m while the image path gave 0.26 m on the same run.
        runtime.model = self._frame_model(runtime, width, height)
        runtime.sequence += 1
        rows = data[4:4 + 5 * count].reshape(-1, 5) if count else np.zeros((0, 5))
        ids = rows[:, 0].astype(np.int64)
        pixels = rows[:, 3:5].copy()
        disparity = 0.0
        if count:
            shift = np.linalg.norm(rows[:, 3:5] - rows[:, 1:3], axis=1)
            disparity = float(np.percentile(shift, 90))
        seconds = int(stamp)
        stamp_msg = Header().stamp
        stamp_msg.sec = seconds
        stamp_msg.nanosec = int(round((stamp - seconds) * 1e9))
        frame = Frame(stamp, stamp_msg, None, runtime.sequence, disparity, pixels, ids)
        runtime.previous = runtime.latest if runtime.latest is not None else frame
        self._queue_frame(name, frame)
        # Queue only. Solving here is what let the two cameras drift apart:
        # rclpy spins one thread, a solve costs ~22 ms, and two 30 Hz streams
        # ask for 60 callbacks a second. The executor cannot serve both at
        # that price, and whichever subscription it favours drains while the
        # other backs up -- measured at 5.4 s of skew, after which no frame
        # finds a partner and the run is lost. Ingestion is now cheap and
        # symmetric; the work happens on the timer below, against whatever
        # has arrived.
        if self.solve_timer is None:
            self._try_process_pair()


    def _ground_band(self, runtime: CameraRuntime) -> float:
        """How far out the ground band is worth reading, for this speed.

        A ground point at distance d moves fx*h*s/d^2 pixels over a hop that
        covered s metres. Where that falls under a pixel the tracker returns
        short of the truth rather than noisily around it, and since every such
        point pulls the same way the whole solve comes out short. Inverting for
        d puts the useful edge at sqrt(fx*h*s/flow), which grows with the root
        of speed -- so one fixed distance can only suit one speed.

        Measured on the four drives: the band that scored best was 6 m at
        7.5 m/s, 4 m at 3.0 and 3 m at 2.3, and reading a threshold back out of
        each gives 0.93 to 1.14 px. The 8 m the parameter file carried is what
        this expression returns at 8 m/s, which is the speed it was set at.
        """
        if self.min_ground_flow_px <= 0.0:
            return self.ground_max_distance
        # Falling back to the declared maximum whenever the speed is not known
        # throws the band wide open exactly when it should not be: a manoeuvre
        # with stops in it hits this on every pause, and park scored 0.172 m
        # against 0.025 at its best fixed band until this held the last value
        # instead.
        fallback = self.last_band if self.last_band > 0.0 else self.ground_max_distance
        if self.last_motion is None or self.last_solve_dt <= 1e-6:
            return fallback
        if self.last_hop_dt <= 0.0:
            return fallback
        speed = math.hypot(self.last_motion.x, self.last_motion.y) / self.last_solve_dt
        fx = float(runtime.model.k[0, 0])
        height = float(runtime.model.translation_base_from_camera[2])
        if speed <= 0.0 or fx <= 0.0 or height <= 0.0:
            return fallback
        hop = speed * self.last_hop_dt
        limit = math.sqrt(fx * height * hop / self.min_ground_flow_px)
        # Never below the near limit, or the band closes on itself.
        floor = 2.0 * runtime.ground_min_distance
        band = float(min(self.ground_max_distance, max(floor, limit)))
        self.last_band = band
        return band

    def _body_tilt(self):
        """Roll and pitch as a rotation from the body into a level frame.

        None while there is no attitude to be had, which keeps the level
        assumption the whole estimate used to be built on. Yaw is left out:
        the planar solve owns it, and putting it here as well would rotate
        the ground twice.
        """
        if self.attitude_filter is None or not self.attitude_filter.started:
            return None
        roll, pitch = self.attitude_filter.roll, self.attitude_filter.pitch
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        return np.array([
            [cp, sp * sr, sp * cr],
            [0.0, cr, -sr],
            [-sp, cp * sr, cp * cr],
        ], dtype=np.float64)


    def _queue_frame(self, name: str, frame) -> None:
        """Queue a frame, and say so out loud if one has to be thrown away."""
        queue = self.pending_frames[name]
        queue.append(frame)
        while len(queue) > self.pair_queue_depth:
            queue.popleft()
            self.frames_evicted += 1

    def _closest_frames(self, queues):
        """One frame per camera, picked so their stamps sit closest together.

        Every queued frame of every camera gets a turn as the reference and
        the others answer with whichever frame lands nearest it; the reference
        whose answers spread least wins. With two cameras this returns the
        same pair the exhaustive search over both queues did, and it stays
        affordable when there are more of them than two.
        """
        best = None
        for queue in queues:
            for frame in queue:
                picks = [
                    min(range(len(other)),
                        key=lambda k, o=other: abs(o[k].stamp - frame.stamp))
                    for other in queues
                ]
                stamps = [o[k].stamp for o, k in zip(queues, picks)]
                spread = max(stamps) - min(stamps)
                if best is None or spread < best[0]:
                    best = (spread, picks, stamps)
        return best

    def _try_process_pair(self):
        runtimes = [self.cameras[name] for name in self.camera_names]
        queues = [self.pending_frames[name] for name in self.camera_names]
        while all(queues):
            delta, picks, stamps = self._closest_frames(queues)
            if delta <= self.sync_tolerance:
                stamp = sum(stamps) / len(stamps)
                if self._imu_still_arriving(stamp):
                    # The bridge publishes this frame before the IMU sample that
                    # covers it. Leave the pair queued; _on_imu retries.
                    return
                # Frames dropped for want of a peer still moved the ground, so
                # their disparity counts towards the next solve even though
                # they are not themselves compared. Counting it here rather
                # than on arrival is what keeps the trigger in step with the
                # image path, which only ever measures a hop when it processes
                # the pair that closes it.
                for runtime, queue, index in zip(runtimes, queues, picks):
                    for _ in range(index + 1):
                        runtime.latest = queue.popleft()
                        runtime.hop_disparity += runtime.latest.disparity
                if self.track_prefix:
                    # Take the pixels that belong to the frame being processed,
                    # not whatever the camera has received since. Offline the
                    # two are always the same and this changes nothing; online
                    # they are not, and the solve was comparing a snapshot
                    # against pixels from a later frame than the one it was
                    # about to stamp -- 23% of the motion went missing that
                    # way.
                    for runtime in runtimes:
                        runtime.track_pixels = runtime.latest.pixels
                        runtime.track_ids = runtime.latest.ids
                # Decimate whole pairs, never single images. Skipping a pair
                # leaves `previous` where it was, so the next solve spans the
                # longer baseline; the frames themselves still pair normally.
                self.pairs_seen += 1
                self.frames_since_solve += 1
                if self.adaptive_solve and not self._ready_to_solve():
                    if not self.track_prefix:
                        list(self.trackers.map(self._advance_tracks, runtimes))
                    return
                if not self.adaptive_solve and self.pairs_seen % self.frame_decimation:
                    # Not a solve frame, but the features still have to be
                    # carried forward. Letting the tracker jump the whole
                    # decimation interval is what killed them: near ground
                    # sweeps hundreds of pixels in that time and the median
                    # feature survived two solves instead of the forty the
                    # geometry allows.
                    if not self.track_prefix:
                        list(self.trackers.map(self._advance_tracks, runtimes))
                    return
                self._process_pair(runtimes)
                return

            # No alignment inside tolerance: the oldest head is the one with
            # no peer coming, so it goes and the rest get another chance.
            oldest = min(range(len(queues)), key=lambda i: queues[i][0].stamp)
            runtimes[oldest].hop_disparity += queues[oldest].popleft().disparity
            self.get_logger().warning(
                f'Dropping camera frame with no synchronized peer: spread={delta:.4f}s',
                throttle_duration_sec=2.0,
            )

    def _on_imu(self, msg: Imu):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if stamp <= 0.0:
            return
        self.arrivals['imu'] += 1
        q = msg.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        self.imu_yaw_samples.append((stamp, yaw))
        if self.attitude_filter is not None:
            step = 0.0 if self.imu_stamp is None else stamp - self.imu_stamp
            if 0.0 <= step <= 0.1:
                self.attitude_filter.update(
                    (msg.angular_velocity.x, msg.angular_velocity.y,
                     msg.angular_velocity.z),
                    (msg.linear_acceleration.x, msg.linear_acceleration.y,
                     msg.linear_acceleration.z),
                    step)
            self.imu_stamp = stamp
        if self.use_zupt:
            self.imu_motion_samples.append(
                (
                    stamp,
                    math.sqrt(
                        msg.angular_velocity.x ** 2
                        + msg.angular_velocity.y ** 2
                        + msg.angular_velocity.z ** 2
                    ),
                    math.sqrt(
                        msg.linear_acceleration.x ** 2
                        + msg.linear_acceleration.y ** 2
                        + msg.linear_acceleration.z ** 2
                    ),
                )
            )
        if self.use_inertial:
            acceleration, step = self.inertial.add_sample(
                stamp,
                (q.x, q.y, q.z, q.w),
                (
                    msg.linear_acceleration.x,
                    msg.linear_acceleration.y,
                    msg.linear_acceleration.z,
                ),
            )
            if not self.inertial_use_acceleration:
                acceleration = np.zeros(2, dtype=np.float64)
            self.velocity_filter.predict(acceleration, step)
            if self.displacement_filter is not None:
                self.imu_window.append(
                    (stamp + self.accel_time_offset,
                     float(acceleration[0]), float(acceleration[1]),
                     step, yaw))
        self._try_process_pair()

    def _log_filter_update(self, began: float, ended: float,
                           spread: float, extra: float,
                           disparity, inliers: int) -> None:
        """Write down what the filter believed, for checking against truth.

        Nothing here changes the estimate; it exists because the process noise
        turned out to have a sharp optimum, and a sharp optimum in a parameter
        that should be a physical quantity means the model is absorbing an
        error rather than describing one.
        """
        if self.filter_debug is None:
            return
        last = self.displacement_filter.last_update
        if last is None:
            return
        self.filter_debug.write(
            f'{began:.6f},{ended:.6f},'
            f'{last["measured"][0]:.6f},{last["measured"][1]:.6f},'
            f'{last["predicted"][0]:.6f},{last["predicted"][1]:.6f},'
            f'{last["innovation"][0]:.6f},{last["innovation"][1]:.6f},'
            f'{last["variance"]:.9f},{last["prior_trace"]:.9f},'
            f'{last["nis"]:.4f},{int(last["accepted"])},{spread:.6f},'
            f'{extra:.9f},{-1.0 if disparity is None else disparity:.4f},'
            f'{self.last_band:.3f},{inliers},'
            f'{last["posterior"][0]:.6f},{last["posterior"][1]:.6f}\n')
        self.filter_debug.flush()

    def _imu_still_arriving(self, stamp: float) -> bool:
        """True while the IMU has not yet reached the given frame time."""
        if not self.use_imu_yaw or not self.imu_yaw_samples:
            return False
        return self.imu_yaw_samples[-1][0] < stamp

    def _imu_gap(self, stamp: float) -> float:
        if not self.imu_yaw_samples:
            return float('inf')
        return min(abs(sample[0] - stamp) for sample in self.imu_yaw_samples)

    def _imu_span(self) -> float:
        if len(self.imu_yaw_samples) < 2:
            return 0.0
        return self.imu_yaw_samples[-1][0] - self.imu_yaw_samples[0][0]

    def _imu_yaw_at(self, stamp: float) -> Optional[float]:
        """Yaw at an arbitrary time, interpolated between IMU samples.

        Taking the nearest sample only works while the IMU runs far faster than
        the cameras. Attached to an externally driven session the simulator is
        asynchronous, every sensor drops to the server frame rate, and nearest
        sample threw away two thirds of the frames.
        """
        samples = self.imu_yaw_samples
        if not samples:
            return None
        before = None
        after = None
        for sample in samples:
            if sample[0] <= stamp:
                before = sample
            else:
                after = sample
                break
        if before is not None and after is not None:
            span = after[0] - before[0]
            if 0.0 < span <= self.imu_max_gap:
                ratio = (stamp - before[0]) / span
                step = math.remainder(after[1] - before[1], 2.0 * math.pi)
                return before[1] + ratio * step
        nearest = min(samples, key=lambda item: abs(item[0] - stamp))
        return nearest[1] if abs(nearest[0] - stamp) <= self.imu_max_age else None

    def _describe(self, gray, pixels):
        """Binary descriptors at the given pixels.

        ORB drops keypoints whose patch runs off the edge of the image, so the
        result is mapped back onto the original order and the ones it declined
        are marked undescribed rather than silently shifting everything along.
        """
        if self.describer is None or len(pixels) == 0:
            return None, None
        keypoints = [cv2.KeyPoint(float(x), float(y), 31.0) for x, y in pixels]
        kept, descriptors = self.describer.compute(gray, keypoints)
        if descriptors is None or not len(kept):
            return None, None
        lookup = {}
        for index, (x, y) in enumerate(pixels):
            lookup.setdefault((round(float(x), 2), round(float(y), 2)), index)
        full = np.zeros((len(pixels), descriptors.shape[1]), dtype=np.uint8)
        valid = np.zeros(len(pixels), dtype=bool)
        for point, descriptor in zip(kept, descriptors):
            index = lookup.get((round(point.pt[0], 2), round(point.pt[1], 2)))
            if index is not None:
                full[index] = descriptor
                valid[index] = True
        return full, valid



    def _follow_tracks(self, runtime: CameraRuntime, previous, current):
        """Move the tracked pixels into the current frame, dropping the lost."""
        source = runtime.track_pixels.astype(np.float32).reshape(-1, 1, 2)
        window = (self.lk_window, self.lk_window)
        guess = None
        hop_dt = current.stamp - previous.stamp
        if hop_dt > 0.0:
            self.last_hop_dt = hop_dt
        if self.last_motion is not None and self.last_solve_dt > 1e-6 and hop_dt > 0.0:
            # last_motion spans a whole solve interval, but this is one frame's
            # hop. Feeding the full figure in overshoots by the number of
            # frames that interval covered and the tracker starts each search
            # in the wrong place.
            # Scale by the time this hop actually spans, not by an assumed
            # uniform frame interval. Online the frames arrive irregularly and
            # some are dropped, so a hop can be two or three frames wide; a
            # prediction built on the nominal interval then lands short and the
            # tracker searches in the wrong place at exactly the moments it can
            # least afford to.
            share = min(hop_dt / self.last_solve_dt, 1.0)
            hop = PlanarMotion(
                self.last_motion.x * share,
                self.last_motion.y * share,
                self.last_motion.yaw * share,
                self.last_motion.inliers,
                1.0,
            )
            predicted = predict_ground_pixels(runtime.track_pixels, runtime.model, hop)
            if predicted is not None:
                guess = predicted.astype(np.float32).reshape(-1, 1, 2)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        target, status, error = cv2.calcOpticalFlowPyrLK(
            previous.gray,
            current.gray,
            source,
            guess,
            winSize=window,
            maxLevel=self.lk_levels,
            criteria=criteria,
            flags=cv2.OPTFLOW_USE_INITIAL_FLOW if guess is not None else 0,
        )
        if target is None:
            return None
        if self.check_backward and self.frames_since_solve % self.backward_stride == 0:
            back, back_status, _ = cv2.calcOpticalFlowPyrLK(
                current.gray, previous.gray, target, None,
                winSize=window, maxLevel=self.lk_levels, criteria=criteria,
            )
            if back is None:
                return None
        else:
            back = source
            back_status = status
        source_flat = source.reshape(-1, 2)
        target_flat = target.reshape(-1, 2)
        back_flat = back.reshape(-1, 2)
        height, width = current.gray.shape
        keep = (
            status.reshape(-1).astype(bool)
            & back_status.reshape(-1).astype(bool)
            & np.isfinite(target_flat).all(axis=1)
            & (error.reshape(-1) <= self.lk_error_threshold)
            & (
                np.linalg.norm(back_flat - source_flat, axis=1)
                <= self.lk_forward_backward_threshold
            )
            & (target_flat[:, 0] >= 0.0)
            & (target_flat[:, 0] < width)
            & (target_flat[:, 1] >= 0.0)
            & (target_flat[:, 1] < height)
        )
        return source_flat[keep], target_flat[keep], runtime.track_ids[keep]


    def _solve_pairs(self, runtime: CameraRuntime):
        """Pixels at the last solve and now, for features that survived both.

        Held as two sorted arrays rather than a dict of id to pixel. The dict
        cost a Python level lookup and a Python level append per feature, twice
        over, and with 700 features on each of two cameras at 60 Hz that was
        4.2 ms a pair -- more than the ground projection and the anchor solve
        together, for what is only bookkeeping.
        """
        if runtime.track_ids is None or runtime.solve_ids is None:
            return None
        held_ids = runtime.solve_ids
        if len(held_ids) == 0:
            return None
        where = np.searchsorted(held_ids, runtime.track_ids)
        np.clip(where, 0, len(held_ids) - 1, out=where)
        matched = held_ids[where] == runtime.track_ids
        if int(np.count_nonzero(matched)) < self.ground_min_inliers:
            return None
        return (
            runtime.solve_points[where[matched]],
            runtime.track_pixels[matched],
            runtime.track_ids[matched],
        )

    def _remember_solve_pixels(self, runtime: CameraRuntime) -> None:
        if runtime.track_ids is None or runtime.track_pixels is None:
            runtime.solve_ids = np.empty(0, dtype=np.int64)
            runtime.solve_points = np.empty((0, 2), dtype=np.float64)
        else:
            # Sorted once here so that every lookup afterwards is a binary
            # search over an array rather than a walk over a dict.
            order = np.argsort(runtime.track_ids, kind='stable')
            runtime.solve_ids = np.ascontiguousarray(runtime.track_ids[order])
            runtime.solve_points = np.ascontiguousarray(
                runtime.track_pixels[order], dtype=np.float64)
        runtime.solve_frame = runtime.latest
        runtime.hop_disparity = 0.0

    def _track_camera(
        self, runtime: CameraRuntime, yaw_delta: Optional[float]
    ) -> Optional[TrackResult]:
        # With the front end supplying tracks the hop has already been made
        # and there is no image here to make it from.
        if not self.track_prefix and not self._advance_tracks(runtime):
            self._remember_solve_pixels(runtime)
            return None
        if self.track_prefix and runtime.anchors is None:
            runtime.anchors = GroundAnchorMap(
                max_anchors=self.max_anchors,
                max_age_frames=self.anchor_max_age,
                update_gain=self.anchor_update_gain,
                max_observations=self.anchor_max_observations,
                select_by_consistency=self.anchor_selection,
            )
        _t = time.perf_counter()
        pairs = self._solve_pairs(runtime)
        self.stage['pair'] += time.perf_counter() - _t
        if pairs is None:
            self._remember_solve_pixels(runtime)
            return None
        previous_pixels, current_pixels, track_ids = pairs

        _t = time.perf_counter()
        tilt = self._body_tilt()
        band = self._ground_band(runtime)
        previous_ground, valid_previous = pixels_to_ground(
            previous_pixels,
            runtime.model,
            band,
            runtime.ground_min_distance,
            tilt,
            runtime.range_scale,
        )
        current_ground, valid_current = pixels_to_ground(
            current_pixels,
            runtime.model,
            band,
            runtime.ground_min_distance,
            tilt,
            runtime.range_scale,
        )
        ground_valid = valid_previous & valid_current
        self.stage['ground'] += time.perf_counter() - _t

        motion = None
        inlier_mask = np.zeros(len(previous_pixels), dtype=bool)
        from_map = False
        spread = 0.0
        if yaw_delta is not None and np.any(ground_valid):
            _t = time.perf_counter()
            motion, inlier_mask, from_map, spread = self._solve_motion(
                runtime,
                track_ids,
                current_ground,
                previous_ground,
                ground_valid,
                yaw_delta,
            )
            self.stage['solve'] += time.perf_counter() - _t
        _rest = time.perf_counter()

        # The reference advances whether or not the caller ends up using this
        # solve. Holding it back on a rejected one to recover that interval's
        # travel was tried and is much worse: the interval then grows, fewer
        # identities survive to be paired, and the next solve fails for the
        # same reason. Three online runs went from 4.42/1.38/0.12 m to
        # 12.5/6.5/8.6 m.
        self._remember_solve_pixels(runtime)
        self.stage['rest'] += time.perf_counter() - _rest
        return TrackResult(
            previous_pixels,
            current_pixels,
            previous_ground,
            current_ground,
            ground_valid,
            inlier_mask,
            motion,
            track_ids,
            from_map,
            spread,
        )

    def _solve_motion(
        self, runtime, track_ids, current_ground, previous_ground, ground_valid, yaw_delta
    ):
        """Prefer the accumulated map; fall back to the previous frame alone.

        Matching against anchors averaged over a feature's whole life is what
        stops a burst of bad matches from carrying the estimate, which is how
        the two-frame solve failed above walking pace.
        """
        _s = time.perf_counter()
        usable = np.flatnonzero(ground_valid)
        ids = track_ids[usable]
        anchored = runtime.anchors.anchored(ids)
        self.stage['lookup'] += time.perf_counter() - _s
        if np.count_nonzero(anchored) >= self.ground_min_inliers:
            selected = usable[anchored]
            _s = time.perf_counter()
            world, weights = runtime.anchors.anchor_view(track_ids[selected])
            self.stage['lookup'] += time.perf_counter() - _s
            _s = time.perf_counter()
            aligned = align_to_anchors(
                current_ground[selected],
                world,
                weights,
                wrap_pi(self.pose.yaw + yaw_delta),
                self.ground_ransac_threshold,
                self.ground_min_inliers,
                self.heading.enabled,
            )
            self.stage['align'] += time.perf_counter() - _s
            if aligned is not None:
                translation, inliers, spread, solved_yaw, yaw_sigma = aligned
                handed = wrap_pi(self.pose.yaw + yaw_delta)
                # Recorded, not applied. Both cameras see the same heading and
                # each has its own opinion of how far it is out; folding them
                # in one at a time gives the filter two updates against one
                # prediction and lets whichever solved last decide the pose.
                # Measured, that learned 0.0095 rad/s of bias on a drive that
                # had none.
                if (self.heading.enabled and math.isfinite(yaw_sigma)
                        and yaw_sigma > 0.0):
                    self.heading_observations.append(
                        (wrap_pi(solved_yaw - handed), yaw_sigma))
                pose = Pose2(float(translation[0]), float(translation[1]), handed)
                motion = relative_motion(self.pose, pose)
                motion = PlanarMotion(
                    motion.x, motion.y, motion.yaw, int(np.count_nonzero(inliers)), 1.0
                )
                mask = np.zeros(len(track_ids), dtype=bool)
                mask[selected[inliers]] = True
                return motion, mask, True, spread

        estimate = estimate_planar_motion_with_yaw(
            previous_ground[ground_valid],
            current_ground[ground_valid],
            yaw_delta,
            self.ground_ransac_threshold,
            self.ground_min_inliers,
        )
        mask = np.zeros(len(track_ids), dtype=bool)
        if estimate is None:
            return None, mask, False, 0.0
        mask[usable] = estimate[1]
        # Spread of the pairwise solution, on the same footing as the map one.
        motion_estimate = estimate[0]
        offsets = previous_ground[ground_valid] - (
            np.array(
                [
                    [math.cos(yaw_delta), -math.sin(yaw_delta)],
                    [math.sin(yaw_delta), math.cos(yaw_delta)],
                ]
            )
            @ current_ground[ground_valid].T
        ).T
        chosen = estimate[1]
        residual = np.linalg.norm(
            offsets[chosen] - np.array([motion_estimate.x, motion_estimate.y]), axis=1
        )
        spread = float(math.sqrt(float(np.mean(residual ** 2)))) if len(residual) else 0.0
        return motion_estimate, mask, False, spread

    def _ready_to_solve(self) -> bool:
        """Whether the views have separated enough to be worth comparing.

        A fixed frame count cannot do this. What a solve needs is ground that
        has visibly moved, and how many frames that takes depends on speed,
        on the rate the camera actually delivers, and on how close the ground
        in view is. Sixteen frames is half a metre at parking pace and three
        metres at road pace, and at three metres optical flow has nothing left
        to match. Counting pixels instead lets the same setting hold across
        both.
        """
        travelled = max(
            (runtime.hop_disparity for runtime in self.cameras.values()), default=0.0
        )
        if self.frames_since_solve < self.solve_min_frames:
            return False
        return (
            travelled >= self.solve_trigger_disparity
            or self.frames_since_solve >= self.solve_max_frames
        )

    def _disparity(self, tracks) -> Optional[float]:  # noqa: ARG002
        """How far the most-moved ground points shifted, in pixels.

        The mean is useless here. Ground features sit anywhere from half a
        metre to twelve away, and pixel shift falls with the square of that
        distance, so the far ones barely move even at speed and drag the
        average under any sensible threshold. Driven slowly in a straight
        line, a mean-based test called 88 % of frames stationary and the
        estimate covered a tenth of the ground actually travelled. The points
        nearest the car are the ones that cannot stay still if it is moving.
        """
        # Cameras that have a frame, not cameras that have movement. The
        # `> 0.0` this used to filter on excluded exactly the case the
        # zero-velocity check exists to find: a vehicle that is perfectly
        # still shifts no pixels, so every camera reported 0.0, the list came
        # back empty, and `standing` was never true. ZUPT fired zero times in
        # every run ever recorded with this build.
        seen = [
            runtime.hop_disparity
            for runtime in self.cameras.values()
            if runtime.latest is not None
        ]
        return max(seen) if seen else None

    def _is_stationary(self, start: float, end: float, speed: float) -> bool:
        """Whether the accelerometer says nothing happened over this interval.

        Gravity is the only specific force on a vehicle at rest, so the
        magnitude sits at g and the rates sit at zero. Anything driving,
        however gently, breaks one of those.
        """
        if not self.use_zupt or speed > self.zupt_speed:
            return False
        window = [s for s in self.imu_motion_samples if start <= s[0] <= end]
        if len(window) < 2:
            return False
        # A fraction, not every sample. CARLA injects contact impulses -- the
        # 99th percentile of a standing vehicle's specific force is 11.5 m/s2
        # against a median of exactly g -- and `all()` hands any one of them a
        # veto over the whole window. Requiring most of the window to be still
        # is the same physical claim without the single-sample tyranny; the
        # median rather than the mean carries the gravity estimate for the
        # same reason.
        forces = sorted(s[2] for s in window)
        gravity = forces[len(forces) // 2]
        rates = sorted(abs(s[1]) for s in window)
        if (self.zupt_dropout > 0.0 and gravity <= self.zupt_dropout
                and rates[len(rates) // 2] <= self.zupt_gyro):
            return True
        quiet = sum(
            1 for _, rate, force in window
            if rate <= self.zupt_gyro and abs(force - gravity) <= self.zupt_accel
        )
        return (quiet >= self.zupt_quiet_fraction * len(window)
                and abs(gravity - 9.81) <= 0.5)

    def _process_pair(self, runtimes: List[CameraRuntime]):
        process_start = time.perf_counter()
        if self.use_camera_info:
            starved = [r.name for r in runtimes if not self.camera_infos[r.name]]
            if starved:
                # Three separate runs have been ruined by this and none of
                # them looked wrong: the solve falls back on the parameter K,
                # the scale stays plausible, and the trajectory is quietly
                # somebody else's. QoS mismatches are the usual cause.
                self.get_logger().error(
                    f'No CameraInfo for {", ".join(starved)}; solving on the '
                    'configured intrinsics. The result is not the sensor\'s.',
                    throttle_duration_sec=5.0)
        # The frames are aligned to within the sync tolerance, so their mean
        # is the time the set describes -- with two cameras, the midpoint the
        # pair used to take.
        current_stamp = sum(r.latest.stamp for r in runtimes) / len(runtimes)
        if any(r.solve_frame is None for r in runtimes):
            for runtime in runtimes:
                runtime.previous = runtime.latest
                # Seed the set here, or the first solve has nothing from the
                # past to compare the present against. With the front end
                # supplying tracks they are already seeded and there is no
                # image to seed them from.
                if not self.track_prefix:
                    self._detect_new_tracks(runtime, runtime.latest.gray)
                self._remember_solve_pixels(runtime)
            self.last_stamp_msg = runtimes[0].latest.stamp_msg
            self._publish_state(runtimes[0].latest.stamp_msg, np.zeros(3))
            return

        # The interval a solve spans, not the gap to the frame before it:
        # tracking now runs every frame while the solve stands further back.
        previous_stamp = (
            sum(r.solve_frame.stamp for r in runtimes) / len(runtimes))
        dt = current_stamp - previous_stamp
        yaw_delta = None
        imu_available = not self.use_imu_yaw
        if self.use_imu_yaw:
            previous_imu_yaw = self._imu_yaw_at(previous_stamp)
            current_imu_yaw = self._imu_yaw_at(current_stamp)
            if previous_imu_yaw is not None and current_imu_yaw is not None:
                yaw_delta = math.remainder(
                    current_imu_yaw - previous_imu_yaw, 2.0 * math.pi
                )
                imu_available = True
            else:
                self.imu_yaw_misses += 1
                self.get_logger().warning(
                    'IMU unavailable at camera timestamps; holding odometry instead '
                    'of falling back to unconstrained two-frame vision '
                    f'(nearest sample {self._imu_gap(previous_stamp):.3f}s from the '
                    f'previous frame, {self._imu_gap(current_stamp):.3f}s from the '
                    f'current one, buffer spans {self._imu_span():.2f}s)',
                    throttle_duration_sec=2.0,
                )
        # Per pair, not per camera. Both cameras run this on their own thread
        # and both used to reset these, so whichever finished first decided
        # what the other one saw: the second camera read a counter of zero and
        # recorded an interval of one frame. Two runs of the same recording
        # then produced trajectories differing by 8 cm over 34 m, with every
        # other count in the report identical, which is how it was found.
        self.frames_in_last_solve = max(self.frames_since_solve, 1)
        self.frames_since_solve = 0

        # Before the cameras speak, not after: they are what fills this.
        self.heading.predict(max(dt, 0.0))
        self.heading_observations = []

        if imu_available:
            done = list(
                self.trackers.map(
                    lambda runtime: self._track_camera(runtime, yaw_delta),
                    runtimes,
                )
            )
            tracks = {r.name: solved for r, solved in zip(runtimes, done)}
        else:
            tracks = {r.name: None for r in runtimes}
        per_camera = {
            name: track.motion
            for name, track in tracks.items()
            if track is not None and track.motion is not None
        }
        # Each camera reports where it is at its own frame's time, but the pair
        # is stamped at the midpoint of the two and dt is measured between
        # midpoints. While both cameras run at the same rate those are the same
        # instant and this changes nothing. When one starts dropping frames
        # they are not: the sync tolerance allows 20 ms, which at 8 m/s is
        # 16 cm, more than the 13 cm the pair is trying to measure in the first
        # place. Averaging two positions taken that far apart is what the
        # front/rear disagreement had been rising to report.
        #
        # So put both on the interval the pair is stamped over before they are
        # compared or fused. Over a single frame the velocity is constant to
        # far better than the registration noise, so the straight ratio is
        # enough; the bounds are there to leave a wild interval alone rather
        # than to correct it.
        for runtime in runtimes:
            name = runtime.name
            measured = per_camera.get(name)
            if measured is None or runtime.solve_frame is None:
                continue
            own = runtime.latest.stamp - runtime.solve_frame.stamp
            if own <= 1e-4:
                continue
            ratio = dt / own
            if not 0.5 <= ratio <= 2.0:
                continue
            # Along the arc the vehicle is on, not along a stretched chord.
            # Multiplying the translation and leaving the heading alone is the
            # straight line answer and is only right with the wheel centred.
            per_camera[name] = rescale_motion(measured, ratio)
        motions = list(per_camera.values())
        precision_inputs = [
            (per_camera[name].x, per_camera[name].y,
             per_camera[name].inliers, track.spread)
            for name, track in tracks.items()
            if track is not None and name in per_camera and track.spread > 0.0
        ]
        # Two cameras looking at different ground disagreeing is the one signal
        # neither can produce alone.
        disagreement = 0.0
        if len(per_camera) >= 2:
            # How far apart the cameras' answers land, at their widest. With
            # two that is the distance between them, which is what the noise
            # model was measured against; with more it is still the honest
            # statement of how much they fail to agree.
            solved = list(per_camera.values())
            disagreement = max(
                math.hypot(a.x - b.x, a.y - b.y)
                for i, a in enumerate(solved) for b in solved[i + 1:]
            )
        self.camera_disagreement = disagreement
        if any(track.anchored_from_map for track in tracks.values() if track):
            self.map_aligned_frames += 1
            self.map_ready = True
        motion = fuse_planar_motions(motions)
        if self.fuse_by_spread and len(precision_inputs) == len(motions) and motions:
            combined = fuse_by_precision(precision_inputs)
            if combined is not None and motion is not None:
                motion = PlanarMotion(
                    combined[0], combined[1], motion.yaw, combined[2], motion.scale
                )
        vision_speed = (
            math.hypot(motion.x, motion.y) / dt
            if motion is not None and dt > 1e-4
            else float('inf')
        )
        disparity = self._disparity(tracks)
        self.last_disparity = disparity
        # Decided before the fusion, not after it. Standing still used to be
        # noticed only once the vision displacement had already gone into the
        # filter, and those are the hops it most needed protecting from: the
        # eight hops of this run where the vehicle did not move carry 76% of
        # the whole run's squared vision error, one of them reporting 104 mm
        # of travel in 67 ms.
        standing = (
            self.use_zupt
            and disparity is not None
            and disparity <= self.zupt_disparity
            and self._is_stationary(previous_stamp, current_stamp, vision_speed)
        )
        if standing:
            self.zupt_holds += 1

        previous_pose = self.pose

        # Accelerometer and vision each hold part of the answer, so neither
        # decides alone: the filter propagates on the IMU and takes vision as a
        # measurement weighted by how many ground points backed it.
        if (self.displacement_filter is not None and motion is not None
                and dt > 1e-4):
            # Propagate over exactly the interval this measurement spans,
            # boundaries included. Each buffered sample carries the step since
            # the one before it, so taking whole samples covers a window
            # shifted early by up to one IMU period -- 17% of a 100 ms hop at
            # 60 Hz. The first step is clipped to where the interval actually
            # begins and a zero-order-hold tail closes it at the end.
            self.displacement_filter.open_hop()
            window = [s for s in self.imu_window
                      if previous_stamp < s[0] <= current_stamp]
            for index, sample in enumerate(window):
                step = min(sample[3], 0.1)
                if index == 0:
                    step = min(step, sample[0] - previous_stamp)
                if step > 0.0:
                    self.displacement_filter.predict(
                        np.array(sample[1:3]), step, sample[4])
            if window:
                tail = current_stamp - window[-1][0]
                if 0.0 < tail <= 0.1:
                    self.displacement_filter.predict(
                        np.array(window[-1][1:3]), tail, window[-1][4])
            c, s = math.cos(previous_pose.yaw), math.sin(previous_pose.yaw)
            world = np.array([c * motion.x - s * motion.y,
                              s * motion.x + c * motion.y])
            # The disagreement between the two cameras is already a distance,
            # so here it needs no division to become a variance.
            extra = ((self.disagreement_weight * disagreement) ** 2
                     if len(per_camera) >= 2 else self.single_camera_variance * dt * dt)
            # Inlier-weighted spread across whichever cameras reported one.
            spreads = [(s, n) for _, _, n, s in precision_inputs if s > 0.0]
            spread = (sum(s * n for s, n in spreads) / sum(n for _, n in spreads)
                      if spreads else 0.0)
            if standing:
                # The measurement is that nothing moved, and it is a better
                # measurement than the solve: truth says these hops travelled
                # 0.0 mm while vision reported up to 104.
                world = np.zeros(2)
                extra = (self.zupt_velocity_sigma * dt) ** 2
                spread = 0.0
            if not self.displacement_filter.update(
                    world, motion.inliers, extra, spread):
                self.filter_rejections += 1
            if standing:
                self.displacement_filter.update_zero_velocity(
                    self.zupt_velocity_sigma)
            self._log_filter_update(previous_stamp, current_stamp, spread, extra,
                                    self._disparity(tracks), motion.inliers)
            self.inertial.correct_velocity(self.displacement_filter.velocity)
            fused = self.displacement_filter.body_translation(previous_pose.yaw)
            if math.hypot(fused[0], fused[1]) <= self.max_translation:
                motion = PlanarMotion(
                    float(fused[0]), float(fused[1]), motion.yaw, motion.inliers, 1.0
                )
        elif self.use_inertial and motion is not None and dt > 1e-4:
            measured = world_velocity_from_motion(
                motion.x, motion.y, dt, previous_pose.yaw
            )
            if measured is not None:
                # Disagreement is a translation over dt, so as a velocity
                # variance it is (d/dt)^2.
                extra = (
                    (self.disagreement_weight * disagreement / dt) ** 2
                    if len(per_camera) >= 2
                    else self.single_camera_variance
                )
                if not self.velocity_filter.update(
                    np.array(measured), motion.inliers, extra
                ):
                    self.filter_rejections += 1
                # The propagator withholds spawn/drop acceleration until a
                # vision velocity has fixed its integration constant. Keep its
                # anchor on the same fused state that drives the published
                # motion; otherwise it remains uncorrected forever and returns
                # zero acceleration to this filter on every IMU sample.
                self.inertial.correct_velocity(self.velocity_filter.velocity)
            fused = self.velocity_filter.body_translation(dt, previous_pose.yaw)
            if math.hypot(fused[0], fused[1]) <= self.max_translation:
                motion = PlanarMotion(
                    float(fused[0]), float(fused[1]), motion.yaw, motion.inliers, 1.0
                )

        if motion is not None and standing:
            # Hold the position, keep the heading. A parking manoeuvre stops
            # several times, so every stop is a chance to stop accumulating.
            motion = PlanarMotion(0.0, 0.0, motion.yaw, motion.inliers, 1.0)

        warming_up = not self.map_ready
        # The cap is on how far the vehicle can plausibly have moved, so it
        # has to cover the whole interval the update is closing, not one
        # frame of it. Against the map -- which returns an absolute position,
        # not a hop -- a rejected frame leaves the pose behind, and the next
        # solve reports the arrears along with the new motion. Capping that at
        # a single frame rejects the correction too, and the arrears never get
        # paid: 49 of 51 rejections at 60 Hz were this, and the trajectory
        # finished 25-35% short.
        allowance = self.max_translation
        if dt > 1e-4 and self.last_accept_stamp is not None:
            since = current_stamp - self.last_accept_stamp
            allowance *= max(1.0, min(since / dt, self.max_arrears_frames))
        if (
            motion is None
            or warming_up
            or dt <= 1e-4
            or math.hypot(motion.x, motion.y) > allowance
            or abs(motion.yaw) > self.max_yaw
        ):
            self.motion_failures += 1
            if motion is None:
                self.fail_no_solve += 1
            elif math.hypot(motion.x, motion.y) > allowance:
                self.fail_translation += 1
                self.worst_rejected = max(
                    self.worst_rejected, math.hypot(motion.x, motion.y))
                self.worst_allowance = max(self.worst_allowance, allowance)
            elif abs(motion.yaw) > self.max_yaw:
                self.fail_yaw += 1
            twist = np.zeros(3, dtype=np.float64)
            # Rejecting the vision translation is no reason to throw away the
            # IMU rotation over the same interval. Dropping it used to leak
            # heading on every rejected pair and never came back.
            #
            # Nor is it a reason to stand still. Holding the position on a
            # rejected pair is a claim that the vehicle did not move, which at
            # road speed is the one thing that is certainly false, and the
            # arrears are never made up afterwards. Tightening the gate so
            # fewer bad values get through makes it worse, not better -- 0.4 m
            # rejected 94-105 pairs and scored 5.4-5.5 m against 3.5-4.2 at
            # 1.0 m -- because what the trajectory loses is the motion, not the
            # error. So carry the filtered velocity across the gap instead:
            # dead reckoning for one frame is wrong by centimetres, standing
            # still is wrong by the whole hop.
            if yaw_delta is not None and abs(yaw_delta) <= self.max_yaw:
                carried = None
                # Not while the map is still being built. There the pose is
                # deliberately pinned at the origin, because the anchors are
                # written from it and a pose that has wandered writes a map
                # that is wrong everywhere. Coasting through that window moved
                # the vehicle 0.12 m before it had a map to be wrong about.
                if dt > 1e-4 and self.coast_on_reject and not warming_up:
                    predicted = self.velocity_filter.body_translation(dt, self.pose.yaw)
                    if math.hypot(predicted[0], predicted[1]) <= allowance:
                        # The IMU already says how fast the heading is turning
                        # over this same gap, so the dead reckoning can follow
                        # the arc rather than shoot off along the tangent.
                        carried = motion_from_twist(
                            float(predicted[0]) / dt, float(predicted[1]) / dt,
                            yaw_delta / dt, dt)
                        self.coasted += 1
                if carried is None:
                    carried = PlanarMotion(0.0, 0.0, yaw_delta, 0, 1.0)
                else:
                    carried = PlanarMotion(
                        carried.x, carried.y, yaw_delta, 0, 1.0)
                self.pose = self.pose.compose(carried)
                self.yaw_only_updates += 1
            if warming_up:
                # The map has to come from somewhere, and while it is missing
                # the pose is held at the origin, which for a vehicle that has
                # not moved is the right place to write these from. A genuine
                # motion failure gets no such treatment: there the pose is not
                # to be trusted and neither would the anchors be.
                self._update_anchors(tracks)
        else:
            self.pose = self.pose.compose(motion)
            # One update against one prediction, from both cameras at once,
            # weighted the way two measurements of the same quantity are.
            # Applied to the heading only: the translations were solved under
            # the heading as handed, and the offset is small enough that what
            # it does to them is second order.
            if self.heading.enabled and self.heading_observations:
                precision = sum(1.0 / (s * s) for _, s in self.heading_observations)
                if precision > 0.0:
                    residual = sum(
                        r / (s * s) for r, s in self.heading_observations) / precision
                    offset = self.heading.update(
                        residual, math.sqrt(1.0 / precision))
                    self.yaw_correction = offset
                    self.yaw_gain_sum += abs(offset)
                    self.yaw_gain_count += 1
                    self.pose = Pose2(
                        self.pose.x, self.pose.y, wrap_pi(self.pose.yaw + offset))
            self.last_accept_stamp = current_stamp
            self.last_motion = motion
            self.last_solve_dt = dt
            c = math.cos(motion.yaw)
            s = math.sin(motion.yaw)
            body_x = (c * motion.x + s * motion.y) / dt
            body_y = (-s * motion.x + c * motion.y) / dt
            raw_twist = np.array([body_x, body_y, motion.yaw / dt])
            alpha = 1.0 if self.twist_tau == 0.0 else dt / (self.twist_tau + dt)
            self.filtered_twist += alpha * (raw_twist - self.filtered_twist)
            twist = self.filtered_twist.copy()
            _t = time.perf_counter()
            self._update_anchors(tracks)
            self.stage['anchor'] += time.perf_counter() - _t
            if (
                self.last_mapping_stamp is None
                or current_stamp - self.last_mapping_stamp >= self.mapping_min_period
            ):
                self.last_mapping_stamp = current_stamp
                _t = time.perf_counter()
                self._integrate_tracks(tracks, previous_pose, motion)
                self.stage['map'] += time.perf_counter() - _t

        self.frames_processed += 1
        self.last_stamp_msg = runtimes[0].latest.stamp_msg
        self._publish_state(runtimes[0].latest.stamp_msg, twist)
        self._publish_points(runtimes[0].latest.stamp_msg)
        self.process_seconds += time.perf_counter() - process_start
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self.last_status_time >= 2.0:
            span = now - self.arrival_since if self.arrival_since else 0.0
            rates = (
                ' '.join(
                    f'{source}_hz={count / span:.1f}'
                    for source, count in self.arrivals.items()
                )
                if span > 1e-6
                else 'rates=pending'
            )
            self.arrival_since = now
            for source in self.arrivals:
                self.arrivals[source] = 0
            inliers = motion.inliers if motion is not None else 0
            anchors = sum(
                len(camera.anchors) if camera.anchors else 0
                for camera in self.cameras.values()
            )
            self.get_logger().info(
                f'vision odometry: frames={self.frames_processed} '
                f'ingest={1000.0 * self.ingest_seconds / max(self.ingest_count, 1):.1f}ms '
                f'[{" ".join(f"{k}={1000.0 * v / max(self.frames_processed, 1):.1f}" for k, v in self.stage.items())}] '
                f'solve={1000.0 * self.process_seconds / max(self.frames_processed, 1):.1f}ms '
                f'zupt={self.zupt_holds} '
                f'disp={self.last_disparity if self.last_disparity is not None else -1:.2f} '
                f'pairs={self.pairs_seen} '
                f'evicted={self.frames_evicted} '
                f'rx={"/".join(str(self.tracks_received[n]) for n in self.camera_names)} '
                f'info={"/".join(str(self.camera_infos[n]) for n in self.camera_names)} '
                f'raw={"/".join(str(self.tracks_raw[n]) for n in self.camera_names)} '
                f'{rates} '
                f'anchors={anchors} map_frames={self.map_aligned_frames} '
                f'failures={self.motion_failures} '
                f'(nosolve={self.fail_no_solve} '
                f'trans={self.fail_translation} '
                f'yaw={self.fail_yaw} '
                f'worst={self.worst_rejected:.2f}m/{self.worst_allowance:.2f}m) '
                f'filter_rej={self.filter_rejections} '
                f'v_filt={self.velocity_filter.speed():.2f} '
                f'disagree={self.camera_disagreement:.3f} '
                f'{" ".join(f"{n}={m.inliers}" for n, m in per_camera.items())} '
                f'yaw_only={self.yaw_only_updates} '
                f'ycorr={self.yaw_gain_sum / max(self.yaw_gain_count, 1):.2e} '
                f'ybias={self.heading.rate:+.2e} '
                f'coasted={self.coasted} '
                f'imu_misses={self.imu_yaw_misses} inliers={inliers} '
                f'pose=({self.pose.x:.2f}, {self.pose.y:.2f}, '
                f'{self.pose.yaw:.2f}) v={twist[0]:.2f}'
            )
            self.last_status_time = now

    def _update_anchors(self, tracks):
        """Fold this frame's ground observations into each feature's anchor.

        Done after the pose is accepted, so anchors are written in the world
        frame the estimate just settled on.
        """
        allow_new = True
        if self.anchor_seed_travel > 0.0:
            here = np.array([self.pose.x, self.pose.y], dtype=np.float64)
            if self.last_seed_position is None:
                self.last_seed_position = here
            elif float(np.linalg.norm(here - self.last_seed_position)) >= (
                self.anchor_seed_travel
            ):
                self.last_seed_position = here
            else:
                allow_new = False
        for name, track in tracks.items():
            if track is None or track.track_ids is None:
                continue
            runtime = self.cameras[name]
            if runtime.anchors is None:
                continue
            usable = np.flatnonzero(track.ground_valid)
            if len(usable) == 0:
                continue
            points = np.column_stack(
                (track.current_ground[usable], np.zeros(len(usable)))
            )
            world = transform_points_to_world(points, self.pose)
            # How much this sighting is worth. A ground point's position error
            # grows with its range -- the same angular error is worth eight
            # times as many centimetres at eight metres as at one -- so the
            # precision goes as the inverse square of it. The offset is the
            # camera, not the vehicle: range is measured from the lens that
            # saw the point.
            offset = track.current_ground[usable] - \
                runtime.model.translation_base_from_camera[None, :2]
            ranges = np.hypot(offset[:, 0], offset[:, 1])
            runtime.anchors.update(
                track.track_ids[usable], world[:, :2], allow_new,
                1.0 / np.maximum(ranges, 0.1) ** 2)
            # Describe from the frame the solve just used. Doing it at
            # detection time meant the batch was overwritten by the fifteen
            # frames that pass before the next solve, and almost no anchor
            # ever got one.
            if self.reassociate and runtime.latest is not None:
                ids = track.track_ids[usable]
                wanted = runtime.anchors.undescribed(ids)
                if np.any(wanted):
                    pixels = track.current_pixels[usable][wanted]
                    described, valid = self._describe(runtime.latest.gray, pixels)
                    if described is not None:
                        runtime.anchors.describe(ids[wanted][valid], described[valid])

    def _collect_points(self, points, origin, label: float) -> None:
        """Hold on to what was seen, to go out as one cloud for the hop."""
        # The ground registration works in the plane and the obstacle solve
        # in three dimensions, so both shapes turn up here.
        points = np.atleast_2d(np.asarray(points, dtype=np.float64))
        if not len(points):
            return
        rows = np.zeros((len(points), 6), dtype=np.float32)
        rows[:, 0:points.shape[1]] = points[:, :3]
        rows[:, 3] = label
        rows[:, 4] = origin[0]
        rows[:, 5] = origin[1]
        self.pending_points.append(rows)

    def _publish_points(self, stamp_msg) -> None:
        if not self.pending_points:
            return
        rows = np.concatenate(self.pending_points)
        self.pending_points = []
        message = PointCloud2()
        message.header.stamp = stamp_msg
        message.header.frame_id = self.map_frame
        message.height = 1
        message.width = len(rows)
        message.fields = [
            PointField(name=name, offset=4 * index,
                       datatype=PointField.FLOAT32, count=1)
            for index, name in enumerate(
                ('x', 'y', 'z', 'label', 'origin_x', 'origin_y'))
        ]
        message.is_bigendian = False
        message.point_step = 24
        message.row_step = 24 * len(rows)
        message.is_dense = True
        message.data = array.array('B', rows.tobytes())
        self.points_pub.publish(message)

    def _integrate_tracks(self, tracks, previous_pose: Pose2, motion: PlanarMotion):
        for name, track in tracks.items():
            if track is None:
                continue
            runtime = self.cameras[name]
            origin_base = runtime.model.translation_base_from_camera
            origin_world = transform_points_to_world(origin_base.reshape(1, 3), previous_pose)[0]

            free_points = np.column_stack(
                (
                    track.previous_ground[track.motion_inliers],
                    np.zeros(np.count_nonzero(track.motion_inliers)),
                )
            )
            free_world = transform_points_to_world(free_points, previous_pose)
            self._collect_points(free_world, origin_world, GROUND)

            if self.obstacle_slip_baseline > 0.0:
                self._integrate_obstacle_slip(runtime, track)
            else:
                self._integrate_keyframe_obstacles(runtime)

    def _integrate_obstacle_slip(self, runtime: CameraRuntime, track):
        """Read each feature's height off how far its ground projection slides.

        A feature really on the road projects to a fixed world point: the ray
        from wherever the camera has moved to meets the plane in the same
        place. One standing h above the road does not. With the camera at
        height H its ray crosses the plane H/(H-h) times as far out, so moving
        the camera by D slides that crossing by -D*h/(H-h), against the travel:

            r = -(slip . D) / |D|^2,    h = H * r / (1 + r)

        and the feature sits on the ray at (H-h)/H of the way to the crossing.
        Nothing is triangulated; this is the residual the ground registration
        already computes and discards as an outlier.

        What makes it work is the length of D. The projection is taken against
        an assumed pitch, and that assumption wobbles: a fifth of a degree
        slides a crossing ten metres out by 0.37 m. Over a 0.35 m baseline a
        0.15 m obstacle only slides 68 mm, so the wobble buries it -- measured,
        that gave precision 0.455 at recall 0.016. The slip grows with the
        travel and the wobble does not, so each feature keeps the camera pose
        and projection from when it was first seen and is only read out once
        it has earned a baseline, however many frames that takes.

        The median is removed first because a pitch error is common to every
        feature in the frame while a height is not, and most features are on
        the road.
        """
        if track.track_ids is None:
            return
        usable = track.ground_valid
        if not np.any(usable):
            return
        ids = track.track_ids[usable]
        origin_base = runtime.model.translation_base_from_camera
        height = float(origin_base[2])
        ground = np.column_stack(
            (track.current_ground[usable], np.zeros(np.count_nonzero(usable)))
        )
        world = transform_points_to_world(ground, self.pose)[:, :2]
        camera = transform_points_to_world(origin_base.reshape(1, 3), self.pose)[0, :2]

        order = np.argsort(ids, kind='stable')
        ids = ids[order]
        world = world[order]

        started = np.repeat(camera[None, :], len(ids), 0)
        held = runtime.slip_origins
        if held is None or len(held[0]) == 0:
            runtime.slip_origins = (ids, world, started, np.zeros(len(ids), np.int32))
            return
        held_ids, held_world, held_camera, misses = held

        where = np.searchsorted(held_ids, ids)
        np.clip(where, 0, len(held_ids) - 1, out=where)
        seen = held_ids[where] == ids

        origin_world = np.where(seen[:, None], held_world[where], world)
        origin_camera = np.where(seen[:, None], held_camera[where], started)

        travel = camera[None, :] - origin_camera
        distance = np.hypot(travel[:, 0], travel[:, 1])
        ready = seen & (distance >= self.obstacle_slip_baseline)

        # A feature whose ground projection is unusable on one step is not a
        # dead feature, and rebuilding this table from the current step alone
        # threw its history away: half of them then looked like they had only
        # ever travelled 0.19 m, which is the front end being blamed for the
        # bookkeeping. Entries persist until they are either read out or have
        # gone missing for long enough to be gone for good.
        # A feature that has just been read starts a fresh baseline where it
        # stands rather than being dropped: one that lives ten metres should
        # be heard from ten times over a one metre baseline, not once.
        held_world = held_world.copy()
        held_camera = held_camera.copy()
        held_world[where[ready]] = world[ready]
        held_camera[where[ready]] = camera[None, :]
        misses = misses + 1
        misses[where[seen]] = 0
        alive = misses <= self.obstacle_slip_patience
        fresh = ~seen
        runtime.slip_origins = _merge_slip_origins(
            (held_ids[alive], held_world[alive], held_camera[alive], misses[alive]),
            (ids[fresh], world[fresh], started[fresh],
             np.zeros(int(np.count_nonzero(fresh)), np.int32)),
        )
        if not np.any(ready):
            return

        # The common part of the slip -- the wobble in the assumed pitch --
        # has to be read off everything in view, not off the handful of
        # features that happen to have earned their baseline on this step.
        # Estimating it from those seven or so put noise into every height.
        measurable = seen & (distance > 0.1)
        common = 0.0
        if np.count_nonzero(measurable) >= 8:
            near = travel[measurable]
            reach = distance[measurable]
            common = float(np.median(
                -np.einsum('ij,ij->i', world[measurable] - origin_world[measurable],
                           near) / (reach * reach)))

        slip = world[ready] - origin_world[ready]
        step = travel[ready]
        span = distance[ready]
        ratio = -np.einsum('ij,ij->i', slip, step) / (span * span) - common
        with np.errstate(invalid='ignore', divide='ignore'):
            heights = height * ratio / (1.0 + ratio)
        obstacle = (
            np.isfinite(heights)
            & (ratio > 0.0)
            & (heights >= self.obstacle_min_height)
            & (heights <= min(self.obstacle_max_height,
                              height * self.obstacle_height_margin))
        )
        if not np.any(obstacle):
            return
        # The crossing overshoots the feature by the same factor that gave the
        # height, so walking back along the ray puts it where it stands.
        reach = ((height - heights[obstacle]) / height)[:, None]
        placed = camera[None, :] + (world[ready][obstacle] - camera[None, :]) * reach
        self._collect_points(placed, camera, OBSTACLE)

    def _integrate_keyframe_obstacles(self, runtime: CameraRuntime):
        """Triangulate the current view against a keyframe far enough behind it.

        Consecutive frames are centimetres apart at parking speed, which is far
        too short a baseline to place an obstacle. Holding a keyframe until the
        vehicle has actually moved gives depth worth mapping.
        """
        frame = runtime.latest
        if runtime.keyframe is None or runtime.keyframe.gray.shape != frame.gray.shape:
            runtime.keyframe = frame
            runtime.keyframe_pose = self.pose
            return

        # Triangulation wants the keyframe-to-current step expressed in the
        # keyframe frame, the same convention the frame to frame estimate uses.
        motion = relative_motion(runtime.keyframe_pose, self.pose)
        baseline = math.hypot(motion.x, motion.y)
        if baseline < self.mapping_baseline:
            return

        tracked = self._track_between(runtime.keyframe.gray, frame.gray)
        runtime.keyframe = frame
        keyframe_pose = runtime.keyframe_pose
        runtime.keyframe_pose = self.pose
        if tracked is None:
            return

        keyframe_pixels, current_pixels = tracked
        points_base, valid = triangulate_temporal_points(
            keyframe_pixels,
            current_pixels,
            runtime.model,
            motion,
            self.min_parallax,
            self.max_reprojection_error,
            self.triangulation_min_distance,
            self.triangulation_max_distance,
        )
        obstacle = (
            valid
            & (points_base[:, 2] >= self.obstacle_min_height)
            & (points_base[:, 2] <= self.obstacle_max_height)
        )
        if not np.any(obstacle):
            return
        self._collect_points(
            transform_points_to_world(points_base[obstacle], keyframe_pose),
            transform_points_to_world(
                runtime.model.translation_base_from_camera.reshape(1, 3),
                keyframe_pose)[0],
            OBSTACLE)


    def _publish_state(self, stamp, twist):
        # Nothing goes out while the map is still being built. The pose is
        # pinned to the origin over that stretch, and a vehicle already at
        # 8 m/s covers 0.375 m before the first map-anchored solve lands --
        # published as (0, 0) it becomes a constant offset that the estimate
        # then carries for the rest of the run, which measured as 0.243 m RMSE
        # against a per-step error of a few millimetres. Whoever is consuming
        # this is better served by no pose than by one that is knowingly
        # standing still while the car is not.
        if self.suppress_warmup_pose and not self.map_ready:
            return
        qx, qy, qz, qw = _quaternion_from_yaw(self.pose.yaw)
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.map_frame
        odom.child_frame_id = self.base_frame
        odom.pose.pose.position.x = self.pose.x
        odom.pose.pose.position.y = self.pose.y
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        odom.twist.twist.linear.x = float(twist[0])
        odom.twist.twist.linear.y = float(twist[1])
        odom.twist.twist.angular.z = float(twist[2])
        self.odom_pub.publish(odom)

        pose = PoseStamped()
        pose.header = odom.header
        pose.pose = odom.pose.pose
        self.pose_pub.publish(pose)

    def _publish_transform(self):
        if self.tf_broadcaster is None:
            return
        _, _, qz, qw = _quaternion_from_yaw(self.pose.yaw)
        transform = TransformStamped()
        # Stamp with the last processed image so the TF never leads the sensor
        # data when CARLA runs slower or faster than wall clock.
        transform.header.stamp = self.last_stamp_msg or self.get_clock().now().to_msg()
        transform.header.frame_id = self.map_frame
        transform.child_frame_id = self.base_frame
        transform.transform.translation.x = self.pose.x
        transform.transform.translation.y = self.pose.y
        transform.transform.rotation.z = qz
        transform.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(transform)

def main():
    rclpy.init()
    node = VisualOdometryNode()
    try:
        if node.executor_threads > 1:
            executor = MultiThreadedExecutor(num_threads=node.executor_threads)
            executor.add_node(node)
            executor.spin()
        else:
            rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
