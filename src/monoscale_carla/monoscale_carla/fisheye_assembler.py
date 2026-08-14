"""Publish one fisheye image per end from the two pinholes CARLA renders.

The vehicle carries a single 1.8 mm, 160 degree diagonal lens at each end.
CARLA cannot render one: every camera blueprint it has is a pinhole, and the
lens distortion attributes are a post-process over an already-rendered frame --
they redraw the same field into a circle rather than widening it.

So each lens is rendered as two square pinholes 70 degrees apart and assembled
here. Every output pixel is an equidistant ray, r = f*theta; the ray is rotated
into each source and sampled from whichever has it in frame. The remap tables
depend only on the geometry, so they are built once and reused.

What this buys is where the pixels go. A pinhole spends r = f*tan(theta) and
puts most of its resolution at the edge of a wide frame; at a 160 degree
diagonal its centre resolves 4.52 px per degree against an equidistant lens's
18.36. This stack reads the centre -- with the camera 0.89 m up and pitched 30
degrees down, ground from 0.6 m to 30 m all falls within 28 degrees of the
axis.

Subscribes:
    /sensing/camera/<end>/fisheye_a/image_raw
    /sensing/camera/<end>/fisheye_b/image_raw
Publishes:
    /sensing/camera/<end>/fisheye/image_raw
    /sensing/camera/<end>/fisheye/camera_info
"""
import array
import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image

try:
    import cv2
except ImportError:  # pragma: no cover - 런타임에만 필요하다
    cv2 = None


def carla_rotation(pitch_deg: float, yaw_deg: float,
                   roll_deg: float = 0.0) -> np.ndarray:
    """World-from-body, in UE4's order: roll, then pitch, then yaw."""
    cy, sy = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    cp, sp = math.cos(math.radians(pitch_deg)), math.sin(math.radians(pitch_deg))
    cr, sr = math.cos(math.radians(roll_deg)), math.sin(math.radians(roll_deg))
    return np.array([
        [cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr],
        [cp * sy, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr],
        [sp, -cp * sr, cp * cr],
    ])


# CARLA 축(x 전방, y 우, z 상) -> 광학 축(x 우, y 하, z 전방).
_OPTICAL = np.array([[0.0, 1.0, 0.0],
                     [0.0, 0.0, -1.0],
                     [1.0, 0.0, 0.0]])


def source_rotation(base_pitch: float, base_yaw: float,
                    yaw_offset: float) -> np.ndarray:
    """Assembled optical axes -> a source's optical axes, as `ray @ R`.

    Taking this for a rotation about the camera's own down axis is wrong when
    the rig is pitched: CARLA's yaw turns about the world vertical. With 30
    degrees of pitch that error put the seams tens of pixels out and dropped
    the overlap correlation to 0.16, where the correct rotation gives 0.95.
    """
    ref = carla_rotation(base_pitch, base_yaw)
    src = carla_rotation(base_pitch, base_yaw + yaw_offset)
    return (_OPTICAL @ src.T @ ref @ _OPTICAL.T).T


def fisheye_focal(width: int, height: int, diagonal_deg: float) -> float:
    """Pixels per radian. Equidistant, so this is the entire camera model."""
    return math.hypot(width, height) / 2.0 / (math.radians(diagonal_deg) / 2.0)


def build_maps(width: int, height: int, source_px: int, source_fov_deg: float,
               diagonal_deg: float, yaw_offsets, base_pitch: float,
               base_yaw: float) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    focal = fisheye_focal(width, height, diagonal_deg)
    u = np.arange(width, dtype=np.float64) - (width - 1) / 2.0
    v = np.arange(height, dtype=np.float64) - (height - 1) / 2.0
    x, y = np.meshgrid(u, v)
    r = np.hypot(x, y)
    theta = r / focal
    scale = np.where(r > 1e-9, np.sin(theta) / np.maximum(r, 1e-9), 1.0 / focal)
    ray = np.stack([x * scale, y * scale, np.cos(theta)], axis=-1)

    fx = (source_px / 2.0) / math.tan(math.radians(source_fov_deg) / 2.0)
    centre = (source_px - 1) / 2.0
    maps = []
    for offset in yaw_offsets:
        turned = ray @ source_rotation(base_pitch, base_yaw, offset)
        z = turned[..., 2]
        with np.errstate(divide='ignore', invalid='ignore'):
            sx = fx * turned[..., 0] / z + centre
            sy = fx * turned[..., 1] / z + centre
        inside = ((z > 1e-6) & (sx >= 0) & (sx <= source_px - 1)
                  & (sy >= 0) & (sy <= source_px - 1))
        maps.append((np.where(inside, sx, -1).astype(np.float32),
                     np.where(inside, sy, -1).astype(np.float32), inside))
    return maps


class EndAssembler:
    """One end of the car: two source frames in, one fisheye out."""

    def __init__(self, node: Node, end: str, width: int, height: int,
                 diagonal_deg: float, source_fov_deg: float,
                 yaw_offsets, base_pitch: float, base_yaw: float,
                 queue_depth: int, tolerance_ns: int,
                 source_reliability: str):
        self.node = node
        self.end = end
        self.width, self.height = width, height
        self.diagonal_deg = diagonal_deg
        self.source_fov_deg = source_fov_deg
        self.yaw_offsets = yaw_offsets
        self.base_pitch, self.base_yaw = base_pitch, base_yaw
        self.queue_depth = queue_depth
        self.maps: Optional[list] = None
        self.joint: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.canvas: Optional[np.ndarray] = None
        self.pending: List[Dict[int, Image]] = [dict(), dict()]
        self.made = 0
        self.dropped = 0
        self.tolerance_ns = tolerance_ns
        self.skew_ns: List[int] = []
        self.spent_ms: List[float] = []

        output_qos = QoSProfile(depth=10)
        output_qos.reliability = ReliabilityPolicy.BEST_EFFORT
        source_qos = QoSProfile(depth=max(queue_depth, 10))
        source_qos.reliability = (
            ReliabilityPolicy.RELIABLE
            if source_reliability == 'reliable'
            else ReliabilityPolicy.BEST_EFFORT
        )
        base = f'/sensing/camera/{end}'
        self.publisher = node.create_publisher(
            Image, f'{base}/fisheye/image_raw', output_qos)
        self.info_publisher = node.create_publisher(
            CameraInfo, f'{base}/fisheye/camera_info', output_qos)
        self.subscriptions = [
            node.create_subscription(
                Image, f'{base}/fisheye_{tag}/image_raw',
                (lambda msg, slot=slot: self.take(slot, msg)), source_qos)
            for slot, tag in enumerate(('a', 'b'))
        ]

    @staticmethod
    def _key(msg: Image) -> int:
        return msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec

    def take(self, slot: int, msg: Image) -> None:
        """Hold the frame until its partner turns up, matched by nearest stamp.

        Pairing by stamp rather than by arrival: the two sources are separate
        CARLA sensors and the bridge delivers them in whatever order the
        simulator finished them. Exact equality was tried first and threw away
        596 of 619 frames at one end -- the bridge stamps each sensor as it
        converts it, so two cameras ticked together still differ by a fraction
        of a step. Nearest-within-tolerance keeps them.
        """
        key = self._key(msg)
        other = self.pending[1 - slot]
        mate_key = None
        if other:
            mate_key = min(other, key=lambda k: abs(k - key))
            if abs(mate_key - key) > self.tolerance_ns:
                mate_key = None
        if mate_key is None:
            held = self.pending[slot]
            held[key] = msg
            while len(held) > self.queue_depth:
                held.pop(next(iter(held)))
                self.dropped += 1
            return
        mate = other.pop(mate_key)
        self.skew_ns.append(abs(mate_key - key))
        frames = [None, None]
        frames[slot] = msg
        frames[1 - slot] = mate
        self.emit(frames)

    def _as_array(self, msg: Image) -> np.ndarray:
        channels = 1 if msg.encoding == 'mono8' else msg.step // msg.width
        data = np.frombuffer(msg.data, np.uint8)
        frame = data.reshape(msg.height, msg.step // channels, channels)
        frame = frame[:, :msg.width]
        return frame[:, :, 0] if channels == 1 else frame

    def emit(self, frames: List[Image]) -> None:
        if cv2 is None:
            return
        started = time.perf_counter()
        arrays = [self._as_array(m) for m in frames]
        if self.joint is None:
            self.prepare(arrays)

        # 소스를 하나로 이어 붙여 remap 을 한 번만 돈다. 겹침은 맵을 만들 때
        # 이미 갈라 두었으므로 프레임마다 마스크를 씌울 일이 없다. 프레임당
        # 8.9 ms 에서 1.5 ms 로 줄어드는 자리다.
        side = arrays[0].shape[1]
        for index, source in enumerate(arrays):
            self.canvas[:, index * side:(index + 1) * side] = source
        out = cv2.remap(self.canvas, self.joint[0], self.joint[1],
                        cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        message = Image()
        message.header = frames[0].header
        message.height, message.width = self.height, self.width
        message.encoding = frames[0].encoding
        message.is_bigendian = frames[0].is_bigendian
        message.step = self.width * (1 if out.ndim == 2 else out.shape[2])
        # Not `= out.tobytes()`. The generated setter for a uint8[] field has
        # one fast path -- an array.array of typecode 'B' is taken as is --
        # and for anything else it asserts every element is an int in range,
        # in Python, one at a time. That check is the whole cost of this node:
        # 7.8 ms of a 640x360 frame and 30.9 ms of a 1280x720 one, against
        # 0.4 ms for the remap that does the actual work. Through the fast
        # path the same assignment is 0.015 ms.
        message.data = array.array('B', out.tobytes())
        self.publisher.publish(message)
        self.info_publisher.publish(self.camera_info(frames[0]))
        self.made += 1
        self.spent_ms.append(1000.0 * (time.perf_counter() - started))

    def prepare(self, arrays: List[np.ndarray]) -> None:
        """Build the remap tables once, with the overlap already resolved.

        Each output pixel is claimed by the first source that covers it, and
        the sources are listed axis-first, so the middle of the frame is never
        a seam. Storing that decision in the map means the per-frame work is a
        single remap.
        """
        side = arrays[0].shape[1]
        self.maps = build_maps(
            self.width, self.height, side, self.source_fov_deg,
            self.diagonal_deg, self.yaw_offsets, self.base_pitch, self.base_yaw)
        joint_x = np.full((self.height, self.width), -1.0, np.float32)
        joint_y = np.full((self.height, self.width), -1.0, np.float32)
        filled = np.zeros((self.height, self.width), bool)
        for index, (mx, my, inside) in enumerate(self.maps):
            take = inside & ~filled
            joint_x[take] = mx[take] + index * side
            joint_y[take] = my[take]
            filled |= take
        self.joint = (joint_x, joint_y)
        shape = (arrays[0].shape[0], side * len(arrays)) + arrays[0].shape[2:]
        self.canvas = np.empty(shape, np.uint8)
        self.node.get_logger().info(
            f'{self.end}: {side}x{arrays[0].shape[0]} 소스 {len(arrays)}장 -> '
            f'{self.width}x{self.height} 어안, 채워짐 {100.0 * filled.mean():.1f}%')

    def camera_info(self, source: Image) -> CameraInfo:
        """Equidistant intrinsics: fx = fy = pixels per radian, no distortion.

        The model is stated as `equidistant` so that consumers pick the fisheye
        branch. geometry.py already has one -- it undistorts feature points
        rather than whole images, which is the right way round, since rectifying
        the picture would spend back the resolution the projection just saved.
        """
        focal = fisheye_focal(self.width, self.height, self.diagonal_deg)
        info = CameraInfo()
        info.header = source.header
        info.height, info.width = self.height, self.width
        info.distortion_model = 'equidistant'
        info.d = [0.0, 0.0, 0.0, 0.0]
        cx, cy = (self.width - 1) / 2.0, (self.height - 1) / 2.0
        info.k = [focal, 0.0, cx, 0.0, focal, cy, 0.0, 0.0, 1.0]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [focal, 0.0, cx, 0.0, 0.0, focal, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        return info


class FisheyeAssembler(Node):
    def __init__(self):
        super().__init__('fisheye_assembler')
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 360)
        self.declare_parameter('diagonal_deg', 160.0)
        self.declare_parameter('source_fov_deg', 100.0)
        self.declare_parameter('yaw_offsets_deg', [-35.0, 35.0])
        self.declare_parameter('pitch_deg', -30.0)
        self.declare_parameter('ends', ['front', 'rear'])
        self.declare_parameter('base_yaw_deg', [0.0, 180.0])
        self.declare_parameter('queue_depth', 100)
        self.declare_parameter('source_reliability', 'reliable')
        self.declare_parameter('report_period_sec', 10.0)
        # 한 스텝의 절반. 같은 틱에 찍힌 두 장은 이 안에 든다.
        self.declare_parameter('tolerance_ms', 15.0)

        width = int(self.get_parameter('width').value)
        height = int(self.get_parameter('height').value)
        diagonal = float(self.get_parameter('diagonal_deg').value)
        source_fov = float(self.get_parameter('source_fov_deg').value)
        offsets = [float(v) for v in self.get_parameter('yaw_offsets_deg').value]
        pitch = float(self.get_parameter('pitch_deg').value)
        ends = list(self.get_parameter('ends').value)
        base_yaws = [float(v) for v in self.get_parameter('base_yaw_deg').value]
        depth = int(self.get_parameter('queue_depth').value)
        source_reliability = str(
            self.get_parameter('source_reliability').value
        ).lower()
        tolerance = int(float(self.get_parameter('tolerance_ms').value) * 1e6)

        if cv2 is None:
            self.get_logger().error('cv2 가 없어 합성할 수 없다')
        self.ends = [
            EndAssembler(self, end, width, height, diagonal, source_fov,
                         offsets, pitch, base_yaw, depth, tolerance,
                         source_reliability)
            for end, base_yaw in zip(ends, base_yaws)
        ]
        focal = fisheye_focal(width, height, diagonal)
        self.get_logger().info(
            f'어안 {width}x{height} 대각 {diagonal}°, '
            f'{focal * math.radians(1):.2f} px/°, '
            f'수평 {math.degrees(width / focal):.1f}° '
            f'수직 {math.degrees(height / focal):.1f}°')
        period = float(self.get_parameter('report_period_sec').value)
        if period > 0.0:
            self.create_timer(period, self.report)

    def report(self) -> None:
        parts = []
        for e in self.ends:
            skew = (f'{np.median(e.skew_ns) / 1e6:.1f}ms' if e.skew_ns else '-')
            spent = (f'{np.median(e.spent_ms):.1f}ms' if e.spent_ms else '-')
            parts.append(f'{e.end} {e.made} 장 (버림 {e.dropped}, '
                         f'스탬프차 {skew}, 합성 {spent})')
            e.skew_ns.clear()
            e.spent_ms.clear()
        self.get_logger().info('발행: ' + ', '.join(parts))


def main(args=None):
    rclpy.init(args=args)
    node = FisheyeAssembler()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
