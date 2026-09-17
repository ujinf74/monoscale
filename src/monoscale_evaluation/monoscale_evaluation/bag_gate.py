"""Refuses a recording that cannot be benchmarked on, before anything is built from it.

Every check here exists because its absence cost a day. The gyro one is the
reason the module exists: CARLA's default PhysX substep is too coarse for the
reported angular velocity to integrate to the rotation the vehicle's own
transform performs, so a drive recorded through an unpatched bridge carries a
phantom yaw rate of up to 0.9 deg/s. It is deterministic and depends on where
the vehicle is, which is why some drives carry it and others do not, and why
it survived a truth-vs-estimate comparison for weeks: the estimator was tuned
against it, so the tuning hid it.

The others are the recording mistakes that followed it. A drive re-recorded on
record_run.sh's default spawn instead of its own runs out of lane and hits
something; the bag still looks recorded, and only the accelerometer says
otherwise. A drive given the pilot log's "manoeuvre" figure as its duration
runs far past its road.

Usage:
    ros2 run monoscale_evaluation bag_gate <bag> [<bag> ...]

Exit status is non-zero if any bag fails, so it can gate a recording script.
"""

import math
import sys

import numpy as np

# A drive that never leaves its lane peaks around 1-2 m/s^2. The parking
# manoeuvres legitimately reach 34-60 through their gear changes, so the bar is
# set where nothing but a collision lives.
COLLISION_ACCEL_MPS2 = 50.0
# Integrated gyro against truth yaw, over the whole drive. A clean recording
# reads 0.005-0.10 deg; the artefact reads 2-54.
GYRO_DRIFT_DEG = 0.5
# Fraction of the drive spent stopped. Parking drives are exempt by name.
STATIONARY_FRACTION = 0.15
STATIONARY_MPS = 0.5


def _yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def read(path):
    from geometry_msgs.msg import PoseWithCovarianceStamped
    from rclpy.serialization import deserialize_message
    from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
    from sensor_msgs.msg import Image, Imu

    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=path, storage_id='sqlite3'),
        ConverterOptions('cdr', 'cdr'))
    imu, truth, front, rear = [], [], [], []
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic.endswith('imu/imu_data'):
            m = deserialize_message(data, Imu)
            imu.append((
                m.header.stamp.sec + m.header.stamp.nanosec * 1e-9,
                m.angular_velocity.z, abs(m.linear_acceleration.x)))
        elif topic.endswith('ground_truth/pose'):
            m = deserialize_message(data, PoseWithCovarianceStamped)
            p = m.pose.pose.position
            truth.append((
                m.header.stamp.sec + m.header.stamp.nanosec * 1e-9,
                p.x, p.y, _yaw(m.pose.pose.orientation)))
        elif topic.endswith('front/fisheye/image_raw'):
            m = deserialize_message(data, Image)
            front.append(m.header.stamp.sec + m.header.stamp.nanosec * 1e-9)
        elif topic.endswith('rear/fisheye/image_raw'):
            m = deserialize_message(data, Image)
            rear.append(m.header.stamp.sec + m.header.stamp.nanosec * 1e-9)
    return (np.array(imu), np.array(truth), np.array(front), np.array(rear))


def check(path):
    """Returns (ok, summary, list of complaints)."""
    imu, truth, front, rear = read(path)
    bad = []
    if len(imu) < 10 or len(truth) < 10:
        return False, {}, ['no IMU or no truth']

    duration = truth[-1, 0] - truth[0, 0]
    step = np.hypot(np.diff(truth[:, 1]), np.diff(truth[:, 2]))
    distance = float(step.sum())
    speed = step / np.maximum(np.diff(truth[:, 0]), 1e-9)

    # The gyro against the truth it is supposed to describe. Integrated, not
    # differenced: the artefact is a bias, and a bias only shows over time.
    yaw = np.interp(imu[:, 0], truth[:, 0], np.unwrap(truth[:, 3]))
    dt = np.diff(imu[:, 0])
    integrated = float(np.sum(0.5 * (imu[:-1, 1] + imu[1:, 1]) * dt))
    drift = math.degrees(integrated - (yaw[-1] - yaw[0]))

    peak = float(imu[:, 2].max())
    stopped = float((speed < STATIONARY_MPS).mean())
    parking = 'park' in path.rsplit('/', 1)[-1]

    summary = {
        'duration_s': duration, 'distance_m': distance,
        'speed_mps': distance / max(duration, 1e-9),
        'gyro_drift_deg': drift, 'peak_accel_mps2': peak,
        'stopped': stopped, 'front': len(front), 'rear': len(rear),
    }

    if abs(drift) > GYRO_DRIFT_DEG:
        bad.append(
            f'gyro integration disagrees with truth by {drift:+.3f} deg '
            f'(limit {GYRO_DRIFT_DEG} deg) -- this can be the PhysX substep defect: '
            'check that max_substep_delta_time in carla_autoware.py is 0.002')
    if not parking and peak > COLLISION_ACCEL_MPS2:
        bad.append(f'|accX| peaks at {peak:.0f} m/s^2 -- a collision (limit {COLLISION_ACCEL_MPS2:.0f})')
    if not parking and stopped > STATIONARY_FRACTION:
        bad.append(f'{stopped * 100:.0f}% of samples are stationary -- it may have left the road')
    if len(front) and len(rear):
        # Nearest stamp, not the same index. One end starting a frame earlier
        # shifts every index-wise pair by a whole period and reports a skew the
        # estimator -- which pairs by minimum spread -- never sees. That false
        # alarm has been raised on this bag set once already.
        order = np.argsort(rear)
        rs = rear[order]
        at = np.clip(np.searchsorted(rs, front), 1, len(rs) - 1)
        near = np.minimum(np.abs(front - rs[at]), np.abs(front - rs[at - 1]))
        # A share, not a maximum: a recording that stops between the two ends
        # leaves one frame at the tail without a partner, every time, and that
        # is not a fault.
        unpaired = float((near > 1e-3).mean()) if len(near) else 0.0
        skew = float(np.median(near)) if len(near) else 0.0
        gaps = np.diff(front)
        nominal = float(np.median(gaps)) if len(gaps) else 0.0
        summary['skew_ms'] = skew * 1e3
        summary['unpaired'] = unpaired
        summary['drops'] = int((gaps > nominal * 1.5).sum()) if nominal > 0 else 0
        if unpaired > 0.01:
            bad.append(
                f'{unpaired * 100:.1f}% of front frames find no rear partner '
                f'(median skew {skew * 1e3:.2f} ms) -- check FISHEYE_SPLIT')
        if summary['drops']:
            bad.append(f"{summary['drops']} dropped frames")
    else:
        bad.append('no images')
    return not bad, summary, bad


def main(argv=None):
    paths = (argv if argv is not None else sys.argv[1:])
    if not paths:
        print(__doc__)
        return 2
    worst = 0
    for path in paths:
        try:
            ok, s, bad = check(path)
        except Exception as error:                       # noqa: BLE001
            print(f'{path}: could not read -- {error}')
            worst = 1
            continue
        name = path.rstrip('/').rsplit('/', 1)[-1]
        if s:
            print(
                f"{name:<28s} {s['duration_s']:5.1f}s {s['distance_m']:7.2f}m "
                f"{s['speed_mps']:5.2f}m/s  gyro {s['gyro_drift_deg']:+7.3f}deg  "
                f"|accX| {s['peak_accel_mps2']:6.1f}  stopped {s['stopped'] * 100:4.1f}%  "
                f"{'pass' if ok else 'fail'}")
        for line in bad:
            print(f'    - {line}')
        worst = max(worst, 0 if ok else 1)
    return worst


if __name__ == '__main__':
    sys.exit(main())
