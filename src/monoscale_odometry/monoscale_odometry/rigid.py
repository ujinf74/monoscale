import numpy as np


def quaternion_matrix(quaternion):
    x, y, z, w = np.asarray(quaternion, dtype=np.float64)
    norm = x * x + y * y + z * z + w * w
    if norm < 1e-12:
        raise ValueError('quaternion has zero norm')
    scale = 2.0 / norm
    return np.array(
        [
            [1.0 - scale * (y * y + z * z), scale * (x * y - z * w), scale * (x * z + y * w)],
            [scale * (x * y + z * w), 1.0 - scale * (x * x + z * z), scale * (y * z - x * w)],
            [scale * (x * z - y * w), scale * (y * z + x * w), 1.0 - scale * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_quaternion(rotation):
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = 2.0 * np.sqrt(trace + 1.0)
        return np.array(
            [
                (rotation[2, 1] - rotation[1, 2]) / s,
                (rotation[0, 2] - rotation[2, 0]) / s,
                (rotation[1, 0] - rotation[0, 1]) / s,
                0.25 * s,
            ]
        )
    index = int(np.argmax(np.diag(rotation)))
    if index == 0:
        s = 2.0 * np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])
        return np.array(
            [0.25 * s, (rotation[0, 1] + rotation[1, 0]) / s,
             (rotation[0, 2] + rotation[2, 0]) / s,
             (rotation[2, 1] - rotation[1, 2]) / s]
        )
    if index == 1:
        s = 2.0 * np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])
        return np.array(
            [(rotation[0, 1] + rotation[1, 0]) / s, 0.25 * s,
             (rotation[1, 2] + rotation[2, 1]) / s,
             (rotation[0, 2] - rotation[2, 0]) / s]
        )
    s = 2.0 * np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])
    return np.array(
        [(rotation[0, 2] + rotation[2, 0]) / s,
         (rotation[1, 2] + rotation[2, 1]) / s, 0.25 * s,
         (rotation[1, 0] - rotation[0, 1]) / s]
    )


def base_pose_from_imu(position_global_imu, rotation_global_imu, imu_position_in_base):
    rotation_global_imu = np.asarray(rotation_global_imu, dtype=np.float64).reshape(3, 3)
    position_global_imu = np.asarray(position_global_imu, dtype=np.float64)
    lever = np.asarray(imu_position_in_base, dtype=np.float64)
    return position_global_imu - rotation_global_imu @ lever, rotation_global_imu


def base_velocity_from_imu(linear_imu, angular_base, imu_position_in_base):
    linear_imu = np.asarray(linear_imu, dtype=np.float64)
    angular_base = np.asarray(angular_base, dtype=np.float64)
    lever = np.asarray(imu_position_in_base, dtype=np.float64)
    return linear_imu - np.cross(angular_base, lever)
