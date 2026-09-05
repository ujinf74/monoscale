"""The tracker, the C++ odometry and the grid it feeds.

The parameter files are the ones the Python stack already carries, so the two
can be compared on the same configuration. Two values are set here rather than
left to the files.

`mapping_min_period_sec` is set explicitly because the shared parameter file
does not carry it: it ends with a comment about the setting and no setting, so
the code default applies and the map is rebuilt at the camera rate. That is the
opposite of what the comment says, and on the Python it is also fatal -- the
track path has no images for the keyframe triangulation the mapper reaches for.

`extrinsics_from_tf` is off because these bags publish no transform tree, and
the mounts in the parameter file are the ones every recorded measurement was
taken with.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    shared = FindPackageShare('monoscale_odometry')
    base = PathJoinSubstitution([shared, 'config', 'vision_fisheye.param.yaml'])
    candidate = PathJoinSubstitution(
        [shared, 'config', 'odometry_candidate.param.yaml']
    )
    use_sim_time = LaunchConfiguration('use_sim_time')
    mapping_period = LaunchConfiguration('mapping_min_period_sec')

    tracker = Node(
        package='monoscale_tracker',
        executable='feature_tracker',
        name='feature_tracker',
        output='screen',
        parameters=[candidate, {'use_sim_time': use_sim_time}],
    )
    odometry = Node(
        package='monoscale_odometry',
        executable='monoscale_odometry',
        name='monoscale_odometry',
        output='screen',
        parameters=[
            base,
            candidate,
            {
                'use_sim_time': use_sim_time,
                'mapping_min_period_sec': mapping_period,
                'extrinsics_from_tf': False,
            },
        ],
    )
    # The grid comes off a plane sweep of the raw fisheye images now, not off
    # the estimator's labelled points. The old path built it from whatever the
    # tracker's corners happened to triangulate, so an obstacle that gave up no
    # corners did not exist; and its depth chain carried five faults the audit
    # of 2026-09-02 measured, of which the largest -- a warp holding pitch and
    # roll at zero -- accounted for 94% of the frame-common error. The sweep
    # decides height per pixel by photoconsistency instead, and takes roll and
    # pitch from the odometry it integrates against.
    occupancy = Node(
        package='monoscale_sweep',
        executable='sweep_node',
        name='monoscale_sweep',
        output='screen',
        condition=IfCondition(LaunchConfiguration('launch_occupancy')),
        parameters=[
            PathJoinSubstitution(
                [FindPackageShare('monoscale_sweep'), 'config',
                 'sweep.param.yaml']
            ),
            {'use_sim_time': use_sim_time},
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument('use_sim_time', default_value='true'),
            DeclareLaunchArgument('launch_occupancy', default_value='true'),
            # Obstacles need `obstacle_slip_baseline_m` on this path; the grid
            # is otherwise built from free space alone, which is still the
            # half of it that was ever reliable.
            DeclareLaunchArgument('mapping_min_period_sec', default_value='0.2'),
            tracker,
            odometry,
            occupancy,
        ]
    )
