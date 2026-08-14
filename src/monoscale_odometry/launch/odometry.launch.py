"""The tracker, the odometry and the grid it feeds."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    share = FindPackageShare('monoscale_odometry')
    base = PathJoinSubstitution([share, 'config', 'vision_fisheye.param.yaml'])
    candidate = PathJoinSubstitution(
        [share, 'config', 'odometry_candidate.param.yaml']
    )
    evaluation = PathJoinSubstitution([share, 'config', 'evaluation.param.yaml'])
    use_sim_time = LaunchConfiguration('use_sim_time')

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
        parameters=[base, candidate, {'use_sim_time': use_sim_time}],
    )
    occupancy = Node(
        package='monoscale_occupancy_grid_map',
        executable='monoscale_occupancy_grid_map',
        name='monoscale_occupancy_grid_map',
        output='screen',
        condition=IfCondition(LaunchConfiguration('launch_occupancy')),
        parameters=[base, {'use_sim_time': use_sim_time}],
    )
    evaluator = Node(
        package='monoscale_evaluation',
        executable='odometry_evaluator',
        name='odometry_evaluator',
        output='screen',
        condition=IfCondition(LaunchConfiguration('launch_evaluator')),
        parameters=[
            evaluation,
            {
                'use_sim_time': use_sim_time,
                'output_directory': LaunchConfiguration('output_directory'),
            },
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument('use_sim_time', default_value='true'),
            DeclareLaunchArgument('launch_occupancy', default_value='true'),
            DeclareLaunchArgument('launch_evaluator', default_value='false'),
            DeclareLaunchArgument('output_directory', default_value=''),
            tracker,
            odometry,
            occupancy,
            evaluator,
        ]
    )
