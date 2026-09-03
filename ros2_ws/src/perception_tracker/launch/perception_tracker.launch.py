"""Launch the replay-oriented perception tracker node.

TF is intentionally not fabricated here.  Supply the robot's TF tree from a
driver, URDF, or another launch description and set ``require_tf:=true`` when
the replay must verify it.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent
from launch.conditions import IfCondition
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    perception_node = Node(
        package="perception_tracker",
        executable="perception_tracker_node",
        name="perception_tracker_node",
        output="screen",
        on_exit=[EmitEvent(event=Shutdown(reason="KITTI replay complete"))],
        parameters=[{
            "dataset_root": LaunchConfiguration("dataset_root"),
            "sequence": LaunchConfiguration("sequence"),
            "start_frame": LaunchConfiguration("start_frame"),
            "end_frame": LaunchConfiguration("end_frame"),
            "replay_rate": LaunchConfiguration("replay_rate"),
            "detector_type": LaunchConfiguration("detector_type"),
            "base_frame": LaunchConfiguration("base_frame"),
            "lidar_frame": LaunchConfiguration("lidar_frame"),
            "camera_frame": LaunchConfiguration("camera_frame"),
            "detection_frame": LaunchConfiguration("detection_frame"),
            "require_tf": LaunchConfiguration("require_tf"),
        }],
    )
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="perception_tracker_rviz",
        arguments=["-d", PathJoinSubstitution([
            FindPackageShare("perception_tracker"), "rviz", "perception_tracker.rviz",
        ])],
        condition=IfCondition(LaunchConfiguration("use_rviz")),
    )
    return LaunchDescription([
        DeclareLaunchArgument("dataset_root", default_value="data"),
        DeclareLaunchArgument("sequence", default_value="0000"),
        DeclareLaunchArgument("start_frame", default_value="0"),
        DeclareLaunchArgument("end_frame", default_value="-1"),
        DeclareLaunchArgument("replay_rate", default_value="10.0"),
        DeclareLaunchArgument("detector_type", default_value="gt"),
        DeclareLaunchArgument("base_frame", default_value="base_link"),
        DeclareLaunchArgument("lidar_frame", default_value="lidar"),
        DeclareLaunchArgument("camera_frame", default_value="camera"),
        DeclareLaunchArgument("detection_frame", default_value="camera"),
        DeclareLaunchArgument("require_tf", default_value="false"),
        DeclareLaunchArgument("use_rviz", default_value="false"),
        perception_node,
        rviz_node,
    ])
