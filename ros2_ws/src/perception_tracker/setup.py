from setuptools import find_packages, setup

package_name = "perception_tracker"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/perception_tracker.launch.py"]),
        ("share/" + package_name + "/rviz", ["rviz/perception_tracker.rviz"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    description="Replay-oriented ROS2 adapter for robot-perception-mot.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "perception_tracker_node = perception_tracker.perception_tracker_node:main",
        ],
    },
)
