"""ROS2 replay adapter around the repository's ROS-independent pipeline."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import numpy as np

# The repository is intentionally not installed as a second ROS package.  Find
# its src directory when running from either ros2_ws/src or a symlink install.
def _ensure_core_importable() -> None:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "src" / "robot_perception"
        if candidate.is_dir():
            src_path = str(parent / "src")
            if src_path not in sys.path:
                sys.path.insert(0, src_path)
            return
    raise RuntimeError("cannot locate repository src/robot_perception; set PYTHONPATH to the repository src")


_ensure_core_importable()

from robot_perception.dataset import KittiTrackingDataset  # noqa: E402
from robot_perception.detection import GTDetector, NoisyGTDetector  # noqa: E402
from robot_perception.tracking import MultiObjectTracker  # noqa: E402


def _as_float_parameter(node: Any, name: str, default: float) -> float:
    value = float(node.declare_parameter(name, default).value)
    if not np.isfinite(value):
        raise ValueError(f"parameter {name} must be finite")
    return value


class PerceptionTrackerNode:
    """Replay KITTI frames and publish sensor/detection/track visualization topics.

    The class is created only after ROS2 is imported so importing the core
    package or running its pytest suite never requires ``rclpy``.
    """

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        # Defining the ROS subclass dynamically avoids importing rclpy at module
        # import time, which is useful for the package smoke test.
        from rclpy.node import Node

        class _Node(Node):
            def __init__(self) -> None:
                super().__init__("perception_tracker_node")
                self._configure()

            def _configure(self) -> None:
                from cv_bridge import CvBridge
                from rclpy.duration import Duration
                from rcl_interfaces.msg import ParameterDescriptor
                from rclpy.time import Time
                from sensor_msgs.msg import Image, PointCloud2, PointField
                from sensor_msgs_py import point_cloud2
                from std_msgs.msg import Header
                from visualization_msgs.msg import Marker, MarkerArray
                import tf2_ros

                self._Image = Image
                self._PointCloud2 = PointCloud2
                self._PointField = PointField
                self._point_cloud2 = point_cloud2
                self._Header = Header
                self._Marker = Marker
                self._MarkerArray = MarkerArray
                self._bridge = CvBridge()
                self._tf_buffer = tf2_ros.Buffer()
                self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)
                self._tf_duration = Duration(seconds=0.05)
                self._Time = Time

                dynamic_text = ParameterDescriptor(dynamic_typing=True)
                self.dataset_root = Path(str(self.declare_parameter("dataset_root", "data", dynamic_text).value))
                # ROS 命令行会把未加引号的 0000 解析成整数；动态类型后统一格式化为 KITTI 四位序列号。
                sequence_value = self.declare_parameter("sequence", "0000", dynamic_text).value
                self.sequence = f"{int(sequence_value):04d}" if str(sequence_value).isdigit() else str(sequence_value)
                self.start_frame = int(self.declare_parameter("start_frame", 0).value)
                self.end_frame = int(self.declare_parameter("end_frame", -1).value)
                self.replay_rate = _as_float_parameter(self, "replay_rate", 10.0)
                if self.replay_rate <= 0.0:
                    raise ValueError("replay_rate must be positive")
                self.detector_type = str(self.declare_parameter("detector_type", "gt").value).lower()
                score_threshold = _as_float_parameter(self, "score_threshold", 0.0)
                class_ids = [int(item) for item in self.declare_parameter("class_ids", [1, 2, 3]).value]
                self.base_frame = str(self.declare_parameter("base_frame", "base_link").value)
                self.lidar_frame = str(self.declare_parameter("lidar_frame", "lidar").value)
                self.camera_frame = str(self.declare_parameter("camera_frame", "camera").value)
                self.detection_frame = str(
                    self.declare_parameter("detection_frame", "camera", dynamic_text).value
                )
                if self.detection_frame not in {self.lidar_frame, self.camera_frame}:
                    raise ValueError("detection_frame must equal lidar_frame or camera_frame")
                self.require_tf = bool(self.declare_parameter("require_tf", False).value)
                self.shutdown_on_end = bool(self.declare_parameter("shutdown_on_end", True).value)
                noise_seed = int(self.declare_parameter("noise_seed", 7).value)

                if self.detector_type == "gt":
                    self.detector = GTDetector(score_threshold=score_threshold, class_ids=class_ids)
                elif self.detector_type in {"noisy_gt", "noisy-gt"}:
                    position_std = self.declare_parameter("position_std", [0.2, 0.2, 0.2]).value
                    self.detector = NoisyGTDetector(
                        position_std=position_std,
                        score_threshold=score_threshold,
                        class_ids=class_ids,
                        random_seed=noise_seed,
                    )
                else:
                    raise ValueError("detector_type must be 'gt' or 'noisy_gt' for the replay node")

                tracker_min_hits = int(self.declare_parameter("min_hits", 3).value)
                tracker_max_age = int(self.declare_parameter("max_age", 5).value)
                gating_threshold = _as_float_parameter(self, "gating_threshold", 9.21)
                self.tracker = MultiObjectTracker(
                    min_hits=tracker_min_hits,
                    max_age=tracker_max_age,
                    gating_threshold=gating_threshold,
                    initial_covariance=1.0,
                    process_noise=0.01,
                    measurement_noise=0.1,
                )

                self.dataset = KittiTrackingDataset(
                    self.dataset_root,
                    sequences=[self.sequence],
                    load_images=True,
                    load_annotations=True,
                )
                # 使用 loader 的帧索引筛选，避免启动时为整段序列解码图片和点云。
                frame_keys = getattr(self.dataset, "_frame_keys", ())
                frame_ids = [key.frame_id for key in frame_keys]
                if not frame_ids:
                    frame_ids = [self.dataset[index].frame_id for index in range(len(self.dataset))]
                selected = [
                    index
                    for index, frame_id in enumerate(frame_ids)
                    if frame_id >= self.start_frame and (self.end_frame < 0 or frame_id <= self.end_frame)
                ]
                if not selected:
                    raise ValueError("no KITTI frames selected by start_frame/end_frame")
                self._selected_indices = selected
                self._cursor = 0

                self.points_pub = self.create_publisher(PointCloud2, "/sensors/lidar/points", 10)
                self.image_pub = self.create_publisher(Image, "/sensors/camera/image", 10)
                self.detections_pub = self.create_publisher(MarkerArray, "/perception/detections", 10)
                self.tracks_pub = self.create_publisher(MarkerArray, "/perception/tracks", 10)
                self._timer = self.create_timer(1.0 / self.replay_rate, self._replay_once)
                self.get_logger().info(
                    f"KITTI replay ready: sequence={self.sequence}, frames={len(selected)}, "
                    f"detector={self.detector_type}"
                )

            def _header(self, stamp: Any, frame_id: str) -> Any:
                header = self._Header()
                header.stamp = stamp
                header.frame_id = frame_id
                return header

            def _point_cloud_message(self, points: np.ndarray, header: Any) -> Any:
                fields = [
                    self._PointField(name=name, offset=offset, datatype=self._PointField.FLOAT32, count=1)
                    for name, offset in (("x", 0), ("y", 4), ("z", 8), ("intensity", 12))
                ]
                return self._point_cloud2.create_cloud(header, fields, np.asarray(points, dtype=np.float32).tolist())

            def _marker(self, marker_id: int, header: Any, *, namespace: str, position: np.ndarray,
                        dimensions: np.ndarray, yaw: float, color: tuple[float, float, float],
                        text: str = "") -> Any:
                marker = self._Marker()
                marker.header = header
                marker.ns = namespace
                marker.id = int(marker_id)
                marker.type = self._Marker.CUBE if not text else self._Marker.TEXT_VIEW_FACING
                marker.action = self._Marker.ADD
                marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = [float(v) for v in position]
                marker.pose.orientation.w = float(np.cos(yaw * 0.5))
                marker.pose.orientation.z = float(np.sin(yaw * 0.5))
                if text:
                    marker.scale.z = 0.8
                    marker.text = text
                else:
                    marker.scale.x, marker.scale.y, marker.scale.z = [float(v) for v in dimensions]
                marker.color.r, marker.color.g, marker.color.b = color
                marker.color.a = 0.85
                return marker

            def _publish_detections(self, detections: list[Any], header: Any) -> None:
                array = self._MarkerArray()
                colors = {1: (0.1, 0.9, 0.1), 2: (0.1, 0.4, 1.0), 3: (1.0, 0.6, 0.1)}
                for index, detection in enumerate(detections):
                    array.markers.append(self._marker(
                        index, header, namespace="detections", position=detection.position,
                        dimensions=detection.dimensions, yaw=detection.yaw,
                        color=colors.get(detection.class_id, (0.8, 0.8, 0.8)),
                    ))
                self.detections_pub.publish(array)

            def _publish_tracks(self, states: list[Any], header: Any) -> None:
                array = self._MarkerArray()
                active = {track.track_id: track for track in self.tracker.tracks}
                for index, state in enumerate(states):
                    track = active.get(state.track_id)
                    if track is None:
                        continue
                    color = (1.0, 0.1, 0.1)
                    array.markers.append(self._marker(
                        index, header, namespace="tracks", position=state.state[:3],
                        dimensions=track.dimensions, yaw=track.yaw, color=color,
                    ))
                    array.markers.append(self._marker(
                        10000 + index, header, namespace="track_ids", position=state.state[:3] + [0.0, 0.0, track.dimensions[2]],
                        dimensions=np.ones(3), yaw=0.0, color=(1.0, 1.0, 1.0), text=f"Track ID {state.track_id}",
                    ))
                self.tracks_pub.publish(array)

            def _check_tf(self, stamp: Any) -> None:
                if not self.require_tf:
                    return
                try:
                    self._tf_buffer.lookup_transform(
                        self.base_frame, self.lidar_frame, self._Time.from_msg(stamp), timeout=self._tf_duration
                    )
                    self._tf_buffer.lookup_transform(
                        self.base_frame, self.camera_frame, self._Time.from_msg(stamp), timeout=self._tf_duration
                    )
                except Exception as error:
                    raise RuntimeError(
                        f"required TF unavailable ({self.base_frame} <- {self.lidar_frame}/{self.camera_frame}): {error}"
                    ) from error

            def _replay_once(self) -> None:
                if self._cursor >= len(self._selected_indices):
                    if self.shutdown_on_end:
                        import rclpy
                        self.get_logger().info("KITTI replay complete")
                        self._timer.cancel()
                        rclpy.shutdown()
                    return
                frame = self.dataset[self._selected_indices[self._cursor]]
                self.get_logger().debug(f"replay callback for frame {frame.frame_id}")
                stamp = self.get_clock().now().to_msg()
                try:
                    self._check_tf(stamp)
                    detections = self.detector.predict(frame)
                    states = self.tracker.update(detections)
                    header_lidar = self._header(stamp, self.lidar_frame)
                    header_camera = self._header(stamp, self.camera_frame)
                    self.points_pub.publish(self._point_cloud_message(frame.lidar_points, header_lidar))
                    if frame.image is not None:
                        image_message = self._bridge.cv2_to_imgmsg(frame.image, encoding="bgr8")
                        image_message.header = header_camera
                        self.image_pub.publish(image_message)
                    detection_header = (
                        header_lidar if self.detection_frame == self.lidar_frame else header_camera
                    )
                    self._publish_detections(detections, detection_header)
                    self._publish_tracks(states, detection_header)
                    self.get_logger().debug(
                        f"replayed sequence={frame.sequence_id} frame={frame.frame_id}: "
                        f"detections={len(detections)} tracks={len(states)}"
                    )
                    self._cursor += 1
                except Exception as error:
                    self.get_logger().error(f"failed to replay frame {frame.frame_id}: {error}")
                    self._cursor += 1

        return _Node(*args, **kwargs)


def main(args: list[str] | None = None) -> int:
    import rclpy
    from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor

    rclpy.init(args=args)
    node = None
    executor = None
    try:
        node = PerceptionTrackerNode()
        # 显式 spin_once 循环使 replay 完成后的 rclpy.shutdown() 能可靠返回；
        # 某些 ROS2/FastDDS 版本在 timer 回调中直接调用 spin() 会迟迟不退出。
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        while rclpy.ok():
            try:
                executor.spin_once(timeout_sec=0.1)
            except ExternalShutdownException:
                break
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        if node is not None:
            node.get_logger().error(str(error))
        else:
            print(f"perception_tracker_node startup failed: {error}", file=sys.stderr)
        return 1
    finally:
        if executor is not None:
            executor.shutdown()
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0
