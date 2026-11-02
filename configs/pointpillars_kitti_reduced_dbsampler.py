"""KITTI PointPillars config aligned with the official reduced-point input."""

_base_ = ['./pointpillars_kitti_dbsampler_finetune.py']

# Official MMDetection3D PointPillars consumes camera-FOV reduced clouds.
data = dict(
    train=dict(dataset=dict(pts_prefix='velodyne_reduced')),
    val=dict(pts_prefix='velodyne_reduced'),
    test=dict(pts_prefix='velodyne_reduced'),
)
