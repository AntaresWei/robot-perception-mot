"""40-epoch continuation config for local KITTI PointPillars fine-tuning.

Run from the repository root.  The official PointPillars voxel layer remains
the only voxelization stage; the input pipeline only applies the model ROI
filter and the documented geometric augmentations.
"""

_base_ = [
    '../mmdetection3d-1.0/configs/pointpillars/'
    'hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py',
]

data_root = 'mmdetection3d-1.0/data/kitti/'
point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]
class_names = ['Pedestrian', 'Cyclist', 'Car']
file_client_args = dict(backend='disk')

# 保持与前 20 epoch 相同的输入定义：不叠加外部 voxel 或 ground removal，
# 避免续训时改变输入分布；PointPillars 自身负责 voxelization。
train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4,
         file_client_args=file_client_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True,
         file_client_args=file_client_args),
    dict(type='ObjectNoise', num_try=100,
         translation_std=[1.0, 1.0, 0.5],
         global_rot_range=[0.0, 0.0],
         rot_range=[-0.78539816, 0.78539816]),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='GlobalRotScaleTrans',
         rot_range=[-0.78539816, 0.78539816],
         scale_ratio_range=[0.95, 1.05]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d']),
]

test_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4,
         file_client_args=file_client_args),
    dict(type='MultiScaleFlipAug3D', img_scale=(1333, 800), pts_scale_ratio=1,
         flip=False, transforms=[
             dict(type='GlobalRotScaleTrans', rot_range=[0, 0],
                  scale_ratio_range=[1., 1.], translation_std=[0, 0, 0]),
             dict(type='RandomFlip3D'),
             dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
             dict(type='DefaultFormatBundle3D', class_names=class_names,
                  with_label=False),
             dict(type='Collect3D', keys=['points']),
         ]),
]

data = dict(
    samples_per_gpu=6,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type='KittiDataset',
            data_root=data_root,
            ann_file=data_root + 'kitti_infos_train.pkl',
            split='training',
            pts_prefix='velodyne',
            pipeline=train_pipeline,
            modality=dict(use_lidar=True, use_camera=False),
            classes=class_names,
            test_mode=False,
            box_type_3d='LiDAR',
            file_client_args=file_client_args)),
    val=dict(
        type='KittiDataset',
        data_root=data_root,
        ann_file=data_root + 'kitti_infos_val.pkl',
        split='training',
        pts_prefix='velodyne',
        pipeline=test_pipeline,
        modality=dict(use_lidar=True, use_camera=False),
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR',
        file_client_args=file_client_args),
    test=dict(
        type='KittiDataset',
        data_root=data_root,
        ann_file=data_root + 'kitti_infos_val.pkl',
        split='training',
        pts_prefix='velodyne',
        pipeline=test_pipeline,
        modality=dict(use_lidar=True, use_camera=False),
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR',
        file_client_args=file_client_args),
)

# 续训阶段沿用低学习率，避免从 epoch 20 的局部最优点跳回大步长。
optimizer = dict(lr=0.000002)
lr_config = dict(_delete_=True, policy='step', step=[28, 36], gamma=0.1)
momentum_config = None
runner = dict(type='EpochBasedRunner', max_epochs=40)
checkpoint_config = dict(interval=1, max_keep_ckpts=5)
# 旧版 TensorBoard hook 与 Python 3.8 distutils 不兼容；文本日志足够记录本次实验。
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
evaluation = dict(interval=1000)
work_dir = 'outputs/pointpillars_finetune_40e'
load_from = None
