"""KITTI PointPillars fine-tuning with the official GT database sampler."""

_base_ = ['./pointpillars_kitti_finetune_40e.py']

data_root = 'mmdetection3d-1.0/data/kitti/'
class_names = ['Pedestrian', 'Cyclist', 'Car']
point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]
file_client_args = dict(backend='disk')

# Official class-balanced object sampling.  It is applied only to training;
# validation always uses untouched raw KITTI point clouds.
db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'kitti_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(Car=5, Pedestrian=5, Cyclist=5)),
    classes=class_names,
    sample_groups=dict(Car=15, Pedestrian=15, Cyclist=15),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args),
    file_client_args=file_client_args,
)

train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4,
         file_client_args=file_client_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True,
         file_client_args=file_client_args),
    dict(type='ObjectSample', db_sampler=db_sampler, use_ground_plane=False),
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
)

optimizer = dict(lr=0.00005)
lr_config = dict(_delete_=True, policy='step', step=[8, 12], gamma=0.1)
momentum_config = None
runner = dict(_delete_=True, type='EpochBasedRunner', max_epochs=15)
checkpoint_config = dict(interval=1, max_keep_ckpts=5)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
evaluation = dict(interval=1000)
work_dir = 'outputs/pointpillars_dbsampler_finetune'
load_from = 'weights/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth'
