"""Safe 20-epoch continuation from the validated epoch-20 weights.

This is an additional 20-epoch run (40 epochs total across both runs).  The
checkpoint is loaded as model weights rather than resumed as optimizer state,
so the learning rate remains the explicitly configured small value.
"""

_base_ = ['./pointpillars_kitti_finetune_40e.py']

runner = dict(_delete_=True, type='EpochBasedRunner', max_epochs=20)
optimizer = dict(lr=0.000002)
lr_config = dict(_delete_=True, policy='step', step=[12, 18], gamma=0.1)
momentum_config = None
checkpoint_config = dict(interval=1, max_keep_ckpts=5)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
evaluation = dict(interval=1000)
work_dir = 'outputs/pointpillars_finetune_extra20'
load_from = 'outputs/pointpillars_finetune/epoch_20.pth'
