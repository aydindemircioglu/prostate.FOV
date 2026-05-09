
_base_ = [
    '../../mmrotate/configs/rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_ms_rr_le90.py'
]

angle_version = 'le90'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1184, 1184)),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=angle_version),
    dict(
        type='PolyRandomRotate',
        rotate_ratio=0.5,
        angles_range=180,
        auto_bound=False,
        version=angle_version),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

data_root = 'NOT_USED'
classes = ('cor','tra', )
dataset_type = 'DOTADataset'
data = dict(
    train=dict(
        pipeline=train_pipeline,
        version=angle_version,
        classes = classes,
        ann_file=data_root + 'train/annotations/',
        img_prefix=data_root + 'train/images/'),
    val=dict(
        version=angle_version,
        classes = classes,
        ann_file=data_root + 'test/annotations/',
        img_prefix=data_root + 'test/images/'),
    test=dict(
        version=angle_version,
        classes = classes,
        ann_file=data_root + 'test/annotations/',
        img_prefix=data_root + 'test/images/'))

load_from = "../data/pretrained/rotated_retinanet_obb_r50_fpn_1x_dota_ms_rr_le90-1da1ec9c.pth"
