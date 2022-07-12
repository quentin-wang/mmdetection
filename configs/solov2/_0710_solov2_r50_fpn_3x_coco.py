_base_ = 'solov2_r50_fpn_1x_coco.py'

classes = ('train', 'air', 'zeng', 'ding', 'taxi', 'others', 'card', 'juan')
num_classes = len(classes)
model = dict(
    # type='SOLOv2',
    mask_head=dict(
        # type='SOLOV2Head',
        num_classes=num_classes))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
#     dict(
#         type='Resize',
#         img_scale=[(1333, 800), (1333, 768), (1333, 736), (1333, 704),
#                    (1333, 672), (1333, 640)],
#         multiscale_mode='value',
#         keep_ratio=True),
#     dict(type='RandomFlip', flip_ratio=0.5),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='Pad', size_divisor=32),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
# ]

albu_train_transforms = [
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.0625,
        scale_limit=0.0,
        rotate_limit=5.0,
        interpolation=1,
        p=0.1),
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=[0.1, 0.3],
        contrast_limit=[0.1, 0.3],
        p=0.1),
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='RGBShift',
                r_shift_limit=10,
                g_shift_limit=10,
                b_shift_limit=10,
                p=1.0),
            dict(
                type='HueSaturationValue',
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.0)
        ],
        p=0.1),
    # dict(
    #     type='Perspective',
    #     p=0.2),
    dict(
        type='RandomRotate90',
        p=0.3),
    dict(type='GridDropout',
        p=0.5),
    dict(type='ChannelShuffle', p=0.1),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=3, p=1.0),
            dict(type='MedianBlur', blur_limit=3, p=1.0),
            dict(type='JpegCompression', quality_lower=85, quality_upper=95, p=1.0),
        ],
        p=0.1),
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=True),
    dict(
        type='Resize',
        # img_scale=[(768, 448), (768, 512), (768, 576), (768, 640), (768, 704), (768, 768)],
        img_scale=[
            (640, 640), (640, 608), (640, 576), (640, 544), (640, 512), (640, 480), (640, 448), (640, 416),
            (896, 896), (896, 848), (896, 800), (896, 752), (896, 704), (896, 672), (896, 624), (896, 576),
            (1152, 1152), (1152, 1088), (1152, 1024), (1152, 976), (1152, 912), (1152, 864), (1152, 800), (1152, 736),
            ], 
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='Pad', size_divisor=32),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_masks': 'masks',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(type='RandomFlip', flip_ratio=0.5), 
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 640),
        flip=False,
        transforms=[
            # dict(type='Pad', size_ratio=0.1), # size_divisor=32),
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# fp16 = dict(loss_scale=512.)
evaluation = dict(interval=1, metric='segm') #, classwise=True) # evaluation = dict(interval=2, metric='segm', classwise=True, iou_thrs=[0.75])
checkpoint_config = dict(interval=1)

# data = dict(train=dict(pipeline=train_pipeline))
data_root = 'datasets/invoice/'
dataset_type = 'CocoDataset'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train/annotation_coco.json',
        img_prefix=data_root + 'train/',
        classes=classes,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'val/annotation_coco.json',
        img_prefix=data_root + 'val/',
        classes=classes,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'val/annotation_coco.json',
        img_prefix=data_root + 'val/',
        classes=classes,
        pipeline=test_pipeline))

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[27, 33])
runner = dict(type='EpochBasedRunner', max_epochs=36)
