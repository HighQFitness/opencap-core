# ViTPose++-B multi-dataset config (ViTAE-Transformer/ViTPose), inference focused on
# COCO-WholeBody via associate head index 4 and dataset_idx 5.
# See: configs/.../vitPose+_base_coco+aic+mpii+ap10k+apt36k+wholebody_256x192_udp.py

log_level = 'INFO'
load_from = None
resume_from = None

target_type = 'GaussianHeatmap'

channel_cfg = dict(
    num_output_channels=17,
    dataset_joints=17,
    dataset_channel=[list(range(17))],
    inference_channel=list(range(17)))

aic_channel_cfg = dict(
    num_output_channels=14,
    dataset_joints=14,
    dataset_channel=[list(range(14))],
    inference_channel=list(range(14)))

mpii_channel_cfg = dict(
    num_output_channels=16,
    dataset_joints=16,
    dataset_channel=list(range(16)),
    inference_channel=list(range(16)))

ap10k_channel_cfg = dict(
    num_output_channels=17,
    dataset_joints=17,
    dataset_channel=[list(range(17))],
    inference_channel=list(range(17)))

cocowholebody_channel_cfg = dict(
    num_output_channels=133,
    dataset_joints=133,
    dataset_channel=[list(range(133))],
    inference_channel=list(range(133)))

model = dict(
    type='TopDownMoE',
    pretrained=None,
    backbone=dict(
        type='ViTMoE',
        img_size=(256, 192),
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        ratio=1,
        use_checkpoint=False,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.3,
        num_expert=6,
        part_features=192),
    keypoint_head=dict(
        type='TopdownHeatmapSimpleHead',
        in_channels=768,
        num_deconv_layers=2,
        num_deconv_filters=(256, 256),
        num_deconv_kernels=(4, 4),
        extra=dict(final_conv_kernel=1),
        out_channels=channel_cfg['num_output_channels'],
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
    associate_keypoint_head=[
        dict(
            type='TopdownHeatmapSimpleHead',
            in_channels=768,
            num_deconv_layers=2,
            num_deconv_filters=(256, 256),
            num_deconv_kernels=(4, 4),
            extra=dict(final_conv_kernel=1),
            out_channels=aic_channel_cfg['num_output_channels'],
            loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
        dict(
            type='TopdownHeatmapSimpleHead',
            in_channels=768,
            num_deconv_layers=2,
            num_deconv_filters=(256, 256),
            num_deconv_kernels=(4, 4),
            extra=dict(final_conv_kernel=1),
            out_channels=mpii_channel_cfg['num_output_channels'],
            loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
        dict(
            type='TopdownHeatmapSimpleHead',
            in_channels=768,
            num_deconv_layers=2,
            num_deconv_filters=(256, 256),
            num_deconv_kernels=(4, 4),
            extra=dict(final_conv_kernel=1),
            out_channels=ap10k_channel_cfg['num_output_channels'],
            loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
        dict(
            type='TopdownHeatmapSimpleHead',
            in_channels=768,
            num_deconv_layers=2,
            num_deconv_filters=(256, 256),
            num_deconv_kernels=(4, 4),
            extra=dict(final_conv_kernel=1),
            out_channels=ap10k_channel_cfg['num_output_channels'],
            loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
        dict(
            type='TopdownHeatmapSimpleHead',
            in_channels=768,
            num_deconv_layers=2,
            num_deconv_filters=(256, 256),
            num_deconv_kernels=(4, 4),
            extra=dict(final_conv_kernel=1),
            out_channels=cocowholebody_channel_cfg['num_output_channels'],
            loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
    ],
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=True,
        post_process='default',
        shift_heatmap=False,
        target_type=target_type,
        modulate_kernel=11,
        use_udp=True))

# OpenCap inference: whole-body stream (dataset_idx 5 in ViTPose++ training recipe).
data_cfg = dict(
    image_size=[192, 256],
    heatmap_size=[48, 64],
    num_output_channels=cocowholebody_channel_cfg['num_output_channels'],
    num_joints=cocowholebody_channel_cfg['dataset_joints'],
    dataset_channel=cocowholebody_channel_cfg['dataset_channel'],
    inference_channel=cocowholebody_channel_cfg['inference_channel'],
    soft_nms=False,
    nms_thr=1.0,
    oks_thr=0.9,
    vis_thr=0.2,
    use_gt_bbox=False,
    det_bbox_thr=0.0,
    bbox_file='',
    max_num_joints=133,
    dataset_idx=5,
)

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownAffine', use_udp=True),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'image_file', 'center', 'scale', 'rotation', 'bbox_score',
            'flip_pairs', 'dataset_idx',
        ]),
]

test_pipeline = val_pipeline

data = dict(
    test=dict(
        type='TopDownCocoWholeBodyDataset',
        ann_file='data/coco/annotations/coco_wholebody_val_v1.0.json',
        img_prefix='data/coco/val2017/',
        data_cfg=data_cfg,
        pipeline=test_pipeline))
