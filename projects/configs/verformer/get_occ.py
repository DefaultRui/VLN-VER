_base_ = [
    '../datasets/custom_nus-3d.py',
    '../_base_/default_runtime.py'
]
#
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-6.0, -6.0, -1.5, 6.0, 6.0, 2.0]
bev_h_ = 15
bev_w_ = 15
bev_z_ = 4
voxel_size = [0.2, 0.2, 8]
occupancy_size = [0.1, 0.1, 0.1]
only_occ_ = False
only_det_ = False
refine_occ_ = True

querynum = 100
bbox_encode_max_num = 50


train_ann_filename = 'path to/obb_occ/mp3d_trainval.pkl'
val_ann_filename = 'path to/obb_occ/mp3d_test.pkl'
test_ann_filename = 'path to/obb_occ/forall.pkl'

savename='/data2/voxel_grid15_500_all.hdf5'#None

max_grad_norm = 300
total_epochs = 500
warmup_epoch = 30


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# For nuScenes we usually do 10-class detection
class_names = [
    'chair', 'door', 'table', 'picture', 'cabinet', 'cushion', 'window', 'sofa', 
    'bed', 'chest', 'plant', 'sink', 'toilet', 'monitor', 'lighting', 'shelving',
    'appliances'
]

occupancy_name = ['space', 'wall', 'floor', 'chair', 'door', 'table', 'objects', 
                'cabinet', 'window', 'sofa', 'bed', 'plant', 'sink', 'stairs',  
                'ceiling', 'shelving+railing']

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)

_dim_ = 768
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
_num_levels_ = 1
queue_length = 3 # each sequence contains `queue_length` frames.
_occupancy_dim_ = 128

model = dict(
    type='VoxelFormer',
    use_grid_mask=True,
    video_test_mode=True,
    use_occ_gts=True,#"True"
    only_occ=only_occ_,
    only_det=only_det_,
    pretrained=dict(img='torchvision://resnet50'),
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3,),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch'),
    img_neck=dict(
        type='FPN',
        in_channels=[2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=_num_levels_,
        relu_before_extra_convs=True),
    pts_bbox_head=dict(
        type='VoxelFormerOccupancyHead',
        bev_h=bev_h_,
        bev_w=bev_w_,
        bev_z=bev_z_,
        # num_query=900,
        getbev=savename,
        num_query=querynum,
        num_classes=17,
        in_channels=_dim_,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        point_cloud_range=point_cloud_range,
        occupancy_size=occupancy_size,
        occ_dims=_occupancy_dim_,
        occupancy_classes=16,
        only_occ=only_occ_,
        only_det=only_det_,
        refine_occ=refine_occ_,
        transformer=dict(
            type='VoxelPerceptionTransformer',
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=True,
            embed_dims=_dim_,
            decoder_on_bev=False,# use bev feature for decoding
            encoder=dict(
                type='VoxelFormerEncoder',
                num_layers=3,
                pc_range=point_cloud_range,
                num_points_in_voxel=4,
                return_intermediate=False,
                transformerlayers=dict(
                    type='VoxelFormerLayer',
                    attn_cfgs=[
                        dict(
                            type='SpatialCrossAttention',
                            pc_range=point_cloud_range,
                            deformable_attention=dict(
                                type='MSDeformableAttention3D',
                                embed_dims=_dim_,
                                num_points=8,
                                num_levels=_num_levels_),
                            embed_dims=_dim_,
                        )
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('cross_attn', 'norm',
                                     'ffn', 'norm'))),
            decoder=dict(
                type='VoxelDetectionTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1),
                         dict(
                            type='VoxelCustomMSDeformableAttention',
                            embed_dims=_dim_,
                            num_levels=1),
                    ],
                    ffn_cfgs=dict(
                     type='FFN',
                     embed_dims=768,
                     feedforward_channels=1024,
                     num_fcs=2,
                     ffn_drop=0.,
                     act_cfg=dict(type='ReLU', inplace=True)),

                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        bbox_coder=dict(
            type='NMSFreeCoder',
            # post_center_range=[-100, -100, -10.0, 100, 100, 10.0],
            post_center_range=[-10, -10, -5.0, 10, 10, 5.0],
            pc_range=point_cloud_range,
            # max_num=300,
            max_num=bbox_encode_max_num,
            voxel_size=voxel_size,
            num_classes=17),
        positional_encoding=dict(
            type='VoxelLearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
            z_num_embed=bev_z_,
            ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0),
        loss_occupancy=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0), # Fake cost. This is just to make it compatible with DETR head.
            pc_range=point_cloud_range))))

dataset_type = 'MP3DDataset'
data_root = 'path to /vitfeature'
file_client_args = dict(backend='disk')


train_pipeline = [
    dict(type='CustomMP3D')
]

test_pipeline = [
    dict(type='CustomMP3D')
]


data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=train_ann_filename,
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        bev_size=(bev_h_, bev_w_),
        pc_range=point_cloud_range,
        occ_size=occupancy_size,
        occ_names=occupancy_name,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=dict(type=dataset_type,
             data_root=data_root,
             ann_file=val_ann_filename,
             pipeline=test_pipeline, 
             bev_size=(bev_h_, bev_w_),
             pc_range=point_cloud_range,
             occ_size=occupancy_size,
             occ_names=occupancy_name,
             classes=class_names, modality=input_modality, samples_per_gpu=1),
    test=dict(type=dataset_type,
              data_root=data_root,
              ann_file=test_ann_filename,
              pipeline=test_pipeline,
              bev_size=(bev_h_, bev_w_),
              pc_range=point_cloud_range,
              occ_size=occupancy_size,
              occ_names=occupancy_name,
              classes=class_names, modality=input_modality),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

optimizer = dict(
    type='AdamW',
    lr=1e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)

optimizer_config = dict(grad_clip=dict(max_norm=max_grad_norm, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    # warmup_iters=500,
    # warmup_iters=20000,
    warmup_iters=warmup_epoch,
    warmup_by_epoch=True,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)

evaluation = dict(interval=20, pipeline=test_pipeline)

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

log_config = dict(
    interval=250,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

checkpoint_config = dict(interval=40)

