# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear, bias_init_with_prob
from mmcv.utils import TORCH_VERSION, digit_version

from mmdet.core import (multi_apply, multi_apply, reduce_mean)
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models import HEADS
from mmdet.models.dense_heads import DETRHead
from mmdet3d.core.bbox.coders import build_bbox_coder
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmcv.runner import force_fp32, auto_fp16
from mmdet3d.models.builder import build_loss, build_head
from projects.mmdet3d_plugin.models.utils.bricks import run_time
import numpy as np
import mmcv
import cv2 as cv
from projects.mmdet3d_plugin.models.utils.visual import save_tensor

import h5py
import os
import copy

@HEADS.register_module()
class VoxelFormerOccupancyHead(DETRHead):
    """Head of Detr3D.
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
        bev_h, bev_w (int): spatial shape of BEV queries.
    """

    def __init__(self,
                 *args,
                 with_box_refine=True,
                 as_two_stage=False,
                 transformer=None,
                 bbox_coder=None,
                 num_cls_fcs=2,
                 code_weights=None,
                 bev_h=120,
                 bev_w=120,
                 bev_z=4,
                 num_layout_query=10,
                 getbev=None,
                 occupancy_size=[0.1, 0.1, 0.1],
                 point_cloud_range=[-6.0, -6.0, -1.5, 6.0, 6.0, 2.0],
                 loss_layout=None,
                 loss_occupancy=None,
                 loss_flow=None,
                 flow_gt_dimension=2,
                 occ_dims=16,
                 det_dims=None,
                 num_occ_fcs=2,
                 occupancy_classes=1,
                 only_occ=False,
                 only_det=False,
                 add_layout=False,
                 with_occupancy_flow=False,
                 with_color_render=False,
                 occ_weights=None,
                 flow_weights=None,
                 occ_loss_type='focal_loss',
                 occ_head_type='mlp',
                 occ_head_network=None,
                 refine_occ=False,
                 **kwargs):

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.bev_z = bev_z
        self.fp16_enabled = False
        self.only_occ = only_occ
        self.only_det = only_det
        self.add_layout = add_layout
        self.occ_loss_type=occ_loss_type
        self.det_dims = None
        self.refine_occ = refine_occ
        self.num_layout_query = num_layout_query
        self.layout_range=[-50.0, -50.0, -5.0, 50.0, 50.0, 5.0]
        self.layout_coder = build_bbox_coder(dict(
            type='LayoutCoder',
            # post_center_range=[-100, -100, -10.0, 100, 100, 10.0],
            post_center_range=[-50, -50, -5.0, 50, 50, 5.0],
            pc_range=self.layout_range,
            # max_num=300,
            max_num=10,
            num_classes=1))
        # if self.det_dims is None:  # TODO 
        #     self.det_dims = kwargs.get('in_channels', 768)

        self.occ_head_type=occ_head_type

        self.getbev = getbev

        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            # self.code_weights = [1.0, 1.0, 1.0,
            #                      1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]
            # self.code_weights = [1.0, 1.0, 1.0,
            #                      1.0, 1.0, 1.0, 1.0, 1.0]
            self.code_weights = [1.0, 1.0, 1.0,
                                 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]

        if occ_weights is not None:
            self.occ_weights = occ_weights
        else:
            self.occ_weights = None
        
        if flow_weights is not None:
            self.flow_weights = flow_weights
        else:
            self.flow_weight = None
        self.bbox_coder = build_bbox_coder(bbox_coder)
        
        self.pc_range = self.bbox_coder.pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.real_z = self.pc_range[5] - self.pc_range[2]
        self.num_cls_fcs = num_cls_fcs - 1
        
        self.occupancy_size = occupancy_size
        self.point_cloud_range = point_cloud_range
        self.occ_xdim = int((point_cloud_range[3] - point_cloud_range[0]) / occupancy_size[0])
        self.occ_ydim = int((point_cloud_range[4] - point_cloud_range[1]) / occupancy_size[1])
        self.occ_zdim = int((point_cloud_range[5] - point_cloud_range[2]) / occupancy_size[2])
        self.occ_dims = occ_dims
        self.num_occ_fcs = num_occ_fcs
        self.occupancy_classes = occupancy_classes
        self.with_occupancy_flow = with_occupancy_flow
        self.with_color_render = with_color_render
        self.voxel_num = self.occ_xdim*self.occ_ydim*self.occ_zdim
        self.bev_num = self.bev_h*self.bev_w*self.bev_z

        self.flow_gt_dimension = flow_gt_dimension  # 2D for nuscenes
        self.loss_flow = loss_flow
        if self.only_occ:
            transformer.decoder = None   # define 3D detection decoder
        super(VoxelFormerOccupancyHead, self).__init__(
            *args, transformer=transformer, **kwargs)
        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights, requires_grad=False), requires_grad=False)

        if self.occ_head_type == 'cnn':
            self.occ_seg_head = build_head(occ_head_network)

        self.loss_occupancy = build_loss(loss_occupancy)
        if self.loss_flow is not None:
            self.loss_flow = build_loss(loss_flow)
            self.predict_flow = True
        else:
            self.predict_flow = False
        if self.only_occ:
            self.loss_cls = None
            self.loss_bbox = None
        if self.add_layout:
            self.loss_layout = build_loss(loss_layout)
        self._training_iter_ = 0 # DEBUG only

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        if self.transformer.decoder is not None:
            cls_branch = []
            for _ in range(self.num_reg_fcs):
                cls_branch.append(Linear(self.embed_dims, self.embed_dims))
                cls_branch.append(nn.LayerNorm(self.embed_dims))
                cls_branch.append(nn.ReLU(inplace=True))
            cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
            fc_cls = nn.Sequential(*cls_branch)

            reg_branch = []
            for _ in range(self.num_reg_fcs):
                reg_branch.append(Linear(self.embed_dims, self.embed_dims))
                reg_branch.append(nn.ReLU())
            reg_branch.append(Linear(self.embed_dims, self.code_size))
            reg_branch = nn.Sequential(*reg_branch)

            layout_branch = []
            for _ in range(self.num_reg_fcs):
                layout_branch.append(Linear(self.embed_dims, self.embed_dims))
                layout_branch.append(nn.ReLU())
            layout_branch.append(Linear(self.embed_dims, self.code_size))
            layout_branch = nn.Sequential(*layout_branch)

            def _get_clones(module, N):
                return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

            # last reg_branch is used to generate proposal from
            # encode feature map when as_two_stage is True.
            num_pred = (self.transformer.decoder.num_layers + 1) if \
                self.as_two_stage else self.transformer.decoder.num_layers

            if self.with_box_refine:
                self.cls_branches = _get_clones(fc_cls, num_pred)
                self.reg_branches = _get_clones(reg_branch, num_pred)
                self.layout_branches = _get_clones(layout_branch, num_pred)
            else:
                self.cls_branches = nn.ModuleList(
                    [fc_cls for _ in range(num_pred)])
                self.reg_branches = nn.ModuleList(
                    [reg_branch for _ in range(num_pred)])
                self.layout_branches = nn.ModuleList(
                    [layout_branch for _ in range(num_pred)])

            if not self.as_two_stage:
                self.voxel_embedding = nn.Embedding(
                    self.bev_num, self.embed_dims)
                self.query_embedding = nn.Embedding(self.num_query,
                                                    self.embed_dims * 2)
                self.query_layout_embedding = nn.Embedding(self.num_layout_query,
                                                    self.embed_dims * 2)
                
        else:
            self.voxel_embedding = nn.Embedding(self.bev_num, self.embed_dims)

        if self.bev_z == self.occ_zdim:
            self.occ_proj = Linear(self.embed_dims, self.occ_dims)  # the voxel size is consistent with the occupancy size
        else:
            self.occ_proj = Linear(self.bev_z*self.embed_dims, self.occ_dims * self.occ_zdim)

        # occupancy branch
        occ_branch = []
        for _ in range(self.num_occ_fcs):
            occ_branch.append(Linear(self.occ_dims, self.occ_dims))
            occ_branch.append(nn.LayerNorm(self.occ_dims))
            occ_branch.append(nn.ReLU(inplace=True))
        occ_branch.append(Linear(self.occ_dims, self.occupancy_classes))
        self.occ_branches = nn.Sequential(*occ_branch)

        # upsampling branch
        if self.refine_occ:
            # self.up_sample = nn.Upsample(size=(self.occ_zdim, self.occ_ydim, self.occ_xdim), mode='trilinear', align_corners=True)
            # self.up_sample = nn.Upsample(size=(self.bev_z, self.occ_ydim, self.occ_xdim), mode='trilinear', align_corners=True)
            self.up_sample = nn.Sequential(
                nn.ConvTranspose3d(768, 768, (3, 5, 5), stride=(1, 2, 2), padding=(2, 4, 4), dilation=(2, 2, 2),output_padding=(0,1,1)),
                nn.ConvTranspose3d(768, 768, (3, 5, 5), stride=(1, 2, 2), padding=(2, 4, 4), dilation=(2, 2, 2),output_padding=(0,1,1)),
                nn.ConvTranspose3d(768, 768, (3, 5, 5), stride=(1, 2, 2), padding=(2, 4, 4), dilation=(2, 2, 2),output_padding=(0,1,1))
            )

        if self.with_color_render:
            color_branch = []
            for _ in range(self.num_occ_fcs):
                color_branch.append(Linear(self.occ_dims, self.occ_dims))
                color_branch.append(nn.LayerNorm(self.occ_dims))
                color_branch.append(nn.ReLU(inplace=True))
            color_branch.append(Linear(self.occ_dims, 3))
            self.color_branches = nn.Sequential(*color_branch)

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        if self.loss_cls is not None:
            if self.loss_cls.use_sigmoid:
                bias_init = bias_init_with_prob(0.01)
                for m in self.cls_branches:
                    nn.init.constant_(m[-1].bias, bias_init)
        if self.loss_occupancy.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            nn.init.constant_(self.occ_branches[-1].bias, bias_init)

    @auto_fp16(apply_to=('mlvl_feats'))
    def forward(self, mlvl_feats, img_metas, prev_bev=None,  only_bev=False):
    # def forward(self, mlvl_feats, img_metas, prev_bev=None,  only_bev=True):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            prev_bev: previous bev featues
            only_bev: only compute BEV features with encoder. 
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """
        
        num_cam, bs, _, _ = mlvl_feats.shape # (6, 1, 196, 768)
        dtype = mlvl_feats.dtype
        if self.transformer.decoder is not None:  # 3D detectio query
            object_query_embeds = self.query_embedding.weight.to(dtype)
        voxel_queries = self.voxel_embedding.weight.to(dtype)
        # import ipdb;ipdb.set_trace()
        voxel_mask = torch.zeros((bs, self.bev_z, self.bev_h, self.bev_w),
                               device=voxel_queries.device).to(dtype)
        voxel_pos = self.positional_encoding(voxel_mask).to(dtype)
        
        if only_bev:  # only use encoder to obtain BEV features, TODO: refine the workaround
            return self.transformer.get_voxel_features(
                mlvl_feats,
                voxel_queries,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=voxel_pos,
                img_metas=img_metas,
                prev_bev=prev_bev,
            )
        
        elif self.only_occ:
            bev_embed = self.transformer.get_voxel_features(
                mlvl_feats,
                voxel_queries,
                self.bev_z,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=voxel_pos,
                img_metas=img_metas,
                prev_bev=prev_bev,
            )
            # bev_embed: (bs, num_query, embed_dims)
            bev_for_occ = bev_embed
            seq_len = 1
            
            bev_for_occ = bev_for_occ.view(bs*seq_len, self.bev_z, self.bev_h, self.bev_w, self.embed_dims)
            if self.bev_z == self.occ_zdim:
                occ_pred = self.occ_proj(bev_for_occ)  # (bs*seq_len, bev_z, bev_h, bev_w, self.occ_dims)
            else:
                bev_for_occ = bev_for_occ.permute(0, 2, 3, 1, 4).flatten(3)  # (bs*seq_len, bev_h, bev_w, -1)
                occ_pred = self.occ_proj(bev_for_occ)
                occ_pred = occ_pred.view(bs*seq_len, self.bev_h, self.bev_w, self.occ_zdim, self.occ_dims)
                occ_pred = occ_pred.permute(0, 3, 1, 2, 4)
            occ_pred = occ_pred.reshape(bs*seq_len, self.occ_zdim, -1, self.occ_dims) # bs*seq_len, z, y*x, dim


            occ_pred = occ_pred.reshape(bs * seq_len, -1, self.occ_dims)
            outputs_occupancy = self.occ_branches(occ_pred)

            outputs_flow = None
                    

            # bev_embed = bev_embed.permute(1, 0, 2)  # (num_query, bs, embed_dims)
            outs = {
                'bev_embed': bev_embed,
                'all_cls_scores': None,
                'all_bbox_preds': None,
                'all_layout_preds': None,
                'occupancy_preds': outputs_occupancy,
                'flow_preds': outputs_flow,
                'enc_cls_scores': None,
                'enc_bbox_preds': None,
                'enc_occupancy_preds': None
            }

            return outs

        elif self.only_det:
            outputs = self.transformer(
                mlvl_feats,
                voxel_queries,
                object_query_embeds,
                self.bev_z,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=voxel_pos,
                reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
                cls_branches=self.cls_branches if self.as_two_stage else None,
                img_metas=img_metas,
                prev_bev=prev_bev
                )
            bev_embed, hs, init_reference, inter_references = outputs

            hs = hs.permute(0, 2, 1, 3)
            outputs_classes = []
            outputs_coords = []
            for lvl in range(hs.shape[0]):
                if lvl == 0:
                    reference = init_reference
                else:
                    reference = inter_references[lvl - 1]
                reference = inverse_sigmoid(reference)
                outputs_class = self.cls_branches[lvl](hs[lvl])
                tmp = self.reg_branches[lvl](hs[lvl])

                # TODO: check the shape of reference
                assert reference.shape[-1] == 3
                tmp[..., 0:2] += reference[..., 0:2]
                tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
                tmp[..., 4:5] += reference[..., 2:3]
                tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
                tmp[..., 0:1] = (tmp[..., 0:1] * (self.pc_range[3] -
                                self.pc_range[0]) + self.pc_range[0])
                tmp[..., 1:2] = (tmp[..., 1:2] * (self.pc_range[4] -
                                self.pc_range[1]) + self.pc_range[1])
                tmp[..., 4:5] = (tmp[..., 4:5] * (self.pc_range[5] -
                                self.pc_range[2]) + self.pc_range[2])

                # TODO: check if using sigmoid
                outputs_coord = tmp
                outputs_classes.append(outputs_class)
                outputs_coords.append(outputs_coord)

            outputs_classes = torch.stack(outputs_classes)
            outputs_coords = torch.stack(outputs_coords)

            outs = {
                'bev_embed': bev_embed,
                'all_cls_scores': outputs_classes,
                'all_bbox_preds': outputs_coords,
                'all_layout_preds': None,
                'occupancy_preds': None,
                'flow_preds': None,
                'enc_cls_scores': None,
                'enc_bbox_preds': None,
                'enc_occupancy_preds': None
            }
            return outs

        elif self.add_layout:
            outputs = self.transformer(
                mlvl_feats,
                voxel_queries,
                object_query_embeds,
                self.bev_z,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=voxel_pos,
                reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
                cls_branches=self.cls_branches if self.as_two_stage else None,
                img_metas=img_metas,
                prev_bev=prev_bev
            )

            bev_embed, hs, init_reference, inter_references = outputs
            # bev_embed -> [100,1,768]; hs -> [6, 100, 1, 768];
            # init_reference -> [1, 100, 3]; inter_references -> [6, 1,100,3]

            bev_for_occ = bev_embed
            seq_len = 1
            
            bev_for_occ = bev_for_occ.view(bs*seq_len, self.bev_z, self.bev_h, self.bev_w, self.embed_dims)
            if self.bev_z == self.occ_zdim:
                occ_pred = self.occ_proj(bev_for_occ)  # (bs*seq_len, bev_z, bev_h, bev_w, self.occ_dims)
            else:
                bev_for_occ = bev_for_occ.permute(0, 2, 3, 1, 4).flatten(3)  # (bs*seq_len, bev_h, bev_w, -1)
                occ_pred = self.occ_proj(bev_for_occ)
                occ_pred = occ_pred.view(bs*seq_len, self.bev_h, self.bev_w, self.occ_zdim, self.occ_dims)
                occ_pred = occ_pred.permute(0, 3, 1, 2, 4)
            occ_pred = occ_pred.reshape(bs*seq_len, self.occ_zdim, -1, self.occ_dims) # bs*seq_len, z, y*x, dim

            occ_pred = occ_pred.reshape(bs * seq_len, -1, self.occ_dims)
            outputs_occupancy = self.occ_branches(occ_pred)
            
            outputs_flow = None

            hs = hs.permute(0, 2, 1, 3) # [6, 1, 100, 768]
            outputs_classes = []
            outputs_coords = []
            outputs_layouts = []
            for lvl in range(hs.shape[0]):
                if lvl == 0:
                    reference = init_reference
                else:
                    reference = inter_references[lvl - 1]
                reference = inverse_sigmoid(reference)
                outputs_class = self.cls_branches[lvl](hs[lvl])
                tmp = self.reg_branches[lvl](hs[lvl])

                assert reference.shape[-1] == 3
                tmp[..., 0:2] += reference[..., 0:2]
                tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
                tmp[..., 4:5] += reference[..., 2:3]
                tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
                tmp[..., 0:1] = (tmp[..., 0:1] * (self.pc_range[3] -
                                self.pc_range[0]) + self.pc_range[0])
                tmp[..., 1:2] = (tmp[..., 1:2] * (self.pc_range[4] -
                                self.pc_range[1]) + self.pc_range[1])
                tmp[..., 4:5] = (tmp[..., 4:5] * (self.pc_range[5] -
                                self.pc_range[2]) + self.pc_range[2])
                
                tmp_layout = self.layout_branches[lvl](hs[lvl])
                assert reference.shape[-1] == 3
                tmp_layout[..., 0:2] += reference[..., 0:2]
                tmp_layout[..., 0:2] = tmp_layout[..., 0:2].sigmoid()
                tmp_layout[..., 4:5] += reference[..., 2:3]
                tmp_layout[..., 4:5] = tmp_layout[..., 4:5].sigmoid()
                tmp_layout[..., 0:1] = (tmp_layout[..., 0:1] * (self.layout_range[3] -
                                self.layout_range[0]) + self.layout_range[0])
                tmp_layout[..., 1:2] = (tmp_layout[..., 1:2] * (self.layout_range[4] -
                                self.layout_range[1]) + self.layout_range[1])
                tmp_layout[..., 4:5] = (tmp_layout[..., 4:5] * (self.layout_range[5] -
                                self.layout_range[2]) + self.layout_range[2])

                outputs_coord = tmp
                outputs_layouts.append(tmp_layout)
                outputs_classes.append(outputs_class)
                outputs_coords.append(outputs_coord)

            outputs_classes = torch.stack(outputs_classes)
            outputs_coords = torch.stack(outputs_coords)
            outputs_layouts = torch.stack(outputs_layouts)

            outs = {
                'bev_embed': bev_embed,
                'all_cls_scores': outputs_classes,
                'all_bbox_preds': outputs_coords,
                'occupancy_preds': outputs_occupancy,
                'all_layout_preds': outputs_layouts,
                'flow_preds': outputs_flow,
                'enc_cls_scores': None,
                'enc_bbox_preds': None,
                'enc_occupancy_preds': None
            }

        else:
            outputs = self.transformer(
                mlvl_feats,
                voxel_queries,
                object_query_embeds,
                self.bev_z,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=voxel_pos,
                reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
                cls_branches=self.cls_branches if self.as_two_stage else None,
                img_metas=img_metas,
                prev_bev=prev_bev
            )

            bev_embed, hs, init_reference, inter_references = outputs
            # bev_embed -> [100,1,768]; hs -> [6, 100, 1, 768];
            # init_reference -> [1, 100, 3]; inter_references -> [6, 1,100,3]
            bev_for_occ = bev_embed
            seq_len = 1

            if self.refine_occ:
                bev_for_occ = bev_for_occ.view(bs*seq_len, self.embed_dims, self.bev_z, self.bev_h, self.bev_w)
                # import ipdb;ipdb.set_trace()
                bev_for_occ = self.up_sample(bev_for_occ)
                # 15 -> 30 -> 60 -> 120
                # 1, 768, 4, 15, 15 -> 1, 768, 4, 60, 60 -> 1, 768, 4, 120, 120
                
                bev_for_occ = bev_for_occ.view(bs*seq_len, self.bev_z, self.occ_xdim, self.occ_ydim, self.embed_dims)
            else:
                bev_for_occ = bev_for_occ.view(bs*seq_len, self.bev_z, self.bev_h, self.bev_w, self.embed_dims)
            if self.bev_z == self.occ_zdim:
                occ_pred = self.occ_proj(bev_for_occ)  # (bs*seq_len, bev_z, bev_h, bev_w, self.occ_dims)
            else:
                bev_for_occ = bev_for_occ.permute(0, 2, 3, 1, 4).flatten(3)  # (bs*seq_len, bev_h, bev_w, -1)
                occ_pred = self.occ_proj(bev_for_occ)
                if self.refine_occ:
                    occ_pred = occ_pred.view(bs*seq_len, self.occ_xdim, self.occ_ydim, self.occ_zdim, self.occ_dims)
                else:
                    occ_pred = occ_pred.view(bs*seq_len, self.bev_h, self.bev_w, self.occ_zdim, self.occ_dims)
                occ_pred = occ_pred.permute(0, 3, 1, 2, 4)
            occ_pred = occ_pred.reshape(bs*seq_len, self.occ_zdim, -1, self.occ_dims) # bs*seq_len, z, y*x, dim

            occ_pred = occ_pred.reshape(bs * seq_len, -1, self.occ_dims)
            outputs_occupancy = self.occ_branches(occ_pred)
            
            outputs_flow = None

            hs = hs.permute(0, 2, 1, 3) # [6, 1, 100, 768]
            outputs_classes = []
            outputs_coords = []
            for lvl in range(hs.shape[0]):
                if lvl == 0:
                    reference = init_reference
                else:
                    reference = inter_references[lvl - 1]
                reference = inverse_sigmoid(reference)
                outputs_class = self.cls_branches[lvl](hs[lvl])
                tmp = self.reg_branches[lvl](hs[lvl])

                assert reference.shape[-1] == 3
                tmp[..., 0:2] += reference[..., 0:2]
                tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
                tmp[..., 4:5] += reference[..., 2:3]
                tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
                tmp[..., 0:1] = (tmp[..., 0:1] * (self.pc_range[3] -
                                self.pc_range[0]) + self.pc_range[0])
                tmp[..., 1:2] = (tmp[..., 1:2] * (self.pc_range[4] -
                                self.pc_range[1]) + self.pc_range[1])
                tmp[..., 4:5] = (tmp[..., 4:5] * (self.pc_range[5] -
                                self.pc_range[2]) + self.pc_range[2])

                outputs_coord = tmp
                outputs_classes.append(outputs_class)
                outputs_coords.append(outputs_coord)

            outputs_classes = torch.stack(outputs_classes)
            outputs_coords = torch.stack(outputs_coords)

            outs = {
                'bev_embed': bev_embed,
                'all_cls_scores': outputs_classes,
                'all_bbox_preds': outputs_coords,
                'all_layout_preds': None,
                'occupancy_preds': outputs_occupancy,
                'flow_preds': outputs_flow,
                'enc_cls_scores': None,
                'enc_bbox_preds': None,
                'enc_occupancy_preds': None
            }

        if self.getbev is not None:
            if os.path.exists(self.getbev):
                outf = h5py.File(self.getbev, 'a')
            else:
                outf = h5py.File(self.getbev, 'w')
            # import ipdb;ipdb.set_trace()
            key = img_metas[0]['sample_idx']
            bev_embed = bev_embed.view(bs*seq_len, self.embed_dims, self.bev_z, self.bev_h, self.bev_w)
            savedata = bev_embed.squeeze().data.cpu().numpy()
            outf.create_dataset(key, savedata.shape, dtype='float', compression='gzip')
            outf[key][...] = savedata
            outf.close()

        return outs

    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """

        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        gt_c = gt_bboxes.shape[-1]
        if len(gt_bboxes.shape) == 1:
            gt_bboxes = gt_bboxes[None]
        
        # bbox_pred -> [100,10]; gt_bboxes -> [2,9]
        # cls_score -> [100,17]; gt_labels -> [2]
        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                             gt_labels, gt_bboxes_ignore)

        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes,),
                                    self.num_classes,
                                    dtype=torch.long)
        if len(gt_labels.shape) < 1:
            labels[pos_inds] = gt_labels
        else:
            labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)[..., :gt_c]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        # DETR
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        return (labels, label_weights, bbox_targets, bbox_weights,
                pos_inds, neg_inds)

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
            self._get_target_single, cls_scores_list, bbox_preds_list,
            gt_labels_list, gt_bboxes_list, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)

    def _get_target_layout_single(self,
                           cls_score,
                           bbox_pred,
                           layout_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_layout,
                           gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """

        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        gt_c = gt_bboxes.shape[-1]
        if len(gt_bboxes.shape) == 1:
            gt_bboxes = gt_bboxes[None]
        
        # bbox_pred -> [100,10]; gt_bboxes -> [2,9]
        # cls_score -> [100,17]; gt_labels -> [2]
        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                             gt_labels, gt_bboxes_ignore)
        # import ipdb;ipdb.set_trace()
        layout_assign_result = self.assigner.assign(layout_pred, torch.zeros(100,1).long().cuda(), gt_layout,
                                             torch.zeros(1).long().cuda(), None, layout=True)

        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        # import ipdb;ipdb.set_trace()
        sampling_layout_result = self.sampler.sample(layout_assign_result, layout_pred,
                                              gt_layout)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        layout_pos_inds = sampling_layout_result.pos_inds
        layout_neg_inds = sampling_layout_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes,),
                                    self.num_classes,
                                    dtype=torch.long)
        if len(gt_labels.shape) < 1:
            labels[pos_inds] = gt_labels
        else:
            labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)[..., :gt_c]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        #layout targets
        # bbox targets
        layout_targets = torch.zeros_like(layout_pred)[..., :gt_layout.shape[-1]]
        layout_weights = torch.zeros_like(layout_pred)
        layout_weights[layout_pos_inds] = 1.0
        # import ipdb;ipdb.set_trace()
        # DETR
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        layout_targets[layout_pos_inds] = sampling_layout_result.pos_gt_bboxes
        return (labels, label_weights, bbox_targets, bbox_weights, layout_targets, layout_weights,
                pos_inds, neg_inds, layout_pos_inds, layout_neg_inds)

    def get_targets_addlayout(self,
                    cls_scores_list,
                    bbox_preds_list,
                    layout_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_layout_list,
                    gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list, 
         bbox_weights_list, layout_targets_list, layout_weights_list, 
         pos_inds_list, neg_inds_list, layout_pos_inds_list, layout_neg_inds_list,) = multi_apply(
            self._get_target_layout_single, cls_scores_list, bbox_preds_list,
            layout_preds_list,
            gt_labels_list, gt_bboxes_list,
            gt_layout_list, gt_bboxes_ignore_list)
        # import ipdb;ipdb.set_trace()
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        num_total_layout_pos = sum((inds.numel() for inds in layout_pos_inds_list))
        num_total_layout_neg = sum((inds.numel() for inds in layout_neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, layout_targets_list, layout_weights_list, 
                num_total_pos, num_total_neg, num_total_layout_pos, num_total_layout_neg)

    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    occupancy_preds,
                    flow_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None,
                    gt_occupancy=None,
                    gt_flow=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)] #[(900,17)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)] #[(900,10)] or [(900,8)??]
        
        # gt_bboxes_list = [torch.stack(gt_bboxes_list).squeeze()]
        gt_labels_list = [gt_labels_list]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list,
                                           gt_bboxes_ignore_list)
        
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights
        # import ipdb;ipdb.set_trace()
        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :10], normalized_bbox_targets[isnotnan,
                                                               :10], bbox_weights[isnotnan, :10],
            avg_factor=num_total_pos)
        
        # loss occupancy
        if occupancy_preds is not None:
            num_pos_occ = torch.sum(gt_occupancy < self.occupancy_classes)
            occ_avg_factor = num_pos_occ * 1.0
            loss_occupancy = self.loss_occupancy(occupancy_preds, gt_occupancy, avg_factor=occ_avg_factor)
        else:
            loss_occupancy = torch.zeros_like(loss_cls)
        loss_flow = torch.zeros_like(loss_cls)
        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_cls = torch.nan_to_num(loss_cls)
            loss_bbox = torch.nan_to_num(loss_bbox)
            loss_occupancy = torch.nan_to_num(loss_occupancy)
            loss_flow = torch.nan_to_num(loss_flow)
        return loss_cls, loss_bbox, loss_occupancy, loss_flow

    def loss_single_layout(self,
                    cls_scores,
                    bbox_preds,
                    layout_preds,
                    occupancy_preds,
                    flow_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_layout_list,
                    gt_bboxes_ignore_list=None,
                    gt_occupancy=None,
                    gt_flow=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)] #[(900,17)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)] #[(900,10)] or [(900,8)??]
        layout_preds_list = [layout_preds[i] for i in range(num_imgs)]
        
        # gt_bboxes_list = [torch.stack(gt_bboxes_list).squeeze()]
        gt_labels_list = [gt_labels_list]
        cls_reg_targets = self.get_targets_addlayout(cls_scores_list, bbox_preds_list, layout_preds_list,
                                           gt_bboxes_list, gt_labels_list,
                                           gt_layout_list,
                                           gt_bboxes_ignore_list)
        
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         layout_targets_list, layout_weights_list, 
         num_total_pos, num_total_neg, num_total_layout_pos, num_total_layout_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        layout_targets = torch.cat(layout_targets_list, 0)
        layout_weights = torch.cat(layout_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        # import ipdb;ipdb.set_trace()
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        num_total_layout_pos = loss_cls.new_tensor([num_total_layout_pos])
        num_total_layout_pos = torch.clamp(reduce_mean(num_total_layout_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights
        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :10], normalized_bbox_targets[isnotnan,
                                                               :10], bbox_weights[isnotnan, :10],
            avg_factor=num_total_pos)
        # L1 for layout
        layout_preds = layout_preds.reshape(-1, layout_preds.size(-1))
        normalized_layout_targets = normalize_bbox(layout_targets, self.layout_range)
        # 1.4538, 1.1534, 0.5704, 9.2248, 6.6380, 4.2486
        # 1.4538, 1.1534, 2.2219, 1.8928, 0.5704, 1.4466
        isnotnan_layout = torch.isfinite(normalized_layout_targets).all(dim=-1)
        layout_weights = layout_weights * self.code_weights
        # import ipdb;ipdb.set_trace()
        loss_layout = self.loss_layout(
            layout_preds[isnotnan_layout, :10], normalized_layout_targets[isnotnan_layout,
                                                               :10], layout_weights[isnotnan_layout, :10],
            avg_factor=num_total_layout_pos)
        
        # loss occupancy
        if occupancy_preds is not None:
            num_pos_occ = torch.sum(gt_occupancy < self.occupancy_classes)
            occ_avg_factor = num_pos_occ * 1.0
            loss_occupancy = self.loss_occupancy(occupancy_preds, gt_occupancy, avg_factor=occ_avg_factor)
        else:
            loss_occupancy = torch.zeros_like(loss_cls)
        loss_flow = torch.zeros_like(loss_cls)
        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_cls = torch.nan_to_num(loss_cls)
            loss_bbox = torch.nan_to_num(loss_bbox)
            loss_layout = torch.nan_to_num(loss_layout)
            loss_occupancy = torch.nan_to_num(loss_occupancy)
            loss_flow = torch.nan_to_num(loss_flow)
        return loss_cls, loss_bbox, loss_layout, loss_occupancy, loss_flow

    @force_fp32(apply_to=('preds_dicts'))
    def loss_addlayout(self,
             gt_bboxes_list_org,
             gt_labels_list,
             gt_layout_list,
             point_coords,
             occ_gts,
             flow_gts,
             preds_dicts,
             gt_bboxes_ignore=None,
             img_metas=None):
        """"Loss function.
        Args:

            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_bbox_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds'] #[6,1,900,10]
        all_layout_preds = preds_dicts['all_layout_preds']
        # import ipdb;ipdb.set_trace()
        # all_bbox_preds = preds_dicts['all_bbox_preds'][...,:8]
        occupancy_preds = preds_dicts['occupancy_preds']  # (bs, occ_zdim*occ_ydim*occ_xdim, 11)
        flow_preds = preds_dicts['flow_preds']  # (bs, occ_zdim*occ_ydim*occ_xdim, 2)
        
        enc_cls_scores = preds_dicts['enc_cls_scores']
        enc_bbox_preds = preds_dicts['enc_bbox_preds']
        enc_occupancy_preds = preds_dicts['enc_occupancy_preds']
        # import ipdb;ipdb.set_trace()
        num_dec_layers = len(all_cls_scores)
        gt_labels_list = torch.from_numpy(gt_labels_list).cuda()
        device = gt_labels_list[0].device
        
        # gt_bboxes_list = []
        # for gtlen in range(len(gt_bboxes_list_org)):
        #     gt_bboxes_list.append(torch.cat((gt_bboxes_list_org[gtlen].gravity_center, 
        #     gt_bboxes_list_org[gtlen].tensor[:, 3:]), dim=1).to(device))
        gt_bboxes_list = []
        newgtbox = torch.cat(
            (gt_bboxes_list_org.gravity_center, gt_bboxes_list_org.tensor[:, 3:]),
            dim=1).to(device)
        newgtlayout = torch.cat(
            (gt_layout_list.gravity_center, gt_layout_list.tensor[:, 3:]),
            dim=1).to(device)
        pad_layout = torch.zeros((newgtlayout.shape[0], 2)).to(device)
        gt_layout_list = [torch.cat((newgtlayout, pad_layout), dim=1)]
        # pad velocity
        # import ipdb;ipdb.set_trace()
        pad_v = torch.zeros((newgtbox.shape[0], 2)).to(device)
        gt_bboxes_list = [torch.cat((newgtbox, pad_v), dim=1)]
        # newgtbox
        # gt_bboxes_list = [torch.cat(
        #     (gt_bboxes_list_org.gravity_center, gt_bboxes_list_org.tensor[:, 3:]),
        #     dim=1).to(device)]
        # import ipdb;ipdb.set_trace()
        
        bs, seq_len = 1, 1

        if occ_gts:
            gt_occupancy = (torch.ones((bs*seq_len, self.voxel_num), dtype=torch.long)*self.occupancy_classes).to(device)
            # gt_occupancy = (torch.ones((bs*seq_len, self.voxel_num), dtype=torch.long)).to(device)
            occ_gt = torch.tensor(occ_gts[0][0]).long().to(device)
            gt_occupancy[0*seq_len+0][[occ_gt[:, 0]]] = occ_gt[:, 1]
        gt_occupancy = gt_occupancy.view(-1) 
        occupancy_preds = occupancy_preds.view(-1, self.occupancy_classes)  
        
        all_gt_layout_list = [gt_layout_list for _ in range(num_dec_layers)]
        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        if flow_preds is None:
            flow_preds, gt_flow = None, None

        all_gt_occupancy_list = [None for _ in range(num_dec_layers - 1)] + [gt_occupancy]
        all_occupancy_preds = [None for _ in range(num_dec_layers - 1)] + [occupancy_preds]
        all_gt_flow_list = [None for _ in range(num_dec_layers - 1)] + [gt_flow]
        all_flow_preds = [None for _ in range(num_dec_layers - 1)] + [flow_preds]
        # import ipdb;ipdb.set_trace()
        losses_cls, losses_bbox, losses_layout, losses_occupancy, losses_flow = multi_apply(
            self.loss_single_layout, all_cls_scores, all_bbox_preds, all_layout_preds,
            all_occupancy_preds,
            all_flow_preds,
            all_gt_bboxes_list, all_gt_labels_list, all_gt_layout_list,
            all_gt_bboxes_ignore_list,
            all_gt_occupancy_list,
            all_gt_flow_list)

        loss_dict = dict()
        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            binary_labels_list = [
                torch.zeros_like(gt_labels_list[i])
                for i in range(len(all_gt_labels_list))
            ]
            enc_loss_cls, enc_losses_bbox, enc_losses_occupancy = \
                self.loss_single(enc_cls_scores, enc_bbox_preds, enc_occupancy_preds,
                                 gt_bboxes_list, binary_labels_list, gt_bboxes_ignore, gt_occupancy)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox
            loss_dict['enc_loss_occupancy'] = enc_losses_occupancy

        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        loss_dict['loss_occupancy'] = losses_occupancy[-1]
        loss_dict['loss_flow'] = losses_flow[-1]
        loss_dict['loss_layout'] = losses_layout[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1],
                                           losses_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            num_dec_layer += 1
        return loss_dict

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_bboxes_list_org,
             gt_labels_list,
             point_coords,
             occ_gts,
             flow_gts,
             preds_dicts,
             gt_bboxes_ignore=None,
             img_metas=None):
        """"Loss function.
        Args:

            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_bbox_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds'] #[6,1,900,10]
        # import ipdb;ipdb.set_trace()
        # all_bbox_preds = preds_dicts['all_bbox_preds'][...,:8]
        occupancy_preds = preds_dicts['occupancy_preds']  # (bs, occ_zdim*occ_ydim*occ_xdim, 11)
        flow_preds = preds_dicts['flow_preds']  # (bs, occ_zdim*occ_ydim*occ_xdim, 2)
        
        enc_cls_scores = preds_dicts['enc_cls_scores']
        enc_bbox_preds = preds_dicts['enc_bbox_preds']
        enc_occupancy_preds = preds_dicts['enc_occupancy_preds']
        # import ipdb;ipdb.set_trace()
        num_dec_layers = len(all_cls_scores)
        gt_labels_list = torch.from_numpy(gt_labels_list).cuda()
        device = gt_labels_list[0].device
        
        # gt_bboxes_list = []
        # for gtlen in range(len(gt_bboxes_list_org)):
        #     gt_bboxes_list.append(torch.cat((gt_bboxes_list_org[gtlen].gravity_center, 
        #     gt_bboxes_list_org[gtlen].tensor[:, 3:]), dim=1).to(device))
        gt_bboxes_list = []
        newgtbox = torch.cat(
            (gt_bboxes_list_org.gravity_center, gt_bboxes_list_org.tensor[:, 3:]),
            dim=1).to(device)
        # pad velocity
        # import ipdb;ipdb.set_trace()
        pad_v = torch.zeros((newgtbox.shape[0], 2)).to(device)
        gt_bboxes_list = [torch.cat((newgtbox, pad_v), dim=1)]
        # newgtbox
        # gt_bboxes_list = [torch.cat(
        #     (gt_bboxes_list_org.gravity_center, gt_bboxes_list_org.tensor[:, 3:]),
        #     dim=1).to(device)]
        # import ipdb;ipdb.set_trace()
        
        bs, seq_len = 1, 1

        if occ_gts:
            gt_occupancy = (torch.ones((bs*seq_len, self.voxel_num), dtype=torch.long)*self.occupancy_classes).to(device)
            # gt_occupancy = (torch.ones((bs*seq_len, self.voxel_num), dtype=torch.long)).to(device)
            occ_gt = torch.tensor(occ_gts[0][0]).long().to(device)
            gt_occupancy[0*seq_len+0][[occ_gt[:, 0]]] = occ_gt[:, 1]
        gt_occupancy = gt_occupancy.view(-1) 
        occupancy_preds = occupancy_preds.view(-1, self.occupancy_classes)  

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        if flow_preds is None:
            flow_preds, gt_flow = None, None

        all_gt_occupancy_list = [None for _ in range(num_dec_layers - 1)] + [gt_occupancy]
        all_occupancy_preds = [None for _ in range(num_dec_layers - 1)] + [occupancy_preds]
        all_gt_flow_list = [None for _ in range(num_dec_layers - 1)] + [gt_flow]
        all_flow_preds = [None for _ in range(num_dec_layers - 1)] + [flow_preds]
        
        losses_cls, losses_bbox, losses_occupancy, losses_flow = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds,
            all_occupancy_preds,
            all_flow_preds,
            all_gt_bboxes_list, all_gt_labels_list,
            all_gt_bboxes_ignore_list,
            all_gt_occupancy_list,
            all_gt_flow_list)

        loss_dict = dict()
        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            binary_labels_list = [
                torch.zeros_like(gt_labels_list[i])
                for i in range(len(all_gt_labels_list))
            ]
            enc_loss_cls, enc_losses_bbox, enc_losses_occupancy = \
                self.loss_single(enc_cls_scores, enc_bbox_preds, enc_occupancy_preds,
                                 gt_bboxes_list, binary_labels_list, gt_bboxes_ignore, gt_occupancy)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox
            loss_dict['enc_loss_occupancy'] = enc_losses_occupancy

        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        loss_dict['loss_occupancy'] = losses_occupancy[-1]
        loss_dict['loss_flow'] = losses_flow[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1],
                                           losses_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            num_dec_layer += 1
        return loss_dict

    @force_fp32(apply_to=('preds_dicts'))
    def loss_only_occupancy(self,
                        gt_bboxes_list,
                        gt_labels_list,
                        point_coords,
                        occ_gts,
                        flow_gts,
                        preds_dicts,
                        gt_bboxes_ignore=None,
                        img_metas=None):
        """"Loss function.
        """
        
        occupancy_preds = preds_dicts['occupancy_preds']  # (bs, occ_zdim*occ_ydim*occ_xdim, 16)
        flow_preds = preds_dicts['flow_preds']  # (bs, occ_zdim*occ_ydim*occ_xdim, 2)
       
        bs, seq_len = 1, 1


        device = gt_labels_list[0].device
        gt_occupancy = (torch.ones((bs*seq_len, self.voxel_num), dtype=torch.long)*self.occupancy_classes).to(device)

        occ_gt = occ_gts[0][0].long()
        gt_occupancy[0][[occ_gt[:, 0]]] = occ_gt[:, 1]

        gt_occupancy = gt_occupancy.view(-1)  # (bs*seq_len*occ_zdim*occ_ydim*occ_xdim)
        occupancy_preds = occupancy_preds.view(-1, self.occupancy_classes)  # (bs*seq_len*occ_zdim*occ_ydim*occ_xdim, 16)

        if flow_preds is None:
            flow_preds, gt_flow = None, None

        # add occ weights
        if self.occ_weights is not None:
            weights = torch.Tensor(self.occ_weights).to(gt_occupancy.device)
            occ_pred_weights = weights[gt_occupancy]
        else:
            occ_pred_weights = None
        num_pos_occ = torch.sum(gt_occupancy < self.occupancy_classes)
        occ_avg_factor = num_pos_occ * 1.0
        loss_occupancy = self.loss_occupancy(occupancy_preds, gt_occupancy, avg_factor=occ_avg_factor, weight=occ_pred_weights)
        
        if flow_preds is not None:
            object_mask = gt_occupancy < 10  # 0-9: object 10-15: background
            num_pos_flow = torch.sum(object_mask)
            flow_avg_factor = num_pos_flow * 1.0
            loss_flow = self.loss_flow(flow_preds[object_mask], gt_flow[object_mask], avg_factor=flow_avg_factor)
        else:
            loss_flow = torch.zeros_like(loss_occupancy)

        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_occupancy = torch.nan_to_num(loss_occupancy)
            loss_flow = torch.nan_to_num(loss_flow)

        loss_dict = dict()
        
        # loss from the last decoder layer
        loss_dict['loss_occupancy'] = loss_occupancy
        loss_dict['loss_flow'] = loss_flow
        return loss_dict




    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        
        preds_dicts = self.bbox_coder.decode(preds_dicts)

        num_samples = len(preds_dicts)
        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']

            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5

            code_size = bboxes.shape[-1]
            bboxes = img_metas[i]['box_type_3d'](bboxes, code_size)
            scores = preds['scores']
            labels = preds['labels']

            ret_list.append([bboxes, scores, labels])

        return ret_list
    
    @force_fp32(apply_to=('preds_dicts'))
    def get_layouts(self, preds_dicts, img_metas):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        
        preds_dicts = self.layout_coder.decode(preds_dicts)
        #TODO
        num_samples = len(preds_dicts)
        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            layouts = preds['layouts']

            layouts[:, 2] = layouts[:, 2] - layouts[:, 5] * 0.5

            code_size = layouts.shape[-1]
            layouts = img_metas[i]['box_type_3d'](layouts, code_size)

            ret_list.append([layouts])

        return ret_list

    def get_occupancy_prediction(self, occ_results, occ_threshold=0.25):
        """
        occupancy_preds: (Tensor): (bs, occ_zdim*occ_ydim*occ_xdim, occupancy_classes)
        flow_preds: (Tensor): (bs, occ_zdim*occ_ydim*occ_xdim, 2)
        """
        occupancy_preds = occ_results['occupancy_preds']
        flow_preds = occ_results['flow_preds']

        if self.occ_loss_type == 'focal_loss':
            occupancy_preds = occupancy_preds.reshape(-1, self.occupancy_classes)
            occupancy_preds = occupancy_preds.sigmoid()
            occupancy_preds = torch.cat((occupancy_preds, torch.ones_like(occupancy_preds)[:, :1] * occ_threshold), dim=-1)
            occ_class = occupancy_preds.argmax(dim=-1)
            occ_index, = torch.where(occ_class < self.occupancy_classes)
            occ_class = occ_class[occ_index[:]]
            occupancy_preds = torch.stack([occ_index, occ_class], dim=-1)

            # add flow preds
            if flow_preds is not None:
                flow_preds = flow_preds.reshape(-1, self.flow_gt_dimension)
                flow_preds = flow_preds[occ_index]
                
        elif self.occ_loss_type == 'ce_loss':
            occupancy_preds = occupancy_preds.reshape(-1, self.occupancy_classes)  
            occ_class = torch.argmax(occupancy_preds, axis=-1)
            occ_index, = torch.where(occ_class < self.occupancy_classes-1)
            occ_class = occ_class[occ_index[:]]
            occupancy_preds = torch.stack([occ_index, occ_class], dim=-1)
            flow_preds = None 
        
        else:
            raise NotImplementedError
            
        occ_results['occupancy_preds'] = occupancy_preds
        occ_results['flow_preds'] = flow_preds
        return occ_results
    

    def loss_single_for_detection(self,
                        cls_scores,
                        bbox_preds,
                        gt_bboxes_list,
                        gt_labels_list,
                        gt_bboxes_ignore_list=None,):
            """"Loss function for outputs from a single decoder layer of a single
            feature level.
            Args:
                cls_scores (Tensor): Box score logits from a single decoder layer
                    for all images. Shape [bs, num_query, cls_out_channels].
                bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                    for all images, with normalized coordinate (cx, cy, w, h) and
                    shape [bs, num_query, 4].
                gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                    with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
                gt_labels_list (list[Tensor]): Ground truth class indices for each
                    image with shape (num_gts, ).
                gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                    boxes which can be ignored for each image. Default None.
            Returns:
                dict[str, Tensor]: A dictionary of loss components for outputs from
                    a single decoder layer.
            """
            num_imgs = cls_scores.size(0)
            cls_scores_list = [cls_scores[i] for i in range(num_imgs)] #[(900,17)]
            bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)] #[(900,10)] or [(900,8)??]
            
            # gt_bboxes_list = [torch.stack(gt_bboxes_list).squeeze()]
            gt_labels_list = [gt_labels_list]
            cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                            gt_bboxes_list, gt_labels_list,
                                            gt_bboxes_ignore_list)
            
            (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
            num_total_pos, num_total_neg) = cls_reg_targets
            labels = torch.cat(labels_list, 0)
            label_weights = torch.cat(label_weights_list, 0)
            bbox_targets = torch.cat(bbox_targets_list, 0)
            bbox_weights = torch.cat(bbox_weights_list, 0)

            # classification loss
            cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
            # construct weighted avg_factor to match with the official DETR repo
            cls_avg_factor = num_total_pos * 1.0 + \
                num_total_neg * self.bg_cls_weight
            if self.sync_cls_avg_factor:
                cls_avg_factor = reduce_mean(
                    cls_scores.new_tensor([cls_avg_factor]))

            cls_avg_factor = max(cls_avg_factor, 1)
            loss_cls = self.loss_cls(
                cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

            # Compute the average number of gt boxes accross all gpus, for
            # normalization purposes
            num_total_pos = loss_cls.new_tensor([num_total_pos])
            num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

            # regression L1 loss
            bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
            normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
            isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
            bbox_weights = bbox_weights * self.code_weights
            # import ipdb;ipdb.set_trace()
            loss_bbox = self.loss_bbox(
                bbox_preds[isnotnan, :10], normalized_bbox_targets[isnotnan,
                                                                :10], bbox_weights[isnotnan, :10],
                avg_factor=num_total_pos)
            
            if digit_version(TORCH_VERSION) >= digit_version('1.8'):
                loss_cls = torch.nan_to_num(loss_cls)
                loss_bbox = torch.nan_to_num(loss_bbox)
            return loss_cls, loss_bbox

    @force_fp32(apply_to=('preds_dicts'))
    def loss_only_detection(self,
            gt_bboxes_list_org,
            gt_labels_list,
            preds_dicts,
            gt_bboxes_ignore=None,
            img_metas=None):
        """"Loss function.
        Args:

            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_bbox_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds'] #[6,1,900,10]     
        enc_cls_scores = preds_dicts['enc_cls_scores']
        enc_bbox_preds = preds_dicts['enc_bbox_preds']
        # import ipdb;ipdb.set_trace()
        num_dec_layers = len(all_cls_scores)
        gt_labels_list = torch.from_numpy(gt_labels_list).cuda()
        device = gt_labels_list[0].device
        
        # gt_bboxes_list = []
        # for gtlen in range(len(gt_bboxes_list_org)):
        #     gt_bboxes_list.append(torch.cat((gt_bboxes_list_org[gtlen].gravity_center, 
        #     gt_bboxes_list_org[gtlen].tensor[:, 3:]), dim=1).to(device))
        gt_bboxes_list = []
        newgtbox = torch.cat(
            (gt_bboxes_list_org.gravity_center, gt_bboxes_list_org.tensor[:, 3:]),
            dim=1).to(device)
        # pad velocity
        # import ipdb;ipdb.set_trace()
        pad_v = torch.zeros((newgtbox.shape[0], 2)).to(device)
        gt_bboxes_list = [torch.cat((newgtbox, pad_v), dim=1)]
        # newgtbox
        # gt_bboxes_list = [torch.cat(
        #     (gt_bboxes_list_org.gravity_center, gt_bboxes_list_org.tensor[:, 3:]),
        #     dim=1).to(device)]
        # import ipdb;ipdb.set_trace()


        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        
        losses_cls, losses_bbox, losses_occupancy, losses_flow = multi_apply(
            self.loss_single_for_detection, all_cls_scores, all_bbox_preds,
            all_gt_bboxes_list, all_gt_labels_list,
            all_gt_bboxes_ignore_list)

        loss_dict = dict()
        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            binary_labels_list = [
                torch.zeros_like(gt_labels_list[i])
                for i in range(len(all_gt_labels_list))
            ]
            enc_loss_cls, enc_losses_bbox, enc_losses_occupancy = \
                self.loss_single_for_detection(enc_cls_scores, enc_bbox_preds,
                                gt_bboxes_list, binary_labels_list, gt_bboxes_ignore)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox

        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1],
                                        losses_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            num_dec_layer += 1
        return loss_dict