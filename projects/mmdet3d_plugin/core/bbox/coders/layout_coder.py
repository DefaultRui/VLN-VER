import torch

from mmdet.core.bbox import BaseBBoxCoder
from mmdet.core.bbox.builder import BBOX_CODERS
from projects.mmdet3d_plugin.core.bbox.util import denormalize_bbox
import numpy as np


@BBOX_CODERS.register_module()
class LayoutCoder(BaseBBoxCoder):
    """Bbox coder for NMS-free detector.
    Args:
        pc_range (list[float]): Range of point cloud.
        post_center_range (list[float]): Limit of the center.
            Default: None.
        max_num (int): Max number to be kept. Default: 100.
        score_threshold (float): Threshold to filter boxes based on score.
            Default: None.
        code_size (int): Code size of bboxes. Default: 9
    """

    def __init__(self,
                 pc_range,
                 voxel_size=None,
                 post_center_range=None,
                 max_num=100,
                 score_threshold=None,
                 num_classes=10):
        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.num_classes = num_classes

    def encode(self):

        pass

    def decode_single(self, layout_preds):
        """Decode bboxes.
        Args:
            cls_scores (Tensor): Outputs from the classification head, \
                shape [num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            bbox_preds (Tensor): Outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [num_query, 9].
        Returns:
            list[dict]: Decoded boxes.
        """
        max_num = self.max_num
        # import ipdb;ipdb.set_trace()
       
        final_layout_preds = denormalize_bbox(layout_preds, self.pc_range)

        if self.post_center_range is not None:
            self.post_center_range = torch.tensor(
                self.post_center_range).cuda()
            mask = (final_layout_preds[..., :3] >=
                    self.post_center_range[:3]).all(1)
            mask &= (final_layout_preds[..., :3] <=
                     self.post_center_range[3:]).all(1)

            boxes3d = final_layout_preds[mask]
            predictions_dict = {
                'layouts': boxes3d,
            }

        else:
            raise NotImplementedError(
                'Need to reorganize output as a batch, only '
                'support post_center_range is not None for now!')
        return predictions_dict
        # return boxes3d, scores, labels

    def decode(self, preds_dicts):
        """Decode bboxes.
        Args:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        Returns:
            list[dict]: Decoded boxes.
        """
        # all_cls_scores = preds_dicts['all_cls_scores'][-1] # [6, 1, 100, 17] -> [1, 100, 17] 6 layer, use last layer
        # all_bbox_preds = preds_dicts['all_bbox_preds'][-1] # [6, 1, 100, 10] -> [1, 100, 10]
        # nb_dec, batch_size, num_query, _ = preds_dicts['all_cls_scores'].shape
        # all_cls_scores = preds_dicts['all_cls_scores'] # [6, 1, 100, 17] 
        # all_bbox_preds = preds_dicts['all_bbox_preds'] # [6, 1, 100, 10] 
        
        # # batch_size = all_cls_scores.size()[0]
        # predictions_list = []
        
        # for i in range(batch_size):
        #     boxes3ds = []
        #     scores = []
        #     labels = []
        #     for j in range(nb_dec):
        #         boxes3d, score, label = self.decode_single(all_cls_scores[j][i], all_bbox_preds[j][i])
        #         boxes3ds.append(boxes3d)
        #         scores.append(score)
        #         labels.append(label)
        #     predictions_dict = {
        #         'bboxes': torch.stack(boxes3ds, 0).reshape(-1, 9),
        #         'scores': torch.stack(scores, 0).reshape(-1),
        #         'labels': torch.stack(labels, 0).reshape(-1),
        #     }
        #     # import ipdb;ipdb.set_trace()
        #     predictions_list.append(predictions_dict)
        # return predictions_list
        all_layout_preds = preds_dicts['all_layout_preds'][-1]
        
        batch_size = all_layout_preds.size()[0]
        predictions_list = []
        # import ipdb;ipdb.set_trace()
        for i in range(batch_size):
            predictions_list.append(self.decode_single(all_layout_preds[i]))
        return predictions_list

