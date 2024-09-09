from mmdet.datasets import DATASETS
from mmdet3d.datasets import Custom3DDataset

import tempfile
import warnings
from os import path as osp
import os
import shutil

import mmcv
import numpy as np
from torch.utils.data import Dataset
from mmdet3d.core.bbox import LiDARInstance3DBoxes, get_box_type
from mmcv.parallel import DataContainer as DC
# from ..core.bbox import get_box_type
# from .builder import DATASETS
# from .pipelines import Compose
# from .utils import extract_result_dict, get_loading_pipeline
import sys
from mmdet.datasets.pipelines import Compose 
# sys.getrecursionlimit(10000)
# sys.setrecursionlimit(10000)
from .occupancy_metrics import SSCMetrics
from tqdm import tqdm

@DATASETS.register_module()
class MP3DDataset(Custom3DDataset):
    r"""MP3D Dataset.

    This datset is custom by Ray L.
    """
    def __init__(self,
                 data_root,
                 ann_file,
                 pipeline=None,
                 classes=None,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 bev_size=(120, 120),
                 pc_range=[-6.0, -6.0, -1.5, 6.0, 6.0, 2.0],
                 occ_size=[0.1, 0.1, 0.1],
                 occ_names=None,
                 **kwargs):
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,)
        self.data_root = data_root
        self.ann_file = ann_file
        self.test_mode = test_mode
        self.modality = modality

        self.occupancy_size = occ_size
        self.point_cloud_range = pc_range
        self.occupancy_names = occ_names
        self.occ_xdim = int((self.point_cloud_range[3] - self.point_cloud_range[0]) / self.occupancy_size[0])
        self.occ_ydim = int((self.point_cloud_range[4] - self.point_cloud_range[1]) / self.occupancy_size[1])
        self.occ_zdim = int((self.point_cloud_range[5] - self.point_cloud_range[2]) / self.occupancy_size[2])
        self.occupancy_classes = len(self.occupancy_names)
        self.voxel_num = self.occ_xdim*self.occ_ydim*self.occ_zdim

        self.bev_size = bev_size
        
        self.filter_empty_gt = filter_empty_gt
        self.box_type_3d, self.box_mode_3d = get_box_type(box_type_3d)
        # self.CLASSES = self.get_classes(classes)
        self.CLASSES = classes # no objects!
        # self.file_client = mmcv.FileClient(**file_client_args)
        self.cat2id = {name: i for i, name in enumerate(self.CLASSES)}
        self.occupancy_class_names = ['space', 'wall', 'floor', 'chair', 'door', 'table', 'objects', 
                'cabinet', 'window', 'sofa', 'bed', 'plant', 'sink', 'stairs',  
                'ceiling', 'shelving+railing']
        # load annotations
        self.data_infos = self.load_annotations(self.ann_file)
        # if hasattr(self.file_client, 'get_local_path'):
        #     with self.file_client.get_local_path(self.ann_file) as local_path:
        #         self.data_infos = self.load_annotations(open(local_path, 'rb'))
        # else:
        #     warnings.warn(
        #         'The used MMCV version does not have get_local_path. '
        #         f'We treat the {self.ann_file} as local paths and it '
        #         'might cause errors if the path is not a local path. '
        #         'Please use MMCV>= 1.3.16 if you meet errors.')
        #     self.data_infos = self.load_annotations(self.ann_file)

        # process pipeline
        if pipeline is not None:
            self.pipeline = Compose(pipeline)

        # set group flag for the samplers
        if not self.test_mode:
            self._set_group_flag()

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations.
        """
        # loading data from a file-like object needs file format
        return mmcv.load(ann_file, file_format='pkl')

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - file_name (str): Filename of point clouds.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        sample_idx = info['token']
        file_name = osp.join(self.data_root,
                                'new_vit_base_patch16_197_imagenet.hdf5')

        input_dict = dict(
            sample_idx=sample_idx,
            file_name=file_name)
        if 'occ_gt_path' in info:
            input_dict['occ_gt_path'] = info['occ_gt_path']

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos
            if self.filter_empty_gt and ~(annos['gt_labels_3d'] != -1).any():
                return None
        return input_dict

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`):
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        info = self.data_infos[index]
        gt_bboxes_3d = np.array(info['gt_boxes'])
        gt_names_3d = info['names']
        gt_labels_3d = info['labels']
        if 'layout' in info.keys():
            gt_layout_3d = np.array(info['layout'])
            gt_layout_3d = LiDARInstance3DBoxes(
                gt_layout_3d,
                box_dim=7,
                origin=(0.5, 0.5, 0)).convert_to(self.box_mode_3d)
        else:
            gt_layout_3d = None
        # for cat in gt_names_3d:
        #     if cat in self.CLASSES:
        #         gt_labels_3d.append(self.CLASSES.index(cat))
        #     else:
        #         gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        # turn original box type to target box type
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=7,
            origin=(0.5, 0.5, 0)).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
            gt_layout_3d=gt_layout_3d)
        return anns_results

    def pre_pipeline(self, results):
        """Initialization before data preparation.

        Args:
            results (dict): Dict before data preprocessing.

                - img_fields (list): Image fields.
                - bbox3d_fields (list): 3D bounding boxes fields.
                - pts_mask_fields (list): Mask fields of points.
                - pts_seg_fields (list): Mask fields of point segments.
                - bbox_fields (list): Fields of bounding boxes.
                - mask_fields (list): Fields of masks.
                - seg_fields (list): Segment fields.
                - box_type_3d (str): 3D box type.
                - box_mode_3d (str): 3D box mode.
        """
        results['img_fields'] = []
        results['bbox3d_fields'] = []
        results['pts_mask_fields'] = []
        results['pts_seg_fields'] = []
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []
        results['box_type_3d'] = self.box_type_3d
        results['box_mode_3d'] = self.box_mode_3d

    def prepare_train_data(self, index):
        """Training data preparation.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Training data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        
        # if self.filter_empty_gt and \
        #         (example is None or
        #             ~(example['ann_info']['gt_labels_3d']._data != -1).any()):
        #     return None
        return example

    def prepare_test_data(self, index):
        """Prepare data for testing.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Testing data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example

    # @classmethod
    # def get_classes(cls, classes=None):
    #     """Get class names of current dataset.

    #     Args:
    #         classes (Sequence[str] | str): If classes is None, use
    #             default CLASSES defined by builtin dataset. If classes is a
    #             string, take it as a file name. The file contains the name of
    #             classes where each line contains one class name. If classes is
    #             a tuple or list, override the CLASSES defined by the dataset.

    #     Return:
    #         list[str]: A list of class names.
    #     """
    #     if classes is None:
    #         return cls.CLASSES

    #     if isinstance(classes, str):
    #         # take it as a file path
    #         class_names = mmcv.list_from_file(classes)
    #     elif isinstance(classes, (tuple, list)):
    #         class_names = classes
    #     else:
    #         raise ValueError(f'Unsupported type {type(classes)} of classes.')

    #     return class_names

    def format_results(self,
                       outputs,
                       pklfile_prefix=None,
                       submission_prefix=None):
        """Format the results to pkl file.

        Args:
            outputs (list[dict]): Testing results of the dataset.
            pklfile_prefix (str): The prefix of pkl files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: (outputs, tmp_dir), outputs is the detection results,
                tmp_dir is the temporal directory created for saving json
                files when ``jsonfile_prefix`` is not specified.
        """
        if pklfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            pklfile_prefix = osp.join(tmp_dir.name, 'results')
            out = f'{pklfile_prefix}.pkl'
        mmcv.dump(outputs, out)
        return outputs, tmp_dir

    def evaluate(self,
                 results,
                 metric=None,
                #  iou_thr=(0.25, 0.5),
                 iou_thr=(0.10, 0.25, 0.5, 0.75),
                 jsonfile_prefix=None,
                 result_names=['pts_bbox'],
                 logger=None,
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluate.

        Evaluation in indoor protocol.

        Args:
            results (list[dict]): List of results.
            metric (str | list[str], optional): Metrics to be evaluated.
                Defaults to None.
            iou_thr (list[float]): AP IoU thresholds. Defaults to (0.25, 0.5).
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Defaults to None.
            show (bool, optional): Whether to visualize.
                Default: False.
            out_dir (str, optional): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict: Evaluation results.
        """
        # from mmdet3d.core.evaluation import indoor_eval
        from .indoor_eval import indoor_eval
        try:
            results = results['bbox_results']
        except:
            results = results
        try:
            assert isinstance(
                results, list), f'Expect results to be list, got {type(results)}.'
        except:
            print(results)
            import ipdb;ipdb.set_trace()
        assert len(results) > 0, 'Expect length of results > 0.'
        assert len(results) == len(self.data_infos)
        assert isinstance(
            results[0], dict
        ), f'Expect elements in results to be dict, got {type(results[0])}.'

        # gt_annos = [info['annos'] for info in self.data_infos]
        gt_annos = []
        for info in self.data_infos:
            tmp_dict = {}
            tmp_dict['gt_boxes_upright_depth'] = np.array(info['gt_boxes'])
            tmp_dict['gt_names_3d'] = info['names']
            tmp_dict['class'] = info['labels']
            tmp_dict['gt_num'] = 10
            gt_annos.append(tmp_dict)

        label2cat = {i: cat_id for i, cat_id in enumerate(self.CLASSES)}
        newresults = []
        for result in results:
            newresults.append(dict(
                boxes_3d = result['pts_bbox']['boxes_3d'],
                scores_3d = result['pts_bbox']['scores_3d'],
                labels_3d = result['pts_bbox']['labels_3d']
            ))
        # import ipdb;ipdb.set_trace()
        ret_dict = indoor_eval(
            gt_annos,
            newresults,
            iou_thr,
            label2cat,
            logger=logger,
            box_type_3d=self.box_type_3d,
            box_mode_3d=self.box_mode_3d)
        if show:
            self.show(results, out_dir, pipeline=pipeline)
        
        return ret_dict

    def _build_default_pipeline(self):
        """Build the default pipeline for this dataset."""
        raise NotImplementedError('_build_default_pipeline is not implemented '
                                  f'for dataset {self.__class__.__name__}')

    # def _get_pipeline(self, pipeline):
    #     """Get data loading pipeline in self.show/evaluate function.

    #     Args:
    #         pipeline (list[dict]): Input pipeline. If None is given,
    #             get from self.pipeline.
    #     """
    #     if pipeline is None:
    #         if not hasattr(self, 'pipeline') or self.pipeline is None:
    #             warnings.warn(
    #                 'Use default pipeline for data loading, this may cause '
    #                 'errors when data is on ceph')
    #             return self._build_default_pipeline()
    #         loading_pipeline = get_loading_pipeline(self.pipeline.transforms)
    #         return Compose(loading_pipeline)
    #     return Compose(pipeline)

    # def _extract_data(self, index, pipeline, key, load_annos=False):
    #     """Load data using input pipeline and extract data according to key.

    #     Args:
    #         index (int): Index for accessing the target data.
    #         pipeline (:obj:`Compose`): Composed data loading pipeline.
    #         key (str | list[str]): One single or a list of data key.
    #         load_annos (bool): Whether to load data annotations.
    #             If True, need to set self.test_mode as False before loading.

    #     Returns:
    #         np.ndarray | torch.Tensor | list[np.ndarray | torch.Tensor]:
    #             A single or a list of loaded data.
    #     """
    #     assert pipeline is not None, 'data loading pipeline is not provided'
    #     # when we want to load ground-truth via pipeline (e.g. bbox, seg mask)
    #     # we need to set self.test_mode as False so that we have 'annos'
    #     if load_annos:
    #         original_test_mode = self.test_mode
    #         self.test_mode = False
    #     input_dict = self.get_data_info(index)
    #     self.pre_pipeline(input_dict)
    #     example = pipeline(input_dict)

    #     # extract data items according to keys
    #     if isinstance(key, str):
    #         data = extract_result_dict(example, key)
    #     else:
    #         data = [extract_result_dict(example, k) for k in key]
    #     if load_annos:
    #         self.test_mode = original_test_mode

    #     return data

    def __len__(self):
        """Return the length of data infos.

        Returns:
            int: Length of data infos.
        """
        return len(self.data_infos)

    def _rand_another(self, idx):
        """Randomly get another item with the same flag.

        Returns:
            int: Another index of item with the same flag.
        """
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def __getitem__(self, idx):
        """Get item from infos according to the given index.

        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode:
            return self.prepare_test_data(idx)
        # else:
        #     return self.prepare_train_data(idx)
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0. In 3D datasets, they are all the same, thus are all
        zeros.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
    
    def evaluate_occ_iou(self, occupancy_results, flow_results, show_dir=None, 
                         save_interval=1, occ_threshold=0.25, runner=None):
        """ save the gt_occupancy_sparse and evaluate the iou metrics"""
        assert len(occupancy_results) == len(self)
        thre_str = 'thre_'+'{:.2f}'.format(occ_threshold)

        if show_dir: 
            scene_names = [info['scene_name'] for info in self.data_infos]
            scene_names = np.unique(scene_names)
            save_scene_names = scene_names[:5]
            print('save_scene_names:', save_scene_names)
        
        # set the metrics
        self.eval_metrics = SSCMetrics(self.occupancy_classes+1)
        # loop the dataset
        for index in tqdm(range(len(occupancy_results))):
            info = self.data_infos[index]
            scene_name, frame_idx = info['token'].split('_')
            
            # load occ_gt
            occ_gt_sparse = np.load(info['occ_gt_path'])
            occ_index = occ_gt_sparse[:, 0]
            occ_class = occ_gt_sparse[:, 1] 
            gt_occupancy = np.ones(self.voxel_num, dtype=np.uint8)*self.occupancy_classes
            gt_occupancy[occ_index] = occ_class  # (num_voxels)

            # load occ_invalid
            if 'occ_invalid_path' in info:
                occ_invalid_index = np.load(info['occ_invalid_path'])
                visible_mask = np.ones(self.voxel_num, dtype=np.uint8)
                visible_mask[occ_invalid_index] = 0
            else:
                visible_mask = None
            #  load occ_pred
            occ_pred_sparse = occupancy_results[index].cpu().numpy()
            pred_occupancy = self.get_voxel_prediction(occ_pred_sparse)

            # load flow info
            flow_pred, flow_true = None, None
            # if flow_results is not None:
            #     flow_pred_sparse = flow_results[index].cpu().numpy()
            #     flow_gt_sparse = np.load(info['flow_gt_path']) 
            #     flow_true, flow_pred = self.parse_flow_info(occ_gt_sparse, occ_pred_sparse, flow_gt_sparse, flow_pred_sparse)

            self.eval_metrics.add_batch(pred_occupancy, gt_occupancy, flow_pred=flow_pred, flow_true=flow_true, visible_mask=visible_mask)

            # save occ, flow, image
            if show_dir and index % save_interval == 0:
                save_result = False
                if scene_name in save_scene_names:
                    save_result = True
                if save_result:
                    occ_gt_save_dir = os.path.join(show_dir, thre_str, scene_name, 'occ_gts')
                    occ_pred_save_dir = os.path.join(show_dir, thre_str, scene_name, 'occ_preds')
                    image_save_dir = os.path.join(show_dir, thre_str, scene_name, 'images')
                    os.makedirs(occ_gt_save_dir, exist_ok=True)
                    os.makedirs(occ_pred_save_dir, exist_ok=True)
                    os.makedirs(image_save_dir, exist_ok=True)

                    np.save(osp.join(occ_gt_save_dir, '{:03d}_occ.npy'.format(frame_idx)), occ_gt_sparse)  
                    np.save(osp.join(occ_pred_save_dir, '{:03d}_occ.npy'.format(frame_idx)), occ_pred_sparse)  

                    # if flow_results is not None:
                    #     np.save(osp.join(occ_gt_save_dir, '{:03d}_flow.npy'.format(frame_idx)), flow_gt_sparse)  
                    #     np.save(osp.join(occ_pred_save_dir, '{:03d}_flow.npy'.format(frame_idx)), flow_pred_sparse)
                    image_save_path = osp.join(image_save_dir, '{:03d}.png'.format(frame_idx))
                    if 'surround_image_path' in info:
                        shutil.copyfile(info['surround_image_path'], image_save_path)

        eval_resuslt = self.eval_metrics.get_stats()
        print(f'======out evaluation metrics: {thre_str}=========')
        
        for i, class_name in enumerate(self.occupancy_class_names):
            print("miou/{}: {:.2f}".format(class_name, eval_resuslt["iou_ssc"][i]))
        print("miou: {:.2f}".format(eval_resuslt["miou"]))
        print("iou: {:.2f}".format(eval_resuslt["iou"]))
        print("Precision: {:.4f}".format(eval_resuslt["precision"]))
        print("Recall: {:.4f}".format(eval_resuslt["recall"]))
        
        # flow_distance = eval_resuslt['flow_distance']
        # flow_states=['flow_distance_sta', 'flow_distance_mov', 'flow_distance_all']
        # for i in range(len(flow_states)):
        #     if flow_distance[i] is not None:
        #         print("{}: {:.4f}".format(flow_states[i], flow_distance[i]))
        
        if runner is not None:
            for i, class_name in enumerate(self.occupancy_class_names):
                runner.log_buffer.output[class_name] =  eval_resuslt["iou_ssc"][i]
            runner.log_buffer.output['miou'] =  eval_resuslt["miou"]
            runner.log_buffer.output['iou'] =  eval_resuslt["iou"]
            runner.log_buffer.output['precision'] =  eval_resuslt["precision"]
            runner.log_buffer.output['recall'] =  eval_resuslt["recall"]
            runner.log_buffer.ready = True
    
    def get_voxel_prediction(self, occupancy):
        occ_index = occupancy[:, 0]
        occ_class = occupancy[:, 1]
        pred_occupancy = np.ones(self.voxel_num, dtype=np.uint8)*self.occupancy_classes
        pred_occupancy[occ_index] = occ_class  # (num_voxels)
        return pred_occupancy