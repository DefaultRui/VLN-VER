# Volumetric Environment Representation for Vision-Language Navigation

![](assets/overview.png)

> This repository is an official PyTorch implementation of paper:<br>
> [Volumetric Environment Representation for Vision-Language Navigation](https://arxiv.org/abs/2403.14158).<br>
> CVPR 2024. ([arXiv 2403.14158](https://arxiv.org/abs/2403.14158))


## Abstract
Vision-language navigation (VLN) requires an agent to navigate through an 3D environment based on visual observations and natural language instructions. It is clear that the pivotal factor for successful navigation lies in the comprehensive scene understanding. Previous VLN agents employ monocular frameworks to extract 2D features of perspective views directly. Though straightforward, they struggle for capturing 3D geometry and semantics, leading to a partial and incomplete environment representation. To achieve a comprehensive 3D representation with fine-grained details, we introduce a Volumetric Environment Representation (VER), which voxelizes the physical world into structured 3D cells. For each cell, VER aggregates multi-view 2D features into such a unified 3D space via 2D-3D sampling. Through coarse-to-fine feature extraction and multi-task learning for VER, our agent predicts 3D occupancy, 3D room layout, and 3D bounding boxes jointly. Based on online collected VERs, our agent performs volume state estimation and builds episodic memory for predicting the next step. Experimental results show our environment representations from multi-task learning lead to evident performance gains on VLN. Our model achieves state-of-the-art performance across VLN benchmarks (R2R, REVERIE, and R4R).

## Installation
The implementation is built on [MMDetection3D v0.17.1](https://github.com/open-mmlab/mmdetection3d), [MMSegmentation v0.14.1](https://github.com/open-mmlab/mmsegmentation), [MMDetection V2.14.0](https://github.com/open-mmlab/mmdetection). Please follow [BEVFormer](https://github.com/fundamentalvision/BEVFormer/blob/master/docs/install.md) for installation.

## Data Preparation
Please fill and sign the [Terms of Use](http://kaldir.vc.in.tum.de/matterport/MP_TOS.pdf) agreement form of [Matterport3D](https://niessner.github.io/Matterport/) and send it to matterport3d@googlegroups.com to request access to the dataset. "Undistorted_color_images" and "camera parameters" also need to be downloaded.

Please follow the [scripts](https://github.com/cshizhe/VLN-HAMT/tree/main/preprocess) to extract visual features for both undistorted_color_images. Note that all the ViT features ($14\times14\times768$) of undistorted_color_images should be used. We use [timm v0.6.7](https://github.com/huggingface/pytorch-image-models) for these features.

Please download object annotations and occupancy labels from the [link](https://pan.baidu.com/s/1jcllGZ4Cg79dYblsKhThgA) (code: dy2j), and put them into data files.

## VER Training
```shell
# multi-gpu train
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=${PORT:id} ./tools/dist_train.sh ./projects/configs/verformer/vocc.py 4

# multi-gpu test
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=${PORT:id} ./tools/dist_test.sh ./projects/configs/verformer/vocc.py ./path/to/ckpts.pth 4

# inference for volumetric representations
CUDA_VISIBLE_DEVICES=0 PORT=${PORT:id} ./tools/dist_test.sh ./projects/configs/verformer/get_occ.py ./path/to/ckpts.pth 1
```
Please also see train and inference for the detailed [usage](https://github.com/open-mmlab/mmdetection3d) of MMDetection3D.

## Citation
```bibtex
@inproceedings{liu2024volumetric,
  title={Volumetric Environment Representation for Vision-Language Navigation},
  author={Liu, Rui and Wang, Wenguan and Yang, Yi},
  booktitle={CVPR},
  pages={16317--16328},
  year={2024}
}
```

## Acknowledgement
Many thanks to the contributors for their excellent projects: [MMDetection3D](https://github.com/open-mmlab/mmdetection3d), [BEVFormer](https://github.com/fundamentalvision/BEVFormer/tree/master), [DUET](https://github.com/cshizhe/VLN-DUET), [OCCNet](https://github.com/OpenDriveLab/OccNet), [HAMT](https://github.com/cshizhe/VLN-HAMT), [VLN-DUET](https://github.com/cshizhe/VLN-DUET), [VLN-BEVBERT](https://github.com/MarSaKi/VLN-BEVBert), and [MP3D Simulator](https://github.com/peteanderson80/Matterport3DSimulator).

## Contact
This repository is currently maintained by [Rui](mailto:rui.liu@zju.edu.cn).