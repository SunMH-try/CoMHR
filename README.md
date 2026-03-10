# MultiModal-HMR
# Multimodal 3D Human Mesh Recovery with Relational Reasoning

The official code for the paper "Multimodal 3D Human Mesh Recovery with Relational Reasoning"<br>
[Buzhen Huang](http://www.buzhenhuang.com/)<br>
\[[Paper](#)\]

![figure](/assets/pipeline.jpg)

## Installation
Create conda environment and install dependencies.
```bash
conda create -n Multimodal python=3.9
conda activate Multimodal
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu111  # install pytorch
pip install -r requirements.txt
```

## Getting Started
**Step 1:**<br>
Download the official SMPL model from [SMPLify website](http://smplify.is.tuebingen.mpg.de/) and put it in `data/SMPL_NEUTRAL.pkl`.

**Step 2:**<br>
Download trained models and place them in the `data` directory.

**Step 3:**<br>
Download datasets and place them in `data/datasets`. The following datasets are supported:

| Dataset | Usage |
|---------|-------|
| [MPII](http://human-pose.mpi-inf.mpg.de/) | Training |
| [COCO](https://cocodataset.org/) | Training |
| [Panoptic](http://domedb.perception.cs.cmu.edu/) | Testing |
| [GigaCrowd](https://gigavision.cn/) | Testing |
| [Human3.6M](http://vision.imar.ro/human3.6m/) | Testing |

**Step 4:**<br>
Run training.
```bash
python main.py --config cfg_files/config.yaml --mode train
```

Run testing.
```bash
python main.py --config cfg_files/config.yaml --mode test
```

## Configuration

Key options in `cfg_files/config.yaml`:

```yaml
model: relation_multimodal   # relation_multimodal | relation_pose_rgb | relation_depth_rgb
trainset: MPII_CLIFF COCO
testset: Panoptic            # Panoptic | GigaCrowd
train_loss: SMPL_Loss Keyp_Loss Mesh_Loss Joint_Loss
batchsize: 8
epoch: 60
lr: 0.0001
use_sch: True                # cyclic learning rate scheduler
pretrain: False
viz: False
```

To use a pretrained model, set `pretrain: True` and update `pretrain_dir` with the path to your checkpoint.

## Models

| Model | Modalities | Description |
|-------|-----------|-------------|
| `relation_multimodal` | RGB + Depth + Pose | Full multimodal fusion model |
| `relation_pose_rgb` | RGB + Pose | RGB and pose fusion |
| `relation_depth_rgb` | RGB + Depth | RGB and depth fusion |

## Method Overview

This work proposes a multimodal framework for multi-person 3D human mesh recovery. The method fuses RGB images, depth maps, and 3D pose cues through:

- **Multi-Scale Hypergraph Neural Network (MS-HGNN)**: Models inter-person relational context at multiple scales via hypergraph message passing.
- **Multimodal PastEncoder**: Encodes relational features independently per modality before cross-modal fusion.
- **Cross-Modal Contrastive Learning**: Aligns representations across modalities to improve robustness and generalization.
- **Multi-Person Loss Supervision**: Combines SMPL parameter regression, 2D keypoint reprojection, 3D mesh, and joint losses.

## TODOS

- [x] Training code release
- [ ] Demo code release
- [ ] Pretrained model release

## Citation
If you find this code useful for your research, please consider citing the paper.
```
@inproceedings{multimodalhmr,
  title={Multimodal 3D Human Mesh Recovery with Relational Reasoning},
  author={Huang, Buzhen},
  booktitle={},
  year={2024},
}
```

## Acknowledgments
Some of the code is based on the following works. We gratefully appreciate the impact they have on our work.<br>
[GroupRec](https://github.com/boycehbz/GroupRec)<br>
[CLIFF](https://github.com/huawei-noah/noah-research/tree/master/CLIFF)<br>
[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)<br>
[PyMAF](https://github.com/HongwenZhang/PyMAF)<br>
