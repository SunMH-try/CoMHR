# CoMHR: Contrastive Multi-Modal Hypergraph Reasoning for 3D Crowd Mesh Recovery

The official code for the paper "Contrastive Multi-Modal Hypergraph Reasoning for 3D Crowd Mesh Recovery".<br>
\[[Paper](#)\]

![figure](/assets/pipeline.png)

## Installation
Create conda environment and install dependencies. The framework is tested on Python 3.10, PyTorch 2.1.2, and CUDA 11.8.
```bash
conda create -n comhr python=3.10
conda activate comhr
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Getting Started
**Step 1:**<br>
Download the official SMPL model from [SMPLify website](http://smplify.is.tuebingen.mpg.de/) and put it in `data/SMPL_NEUTRAL.pkl`.

**Step 2:**<br>
Download trained models from [Baidu Netdisk](https://pan.baidu.com/s/1ar3jVceyKqUiLuGoqGX1Ww?pwd=k3qd) (code: k3qd) and place them in the `data` directory.

**Step 3:**<br>
Download datasets and place them in `data/datasets`. The following datasets are supported:

| Dataset | Usage |
|---------|-------|
| [MPII](http://human-pose.mpi-inf.mpg.de/) | Training |
| [COCO](https://cocodataset.org/) | Training |
| [Panoptic](http://domedb.perception.cs.cmu.edu/) | Testing & Visualization |
| [GigaCrowd](https://gigavision.cn/) | Testing & Visualization |
| [CrowdPose](https://github.com/Jeff-sjtu/CrowdPose) | Visualization |

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
batchsize: 32
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

To address severe occlusions and depth ambiguity in dense crowds, we propose Contrastive Multi-modal Hypergraph Reasoning (CoMHR), which synergizes complementary modalities within a high-order topological structure:

- **Multi-Modal Node Initialization**: Augments RGB features with pseudo-depth maps and occlusion-aware 3D poses. We introduce a *Pelvis Depth Indicator* as a global spatial anchor to explicitly enforce front-back ordering.
- **Contrastive Hypergraph Construction**: Dynamically constructs a shared-topology hypergraph derived from holistic aggregated features rather than relying on predefined topologies.
- **Hypergraph Contrastive Learning**: A dual-branch strategy that enhances intra-modal discriminability (clustering individuals with similar actions) and enforces cross-modal orthogonality to maximize feature complementarity.
- **High-Order Reasoning**: Mitigates single-modal fragility by utilizing collective neighbor cues to infer missing information for occluded subjects across the shared topology.

## TODOS

- [x] Training code release
- [x] Pretrained model release
- [ ] Demo code release

## Citation
If you find this code useful for your research, please consider citing the paper.
```bibtex
@inproceedings{comhr,
  title={Contrastive Multi-Modal Hypergraph Reasoning for 3D Crowd Mesh Recovery},
  author={},
  booktitle={},
  year={},
}
```

## Acknowledgments
Our work builds heavily upon [GroupRec](https://github.com/boycehbz/GroupRec). We sincerely thank the authors for their outstanding contribution and open-source effort.<br>
Some other code is also based on the following works.<br>
[Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)<br>
[OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
