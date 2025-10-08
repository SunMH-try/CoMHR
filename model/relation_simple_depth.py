
import torch
from torch import nn
from torch.nn import functional as F
from utils.imutils import cam_crop2full, vis_img
from collections import namedtuple
from utils.geometry import perspective_projection, rot6d_to_rotmat
from utils.rotation_conversions import *
import cv2
from model.backbones.resnet50 import ResNet50

args = namedtuple('args', [
    'hidden_dim',
    'hyper_scales',
    'learn_prior',
    'nmp_layers',
])

        
class relation_simple_depth(nn.Module):
    def __init__(self, smpl, num_joints=21):
        super().__init__()
        self.smpl = smpl
        self.num_joints = num_joints

        # depth backbone
        self.depth_encoder = ResNet50(input_c=1)
        self.depth_fc = nn.Linear(2048, 1024)

        # heads
        embed_dim = 1024
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim + 3),
            nn.Linear(embed_dim + 3, 24 * 6),
        )
        self.cam_head = nn.Sequential(
            nn.LayerNorm(embed_dim + 3),
            nn.Linear(embed_dim + 3, 3),
        )
        self.shape_head = nn.Sequential(
            nn.LayerNorm(embed_dim + 3),
            nn.Linear(embed_dim + 3, 10),
        )

    def set_device(self, device):
        self.device = device
        self.to(device)

    def forward(self, data):
        batch_size, agent_num, H, W = data['depth'].shape

        # reshape depth
        depth = data['depth'].reshape(batch_size * agent_num, H, W).unsqueeze(1)  # (B*N,1,H,W)
        depth_feat = self.depth_encoder(depth)  # (B*N,2048)
        depth_feat = self.depth_fc(depth_feat)  # (B*N,1024)

        # bbox 信息
        center = data['center'].reshape(batch_size * agent_num, -1)
        scale = data['scale'].reshape(batch_size * agent_num,)
        img_h = data['img_h'].reshape(batch_size * agent_num,)
        img_w = data['img_w'].reshape(batch_size * agent_num,)
        focal_length = data['focal_length'].reshape(batch_size * agent_num,)

        cx, cy, b = center[:, 0], center[:, 1], scale * 200
        bbox_info = torch.stack([cx - img_w / 2., cy - img_h / 2., b], dim=-1)
        bbox_info[:, :2] = bbox_info[:, :2] / focal_length.unsqueeze(-1) * 2.8
        bbox_info[:, 2] = (bbox_info[:, 2] - 0.24 * focal_length) / (0.06 * focal_length)

        # 只保留 valid
        valid = data['valid'].reshape(-1,)
        depth_feat = depth_feat[valid == 1]
        bbox_info = bbox_info[valid == 1]
        center = center[valid == 1]
        scale = scale[valid == 1]
        img_h = img_h[valid == 1]
        img_w = img_w[valid == 1]
        focal_length = focal_length[valid == 1]

        # 拼接特征 + bbox
        xc = torch.cat([depth_feat, bbox_info], dim=1)

        # heads
        pred_pose = self.head(xc)
        pred_shape = self.shape_head(xc)
        pred_cam = self.cam_head(xc)

        num_valid = depth_feat.shape[0]

        # 6D -> rotmat -> axis-angle
        pred_rotmat = rot6d_to_rotmat(pred_pose).view(num_valid, 24, 3, 3)
        pred_pose = matrix_to_axis_angle(pred_rotmat.view(-1, 3, 3)).view(num_valid, 72)

        # 相机转换
        full_img_shape = torch.stack([img_h, img_w], dim=-1)
        pred_trans = cam_crop2full(pred_cam, center, scale, full_img_shape, focal_length)

        # SMPL 输出
        temp_trans = torch.zeros((num_valid, 3), dtype=pred_rotmat.dtype, device=pred_rotmat.device)
        pred_verts, pred_joints = self.smpl(pred_shape, pred_pose, temp_trans, halpe=True)

        # 2D keypoints
        camera_center = torch.stack([img_w/2, img_h/2], dim=-1)
        pred_keypoints_2d = perspective_projection(
            pred_joints + pred_trans[:, None, :],
            rotation=torch.eye(3, device=pred_pose.device).unsqueeze(0).expand(num_valid, -1, -1),
            translation=torch.zeros(3, device=pred_pose.device).unsqueeze(0).expand(num_valid, -1),
            focal_length=focal_length,
            camera_center=camera_center
        )
        pred_keypoints_2d = (pred_keypoints_2d - center[:, None, :]) / 256

        return {
            'pred_pose': pred_pose,
            'pred_shape': pred_shape,
            'pred_cam_t': pred_trans,
            'pred_rotmat': pred_rotmat,
            'pred_verts': pred_verts,
            'pred_joints': pred_joints,
            'focal_length': focal_length,
            'pred_keypoints_2d': pred_keypoints_2d
        }