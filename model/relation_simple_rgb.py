
import torch
from torch import nn
from torch.nn import functional as F
from utils.imutils import cam_crop2full, vis_img
from collections import namedtuple
from utils.geometry import perspective_projection, rot6d_to_rotmat
from utils.rotation_conversions import *
import cv2

args = namedtuple('args', [
    'hidden_dim',
    'hyper_scales',
    'learn_prior',
    'nmp_layers',
])

        
class relation_simple_rgb(nn.Module):
    def __init__(self, smpl, num_joints=21):
        super().__init__()
        self.smpl = smpl
        self.args = args(hidden_dim=256, hyper_scales=[3,5], learn_prior=True, nmp_layers=1)

        # models
        scale_num = 2 + len(self.args.hyper_scales)
        
        embed_dim = 2048
        out_dim = 24 * 6
        hidden_dim = 256
        self.project = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 1024),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(1027),
            nn.Linear(1027, out_dim),
        )
        self.cam_head = nn.Sequential(
            nn.LayerNorm(1027),
            nn.Linear(1027 , 3),
        )
        self.shape_head = nn.Sequential(
            nn.LayerNorm(1027),
            nn.Linear(1027, 10),
        )

    def set_device(self, device):
        self.device = device
        self.to(device)
    

    def forward(self, data):
        batch_size, agent_num, d = data['features'].shape

        valid = data['valid'].reshape(-1,)
        features = data['features'].reshape(-1, d)
        
        # bbox 和相机信息
        center = data['center'].reshape(batch_size*agent_num, -1)
        scale = data['scale'].reshape(batch_size*agent_num,)
        img_h = data['img_h'].reshape(batch_size*agent_num,)
        img_w = data['img_w'].reshape(batch_size*agent_num,)
        focal_length = data['focal_length'].reshape(batch_size*agent_num,)

        cx, cy, b = center[:, 0], center[:, 1], scale * 200
        bbox_info = torch.stack([cx - img_w / 2., cy - img_h / 2., b], dim=-1)
        bbox_info[:, :2] = bbox_info[:, :2] / focal_length.unsqueeze(-1) * 2.8
        bbox_info[:, 2] = (bbox_info[:, 2] - 0.24 * focal_length) / (0.06 * focal_length)

        # 只经过project（对应ResNet特征处理）
        features = self.project(features)

        # 取valid部分
        features = features[valid == 1]
        center = center[valid == 1]
        scale = scale[valid == 1]
        img_h = img_h[valid == 1]
        img_w = img_w[valid == 1]
        focal_length = focal_length[valid == 1]
        bbox_info = bbox_info[valid == 1]

        num_valid = len(features)

        # 拼接特征 + bbox信息
        xc = torch.cat([features, bbox_info], 1)

        # 直接预测
        pred_pose = self.head(xc)
        pred_shape = self.shape_head(xc).view(num_valid, 10)
        pred_cam = self.cam_head(xc).view(num_valid, 3)

        pred_rotmat = rotation_6d_to_matrix(pred_pose).view(num_valid, 24, 3, 3)
        pred_pose = matrix_to_axis_angle(pred_rotmat.view(-1, 3, 3)).view(num_valid, 72)

        # 相机转换
        full_img_shape = torch.stack((img_h, img_w), dim=-1)
        pred_trans = cam_crop2full(pred_cam, center, scale, full_img_shape, focal_length)
        temp_trans = torch.zeros((pred_rotmat.shape[0], 3), dtype=pred_rotmat.dtype, device=pred_rotmat.device)

        # SMPL预测
        pred_verts, pred_joints = self.smpl(pred_shape, pred_pose, temp_trans, halpe=True)

        # 2D关键点
        camera_center = torch.stack([img_w/2, img_h/2], dim=-1)
        pred_keypoints_2d = perspective_projection(
            pred_joints + pred_trans[:,None,:],
            rotation=torch.eye(3, device=pred_pose.device).unsqueeze(0).expand(num_valid, -1, -1),
            translation=torch.zeros(3, device=pred_pose.device).unsqueeze(0).expand(num_valid, -1),
            focal_length=focal_length,
            camera_center=camera_center
        )
        pred_keypoints_2d = (pred_keypoints_2d - center[:,None,:]) / 256

        # 输出
        pred = {
            'pred_pose': pred_pose,
            'pred_shape': pred_shape,
            'pred_cam_t': pred_trans,
            'pred_rotmat': pred_rotmat,
            'pred_verts': pred_verts,
            'pred_joints': pred_joints,
            'focal_length': focal_length,
            'pred_keypoints_2d': pred_keypoints_2d,
        }

        return pred
