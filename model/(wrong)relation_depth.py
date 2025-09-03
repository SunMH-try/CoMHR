import torch
from torch import nn
from torch.nn import functional as F
from utils.imutils import cam_crop2full
from model.relation_head import PastEncoder
from collections import namedtuple
from utils.geometry import perspective_projection
from utils.rotation_conversions import *
from model.backbones.resnet50 import ResNet50

args = namedtuple('args', [
    'hidden_dim',
    'hyper_scales',
    'learn_prior',
    'nmp_layers',
])

class relation_depth(nn.Module):
    def __init__(self, smpl, num_joints=21):
        super().__init__()
        self.smpl = smpl
        self.args = args(hidden_dim=256, hyper_scales=[3,5], learn_prior=True, nmp_layers=1)

        # models
        scale_num = 2 + len(self.args.hyper_scales)
        
        self.past_encoder = PastEncoder(self.args)

        embed_dim = 3072  # 修改为实际特征维度
        out_dim = 24 * 6
        hidden_dim = 256
        self.project = nn.Sequential(
            nn.LayerNorm(2051),  # 2048 + 3 = 2051
            nn.Linear(2051, hidden_dim),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim + 3),  
            nn.Linear(embed_dim + 3, out_dim),
        )
        self.cam_head = nn.Sequential(
            nn.LayerNorm(embed_dim + 3),  
            nn.Linear(embed_dim + 3, 3),
        )
        self.shape_head = nn.Sequential(
            nn.LayerNorm(embed_dim + 3),  # 2048 + 3 = 2051
            nn.Linear(embed_dim + 3, 10),
        )
        self.depth_encoder = ResNet50()
        self.rgb_encoder = ResNet50()

        self.depth_fc = nn.Linear(2048, 1024)
        self.rgb_fc = nn.Linear(2048, 1024)        

    def set_device(self, device):
        self.device = device
        self.to(device)
    
    def forward(self, data):

        valid = data['valid'].reshape(-1,)

        # ========================
        # 1. Depth 特征
        # ========================
        depth = data['depth']   # [B, P, H, W]
        if depth.ndim == 4:
            B, P, H, W = depth.shape
            depth = depth.reshape(B * P, H, W)

        depth = depth.unsqueeze(1)  # → [B*P, 1, H, W]
        depth_input = depth.repeat(1, 3, 1, 1)  # → [B*P, 3, H, W]
        depth_feat = self.depth_encoder(depth_input)   # [N, 2048]
        depth_features = self.depth_fc(depth_feat)     # [N, 1024]

        # ========================
        # 2. RGB 特征（新加的）
        # ========================
        rgb = data['rgb']   # [B, P, 3, H, W]
        if rgb.ndim == 5:   # flatten batch
            B, P, C, H, W = rgb.shape
            rgb = rgb.reshape(B * P, C, H, W)

        rgb_feat = self.rgb_encoder(rgb)     # [N, 2048]
        rgb_features = self.rgb_fc(rgb_feat) # [N, 1024]

        # ========================
        # 3. RGB + Depth concat
        # ========================
        features = torch.cat([rgb_features, depth_features], dim=1)  # → [N, 2048]

        # ========================
        # 4. bbox info
        # ========================
        batch_size = B
        agent_num = P

        center = data['center'].reshape(batch_size * agent_num, -1)
        scale = data['scale'].reshape(batch_size * agent_num,)
        img_h = data['img_h'].reshape(batch_size * agent_num,)
        img_w = data['img_w'].reshape(batch_size * agent_num,)
        focal_length = data['focal_length'].reshape(batch_size * agent_num,)

        cx, cy, b = center[:, 0], center[:, 1], scale * 200
        bbox_info = torch.stack([cx - img_w / 2., cy - img_h / 2., b], dim=-1)
        bbox_info[:, :2] = bbox_info[:, :2] / focal_length.unsqueeze(-1) * 2.8
        bbox_info[:, 2] = (bbox_info[:, 2] - 0.24 * focal_length) / (0.06 * focal_length)


        # Concat: features + bbox_info → project
        aff_features = torch.cat([features, bbox_info], dim=1)  # → [N, 2051]
        inputs = self.project(aff_features)  # → [N, hidden_dim]
        relation_features = self.past_encoder(inputs, batch_size, agent_num, valid)

        # extract valid entries
        features = features[valid == 1]
        relation_features = relation_features[valid == 1]
        center = center[valid == 1]
        scale = scale[valid == 1]
        img_h = img_h[valid == 1]
        img_w = img_w[valid == 1]
        focal_length = focal_length[valid == 1]
        bbox_info = bbox_info[valid == 1]

        # fuse relation
        feature = torch.cat([features, relation_features], dim=1)  # → [N_valid, 3072]
        num_valid = feature.shape[0]
        

        # Final concat
        xc = torch.cat([feature, bbox_info], dim=1)  # → [N_valid, 3075]

        # Heads
        pred_pose = self.head(xc)
        pred_shape = self.shape_head(xc).view(num_valid, 10)
        pred_cam = self.cam_head(xc).view(num_valid, 3)

        pred_rotmat = rotation_6d_to_matrix(pred_pose).view(num_valid, 24, 3, 3)
        pred_pose = matrix_to_axis_angle(pred_rotmat.view(-1, 3, 3)).view(num_valid, 72)

        full_img_shape = torch.stack((img_h, img_w), dim=-1)
        pred_trans = cam_crop2full(pred_cam, center, scale, full_img_shape, focal_length)

        temp_trans = torch.zeros((num_valid, 3), dtype=pred_rotmat.dtype, device=pred_rotmat.device)
        pred_verts, pred_joints = self.smpl(pred_shape, pred_pose, temp_trans, halpe=True)

        camera_center = torch.stack([img_w / 2, img_h / 2], dim=-1)
        pred_keypoints_2d = perspective_projection(
            pred_joints + pred_trans[:, None, :],
            rotation=torch.eye(3, device=pred_pose.device).unsqueeze(0).expand(num_valid, -1, -1),
            translation=torch.zeros(3, device=pred_pose.device).unsqueeze(0).expand(num_valid, -1),
            focal_length=focal_length,
            camera_center=camera_center
        )
        pred_keypoints_2d = (pred_keypoints_2d - center[:, None, :]) / 256

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