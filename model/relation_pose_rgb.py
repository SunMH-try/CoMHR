import torch
from torch import nn
from torch.nn import functional as F
from utils.imutils import cam_crop2full
from model.PastEncoder import PastEncoder
from collections import namedtuple
from utils.geometry import perspective_projection, rot6d_to_rotmat
from utils.rotation_conversions import *
from model.backbones.pose_encoder import PoseEncoder

args = namedtuple('args', [
    'hidden_dim',
    'hyper_scales',
    'learn_prior',
    'nmp_layers',
])

class relation_pose_rgb(nn.Module):
    def __init__(self, smpl, num_joints=18):
        super().__init__()
        self.smpl = smpl
        self.args = args(hidden_dim=256, hyper_scales=[3,5], learn_prior=True, nmp_layers=1)

        # Modalities: RGB + Pose
        self.past_encoder_rgb = PastEncoder(self.args)
        self.past_encoder_pose = PastEncoder(self.args)

        embed_dim = 4096  # RGB(2048) + Pose(2048)
        hidden_dim = 256
        out_dim = 24 * 6

        # Fusion layer
        self.project = nn.Sequential(
            nn.LayerNorm(embed_dim + 3),
            nn.Linear(embed_dim + 3, hidden_dim),
        )

        # Modality-specific projections
        self.project_rgb = nn.Sequential(
            nn.LayerNorm(2048),
            nn.Linear(2048, 1024),
        )
        self.project_pose = nn.Sequential(
            nn.LayerNorm(2048),
            nn.Linear(2048, 1024),
        )

        # Regression heads
        self.head = nn.Sequential(
            nn.LayerNorm(6147),  
            nn.Linear(6147, out_dim),
        )
        self.cam_head = nn.Sequential(
            nn.LayerNorm(6147),  
            nn.Linear(6147, 3),
        )
        self.shape_head = nn.Sequential(
            nn.LayerNorm(6147),  
            nn.Linear(6147, 10),
        )

        self.pose_encoder = PoseEncoder(input_dim=3)

    def set_device(self, device):
        self.device = device
        self.to(device)

    def forward(self, data):
        # Basic inputs
        valid = data['valid'].reshape(-1,)
        mask = valid == 1
        features = data['features']  # RGB features
        batch_size, agent_num, d = features.shape
        features = features.reshape(-1, d)

        # Process pose features
        keypoints_3d = data['keypoints_3d'].to(features.device).float()
        keypoints_mask = data['mask'].to(features.device).float()

        keypoints_3d = keypoints_3d.view(batch_size * agent_num, -1, 3)
        keypoints_mask = keypoints_mask.view(batch_size * agent_num, -1, 1)

        pose_feat = self.pose_encoder(keypoints_3d, mask=keypoints_mask)

        # Multimodal concatenation (RGB + Pose)
        features_concat = torch.cat([features, pose_feat], dim=1)

        # Bounding box information
        center = data['center'].reshape(batch_size * agent_num, -1)
        scale = data['scale'].reshape(batch_size * agent_num,)
        img_h = data['img_h'].reshape(batch_size * agent_num,)
        img_w = data['img_w'].reshape(batch_size * agent_num,)
        focal_length = data['focal_length'].reshape(batch_size * agent_num,)
        cx, cy, b = center[:, 0], center[:, 1], scale * 200
        bbox_info = torch.stack([cx - img_w / 2., cy - img_h / 2., b], dim=-1)
        bbox_info[:, :2] = bbox_info[:, :2] / focal_length.unsqueeze(-1) * 2.8
        bbox_info[:, 2] = (bbox_info[:, 2] - 0.24 * focal_length) / (0.06 * focal_length)

        # Feature fusion
        aff_features = torch.cat([features_concat, bbox_info], dim=1)
        inputs = self.project(aff_features)

        # Modality projections
        rgb_features = self.project_rgb(features)
        pose_features = self.project_pose(pose_feat)

        # Bi-modal PastEncoders
        relation_rgb = self.past_encoder_rgb(inputs, rgb_features, None, batch_size, agent_num, valid)
        relation_pose = self.past_encoder_pose(inputs, pose_features, None, batch_size, agent_num, valid)

        # Extract valid agents
        features_valid = features_concat[mask]
        relation_rgb_valid = relation_rgb[mask]
        relation_pose_valid = relation_pose[mask]
        bbox_info_valid = bbox_info[mask]
        center_valid = center[mask]
        scale_valid = scale[mask]
        img_h_valid = img_h[mask]
        img_w_valid = img_w[mask]
        focal_length_valid = focal_length[mask]

        # Fuse relation features
        relation_features = torch.cat([relation_rgb_valid, relation_pose_valid], dim=1)
        feature = torch.cat([features_valid, relation_features], dim=1)
        num_valid = feature.shape[0]

        # Final concatenation
        xc = torch.cat([feature, bbox_info_valid], dim=1)

        # Regress SMPL parameters
        pred_pose = self.head(xc)
        pred_shape = self.shape_head(xc).view(num_valid, 10)
        pred_cam = self.cam_head(xc).view(num_valid, 3)

        pred_rotmat = rot6d_to_rotmat(pred_pose).view(num_valid, 24, 3, 3)
        pred_pose = matrix_to_axis_angle(pred_rotmat.view(-1, 3, 3)).view(num_valid, 72)

        full_img_shape = torch.stack((img_h_valid, img_w_valid), dim=-1)
        pred_trans = cam_crop2full(pred_cam, center_valid, scale_valid, full_img_shape, focal_length_valid)

        temp_trans = torch.zeros((num_valid, 3), dtype=pred_rotmat.dtype, device=pred_rotmat.device)
        pred_verts, pred_joints = self.smpl(pred_shape, pred_pose, temp_trans, halpe=True)

        camera_center = torch.stack([img_w_valid / 2, img_h_valid / 2], dim=-1)
        pred_keypoints_2d = perspective_projection(
            pred_joints + pred_trans[:, None, :],
            rotation=torch.eye(3, device=pred_pose.device).unsqueeze(0).expand(num_valid, -1, -1),
            translation=torch.zeros(3, device=pred_pose.device).unsqueeze(0).expand(num_valid, -1),
            focal_length=focal_length_valid,
            camera_center=camera_center
        )
        pred_keypoints_2d = (pred_keypoints_2d - center_valid[:, None, :]) / 256

        pred = {
            'pred_pose': pred_pose,
            'pred_shape': pred_shape,
            'pred_cam_t': pred_trans,
            'pred_rotmat': pred_rotmat,
            'pred_verts': pred_verts,
            'pred_joints': pred_joints,
            'focal_length': focal_length_valid,
            'pred_keypoints_2d': pred_keypoints_2d,
        }
        return pred