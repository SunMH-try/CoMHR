import torch
from torch import nn
from torch.nn import functional as F
from utils.imutils import cam_crop2full
from model.PastEncoder_depth_rgb import PastEncoder_depth_rgb
from collections import namedtuple
from utils.geometry import perspective_projection, rot6d_to_rotmat
from utils.rotation_conversions import *
from model.backbones.resnet50 import ResNet50

args = namedtuple('args', [
    'hidden_dim',
    'hyper_scales',
    'learn_prior',
    'nmp_layers',
])

class relation_depth_rgb(nn.Module):
    def __init__(self, smpl, num_joints=21):
        super().__init__()
        self.smpl = smpl
        self.args = args(hidden_dim=256, hyper_scales=[3,5], learn_prior=True, nmp_layers=1)

        # models
        scale_num = 2 + len(self.args.hyper_scales)
        
        self.past_encoder_rgb   = PastEncoder_depth_rgb(self.args)
        self.past_encoder_depth = PastEncoder_depth_rgb(self.args)

        embed_dim = 4096 
        out_dim = 24 * 6
        hidden_dim = 256
        self.project = nn.Sequential(
            nn.LayerNorm(embed_dim+3), 
            nn.Linear(embed_dim+3, hidden_dim),
        )
        self.project_rgb = nn.Sequential(
            nn.LayerNorm(2048),
            nn.Linear(2048, 1024),
        )
        self.project_depth = nn.Sequential(
            nn.LayerNorm(2048),
            nn.Linear(2048, 1024),
        )
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
        self.depth_encoder = ResNet50(input_c=1)
        self.depth_fc = nn.Linear(2048, 2048)
      
    def set_device(self, device):
        self.device = device
        self.to(device)
    
    def forward(self, data, return_contrastive=True):
        valid = data['valid'].reshape(-1,)
        mask = valid == 1  
        features = data['features']
        batch_size, agent_num, d = features.shape
        features = features.reshape(-1, d)

        # depth 
        depth = data['depth']
        if depth.ndim == 4:
            B, P, H, W = depth.shape
            depth = depth.reshape(B * P, H, W)
        depth = depth.unsqueeze(1)
        depth_feat = self.depth_encoder(depth)
        depth_feat = self.depth_fc(depth_feat)

        # RGB + depth 
        features_concat = torch.cat([features, depth_feat], dim=1)

        # bbox info
        center = data['center'].reshape(batch_size * agent_num, -1)
        scale = data['scale'].reshape(batch_size * agent_num,)
        img_h = data['img_h'].reshape(batch_size * agent_num,)
        img_w = data['img_w'].reshape(batch_size * agent_num,)
        focal_length = data['focal_length'].reshape(batch_size * agent_num,)
        cx, cy, b = center[:, 0], center[:, 1], scale * 200
        bbox_info = torch.stack([cx - img_w / 2., cy - img_h / 2., b], dim=-1)
        bbox_info[:, :2] = bbox_info[:, :2] / focal_length.unsqueeze(-1) * 2.8
        bbox_info[:, 2] = (bbox_info[:, 2] - 0.24 * focal_length) / (0.06 * focal_length)

        aff_features = torch.cat([features_concat, bbox_info], dim=1)
        inputs = self.project(aff_features)

        rgb_features = self.project_rgb(features)
        depth_features = self.project_depth(depth_feat)

        # PastEncoder
        relation_rgb = self.past_encoder_rgb(inputs, rgb_features, batch_size, agent_num, valid)
        relation_depth = self.past_encoder_depth(inputs, depth_features, batch_size, agent_num, valid)

        features_valid = features_concat[mask]
        relation_rgb_valid = relation_rgb[mask]
        relation_depth_valid = relation_depth[mask]
        bbox_info_valid = bbox_info[mask]
        center_valid = center[mask]
        scale_valid = scale[mask]
        img_h_valid = img_h[mask]
        img_w_valid = img_w[mask]
        focal_length_valid = focal_length[mask]

        relation_features = torch.cat([relation_rgb_valid, relation_depth_valid], dim=1)
        feature = torch.cat([features_valid, relation_features], dim=1)
        num_valid = feature.shape[0]

        # Final concat
        xc = torch.cat([feature, bbox_info_valid], dim=1)

        # --------------------- Heads ---------------------
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
