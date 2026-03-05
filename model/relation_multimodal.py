import torch
from torch import nn
from torch.nn import functional as F
from utils.imutils import cam_crop2full
from model.PastEncoder import PastEncoder
from collections import namedtuple
from utils.geometry import perspective_projection, rot6d_to_rotmat
from utils.rotation_conversions import *
from model.backbones.resnet50 import ResNet50
from model.backbones.pose_encoder import PoseEncoder

args = namedtuple('args', [
    'hidden_dim',
    'hyper_scales',
    'learn_prior',
    'nmp_layers',
])

class relation_multimodal(nn.Module):
    def compute_mpjpe_matrix(self, gt_pose):
        N = gt_pose.shape[0]
        if gt_pose.shape[1] == 72:
            gt_pose = gt_pose.view(N, 24, 3)
        diff = gt_pose.unsqueeze(1) - gt_pose.unsqueeze(0)
        mpjpe_matrix = diff.norm(dim=-1).mean(dim=-1)
        return mpjpe_matrix

    def contrastive_loss_intra(self, features, mpjpe_matrix, thresh=0.2, temperature=0.08):
        # Intra-modal contrastive implementation
        eps = 1e-8
        device = features.device
        N = features.shape[0]
        if N <= 1:
            return torch.tensor(0., device=device)

        feats_norm = F.normalize(features, p=2, dim=1)
        sim_matrix = torch.matmul(feats_norm, feats_norm.t()) / (temperature + eps)

        pos_mask = (mpjpe_matrix < thresh).float().to(device)
        diag = torch.eye(N, device=device)
        pos_mask = pos_mask * (1.0 - diag)

        sim_masked = sim_matrix.masked_fill(pos_mask == 0, -1e9)
        log_prob = F.log_softmax(sim_masked, dim=1)

        pos_counts = pos_mask.sum(dim=1)
        valid_rows = pos_counts > 0

        if valid_rows.sum() == 0:
            return torch.tensor(0., device=device)

        loss_per_row = - (log_prob * pos_mask).sum(dim=1) / (pos_counts + eps)
        return loss_per_row[valid_rows].mean()

    def contrastive_loss_cross(self, relation_rgb, relation_depth, relation_pose, method='orthogonality_loss', alpha=0.03):
        # Cross-modal contrastive logic
        device = relation_rgb.device
        N = relation_rgb.shape[0]
        if N == 0:
            return torch.tensor(0., device=device)

        r = F.normalize(relation_rgb, p=2, dim=1)
        d = F.normalize(relation_depth, p=2, dim=1)
        p = F.normalize(relation_pose, p=2, dim=1)

        if method == 'orthogonality_loss':
            sim_rd = F.cosine_similarity(r, d, dim=-1)
            sim_rp = F.cosine_similarity(r, p, dim=-1)
            sim_dp = F.cosine_similarity(d, p, dim=-1)
            loss_vec = torch.relu(-(sim_rd + sim_rp + sim_dp) / 3.0)
            loss = loss_vec.mean()
        elif method == 'pull_close':
            loss_vec = ((r - d).pow(2).sum(dim=1) +
                        (r - p).pow(2).sum(dim=1) +
                        (d - p).pow(2).sum(dim=1)) / 3.0
            loss = loss_vec.mean()
        else:
            raise ValueError("Invalid method")

        return alpha * loss

    def compute_contrastive_loss(self, relation_rgb, relation_depth, relation_pose, gt_pose, thresh=0.2):
        mpjpe_matrix = self.compute_mpjpe_matrix(gt_pose)
        loss_rgb   = self.contrastive_loss_intra(relation_rgb, mpjpe_matrix, thresh)
        loss_depth = self.contrastive_loss_intra(relation_depth, mpjpe_matrix, thresh)
        loss_pose  = self.contrastive_loss_intra(relation_pose, mpjpe_matrix, thresh)
        loss_cross = self.contrastive_loss_cross(relation_rgb, relation_depth, relation_pose)
        return loss_rgb + loss_depth + loss_pose + loss_cross

    def __init__(self, smpl, num_joints=18):
        super().__init__()
        self.smpl = smpl
        self.args = args(hidden_dim=256, hyper_scales=[3,5], learn_prior=True, nmp_layers=1)
        
        self.past_encoder_rgb   = PastEncoder(self.args)
        self.past_encoder_depth = PastEncoder(self.args)
        self.past_encoder_pose  = PastEncoder(self.args)

        embed_dim = 6144
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
        self.project_pose = nn.Sequential(
            nn.LayerNorm(2048),
            nn.Linear(2048, 1024),
        )

        head_in = 9219
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),  
            nn.Linear(head_in, out_dim),
        )
        self.cam_head = nn.Sequential(
            nn.LayerNorm(head_in),  
            nn.Linear(head_in, 3),
        )
        self.shape_head = nn.Sequential(
            nn.LayerNorm(head_in), 
            nn.Linear(head_in, 10),
        )
        
        self.depth_encoder = ResNet50(input_c=1)
        self.depth_fc = nn.Linear(2048, 2048)
        self.pose_encoder = PoseEncoder(input_dim=3)
      
    def set_device(self, device):
        self.device = device
        self.to(device)
    
    def forward(self, data, return_contrastive=True):
        valid = data['valid'].reshape(-1,)
        mask = valid == 1
        features = data['features']
        batch_size, agent_num, d = features.shape
        features = features.reshape(-1, d)

        depth = data['depth']
        if depth.ndim == 4:
            B, P, H, W = depth.shape
            depth = depth.reshape(B * P, H, W)
        depth = depth.unsqueeze(1)
        depth_feat = self.depth_encoder(depth)
        depth_feat = self.depth_fc(depth_feat)

        keypoints_3d = data['keypoints_3d'].to(features.device).float()
        keypoints_mask = data['mask'].to(features.device).float()     
        Tz = data['Tz'].float().to(features.device)

        keypoints_3d = keypoints_3d.view(batch_size * agent_num, -1, 3)
        keypoints_mask = keypoints_mask.view(batch_size * agent_num, -1, 1)
        pose_feat = self.pose_encoder(keypoints_3d, mask=keypoints_mask)

        features_concat = torch.cat([features, depth_feat, pose_feat], dim=1)

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

        rgb_features   = self.project_rgb(features)
        depth_features = self.project_depth(depth_feat)
        pose_features = self.project_pose(pose_feat)

        Tz_flat = Tz.view(-1, 1)
        relation_rgb   = self.past_encoder_rgb(inputs, rgb_features, Tz_flat, batch_size, agent_num, valid)
        relation_depth = self.past_encoder_depth(inputs, depth_features, None, batch_size, agent_num, valid)
        relation_pose  = self.past_encoder_pose(inputs, pose_features, None, batch_size, agent_num, valid)

        features_valid = features_concat[mask]
        relation_rgb_valid = relation_rgb[mask]
        relation_depth_valid = relation_depth[mask]
        relation_pose_valid = relation_pose[mask]
        bbox_info_valid = bbox_info[mask]
        center_valid = center[mask]
        scale_valid = scale[mask]
        img_h_valid = img_h[mask]
        img_w_valid = img_w[mask]
        focal_length_valid = focal_length[mask]

        relation_features = torch.cat([relation_rgb_valid, relation_depth_valid, relation_pose_valid], dim=1)
        feature = torch.cat([features_valid, relation_features], dim=1)
        num_valid = feature.shape[0]
        xc = torch.cat([feature, bbox_info_valid], dim=1)

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

        gt_joints = data['gt_joints']
        gt_joints_xyz = gt_joints[..., :3]
        mask_valid = valid.bool()
        
        relation_rgb_valid_contra   = relation_rgb[mask_valid]      
        relation_depth_valid_contra = relation_depth[mask_valid]    
        relation_pose_valid_contra  = relation_pose[mask_valid]     
        
        if gt_joints_xyz.shape[0] > 1:
            mpjpe_matrix = self.compute_mpjpe_matrix(gt_joints_xyz)
            loss_rgb   = self.contrastive_loss_intra(relation_rgb_valid_contra, mpjpe_matrix)
            loss_depth = self.contrastive_loss_intra(relation_depth_valid_contra, mpjpe_matrix)
            loss_pose  = self.contrastive_loss_intra(relation_pose_valid_contra, mpjpe_matrix)
        else:
            loss_rgb   = torch.tensor(0., device=relation_rgb_valid_contra.device)
            loss_depth = torch.tensor(0., device=relation_depth_valid_contra.device)
            loss_pose  = torch.tensor(0., device=relation_pose_valid_contra.device)
        
        # Validation and protection against unstable loss values
        losses = [loss_rgb, loss_depth, loss_pose]
        for i in range(len(losses)):
            if torch.isnan(losses[i]) or losses[i] > 1e5:
                losses[i] = torch.tensor(0., device=losses[i].device)
        
        loss_rgb, loss_depth, loss_pose = losses
        
        loss_cross = self.contrastive_loss_cross(
            relation_rgb_valid_contra, relation_depth_valid_contra, relation_pose_valid_contra, 
            method='orthogonality_loss', alpha=0.03
        )
        
        if torch.isnan(loss_cross) or loss_cross > 1e5:
            loss_cross = torch.tensor(0., device=loss_cross.device)
        
        pred['loss_contrastive'] = loss_rgb + loss_depth + loss_pose + loss_cross
        return pred