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

class relation_multimodal(nn.Module):
    def compute_mpjpe_matrix(self, gt_pose):
        N = gt_pose.shape[0]
        if gt_pose.shape[1] == 72:
            gt_pose = gt_pose.view(N, 24, 3)
        diff = gt_pose.unsqueeze(1) - gt_pose.unsqueeze(0)
        mpjpe_matrix = diff.norm(dim=-1).mean(dim=-1)
        return mpjpe_matrix

    def contrastive_loss_intra(self, features, mpjpe_matrix, thresh=40.0, temperature=0.1):
        sim_matrix = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=-1)
        pos_mask = (mpjpe_matrix < thresh).float()
        exp_sim = torch.exp(sim_matrix / temperature)
        pos_loss = -torch.log((exp_sim * pos_mask).sum(dim=1) / (exp_sim.sum(dim=1) + 1e-8))
        return pos_loss.mean()
    def contrastive_loss_cross(self, relation_rgb, relation_depth):
        # 只有 RGB 和 Depth
        loss = (relation_rgb - relation_depth).pow(2).sum(dim=1)
        return loss.mean()

    def compute_contrastive_loss(self, relation_rgb, relation_depth, gt_pose, thresh=40.0):
        mpjpe_matrix = self.compute_mpjpe_matrix(gt_pose)
        loss_rgb   = self.contrastive_loss_intra(relation_rgb, mpjpe_matrix, thresh)
        loss_depth = self.contrastive_loss_intra(relation_depth, mpjpe_matrix, thresh)
        loss_cross = self.contrastive_loss_cross(relation_rgb, relation_depth)
        total_loss = loss_rgb + loss_depth + loss_cross
        return total_loss

    # def contrastive_loss_cross(self, relation_rgb, relation_depth, relation_pose):
    #     loss = ((relation_rgb - relation_depth).pow(2).sum(dim=1) +
    #             (relation_rgb - relation_pose).pow(2).sum(dim=1) +
    #             (relation_depth - relation_pose).pow(2).sum(dim=1)) / 3.0
    #     return loss.mean()

    # def compute_contrastive_loss(self, relation_rgb, relation_depth, relation_pose, gt_pose, thresh=40.0):
    #     mpjpe_matrix = self.compute_mpjpe_matrix(gt_pose)
    #     loss_rgb   = self.contrastive_loss_intra(relation_rgb, mpjpe_matrix, thresh)
    #     loss_depth = self.contrastive_loss_intra(relation_depth, mpjpe_matrix, thresh)
    #     loss_pose  = self.contrastive_loss_intra(relation_pose, mpjpe_matrix, thresh)
    #     loss_cross = self.contrastive_loss_cross(relation_rgb, relation_depth, relation_pose)
    #     total_loss = loss_rgb + loss_depth + loss_pose + loss_cross
    #     return total_loss
    
    
    def __init__(self, smpl, num_joints=21):
        super().__init__()
        self.smpl = smpl
        self.args = args(hidden_dim=256, hyper_scales=[3,5], learn_prior=True, nmp_layers=1)

        # models
        scale_num = 2 + len(self.args.hyper_scales)
        
        #self.past_encoder = PastEncoder(self.args)
        self.past_encoder_rgb   = PastEncoder(self.args)
        self.past_encoder_depth = PastEncoder(self.args)
        self.past_encoder_pose  = PastEncoder(self.args)


        embed_dim = 4096  # 修改为实际特征维度
        out_dim = 24 * 6
        hidden_dim = 256
        self.project = nn.Sequential(
            nn.LayerNorm(2051),  # 2048 + 3 = 2051
            nn.Linear(2051, hidden_dim),
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
        self.depth_fc = nn.Linear(2048, 2048)
      
    def set_device(self, device):
        self.device = device
        self.to(device)
    
    def forward(self, data, return_contrastive=True):
        # --------------------- 基本特征 ---------------------
        valid = data['valid'].reshape(-1,)
        mask = valid == 1  # 有效 agent mask
        features = data['features']
        batch_size, agent_num, d = features.shape
        features = features.reshape(-1, d)

        # depth 特征
        depth = data['depth']
        if depth.ndim == 4:
            B, P, H, W = depth.shape
            depth = depth.reshape(B * P, H, W)
        depth = depth.unsqueeze(1).repeat(1, 3, 1, 1)
        depth_feat = self.depth_encoder(depth)
        depth_feat = self.depth_fc(depth_feat)

        # RGB + depth 投影
        rgb_features = self.project_rgb(features)
        depth_features = self.project_depth(depth_feat)
        features_concat = torch.cat([rgb_features, depth_features], dim=1)

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

        # PastEncoder 两模态
        relation_rgb = self.past_encoder_rgb(inputs, rgb_features, batch_size, agent_num, valid)
        relation_depth = self.past_encoder_depth(inputs, depth_features, batch_size, agent_num, valid)

        # --------------------- 提取有效 agent ---------------------
        features_valid = features_concat[mask]
        relation_rgb_valid = relation_rgb[mask]
        relation_depth_valid = relation_depth[mask]
        bbox_info_valid = bbox_info[mask]
        center_valid = center[mask]
        scale_valid = scale[mask]
        img_h_valid = img_h[mask]
        img_w_valid = img_w[mask]
        focal_length_valid = focal_length[mask]

        # 融合 relation 特征
        relation_features = torch.cat([relation_rgb_valid, relation_depth_valid], dim=1)
        feature = torch.cat([features_valid, relation_features], dim=1)
        num_valid = feature.shape[0]

        # Final concat
        xc = torch.cat([feature, bbox_info_valid], dim=1)

        # --------------------- Heads ---------------------
        pred_pose = self.head(xc)
        pred_shape = self.shape_head(xc).view(num_valid, 10)
        pred_cam = self.cam_head(xc).view(num_valid, 3)

        pred_rotmat = rotation_6d_to_matrix(pred_pose).view(num_valid, 24, 3, 3)
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

        # ---------------- contrastive loss ----------------
        gt_joints = data['gt_joints']  # shape = [batch_size, actual_agent_num, joint_num, dim] 或 [num_agents, joint_num, dim]
        joint_num, dim = gt_joints.shape[-2], gt_joints.shape[-1]

        # flatten 成 (B*A, J, dim) 与 features 对齐
        num_features = features.shape[0]  # 总 agent 数
        num_agents = gt_joints.shape[0]

        # 如果 gt_joints 第一维比 features 少，pad 0
        if num_agents < num_features:
            pad_size = num_features - num_agents
            padding = torch.zeros((pad_size, joint_num, dim), device=gt_joints.device, dtype=gt_joints.dtype)
            gt_joints = torch.cat([gt_joints, padding], dim=0)
        elif num_agents > num_features:
            gt_joints = gt_joints[:num_features]

        # mask 选出有效 agent
        gt_joints_valid = gt_joints[mask]  # shape = [num_valid, joint_num, dim]

        # 补 z 维度
        if gt_joints_valid.shape[2] == 2:
            gt_joints_xyz = torch.cat([gt_joints_valid, torch.zeros_like(gt_joints_valid[..., :1])], dim=-1)
        else:
            gt_joints_xyz = gt_joints_valid[..., :3]

        # 至少两个 agent 才能计算 intra-modal loss
        if gt_joints_valid.shape[0] > 1:
            mpjpe_matrix = self.compute_mpjpe_matrix(gt_joints_xyz)
            loss_rgb = self.contrastive_loss_intra(relation_rgb_valid, mpjpe_matrix)
            loss_depth = self.contrastive_loss_intra(relation_depth_valid, mpjpe_matrix)
        else:
            loss_rgb = torch.tensor(0., device=features.device)
            loss_depth = torch.tensor(0., device=features.device)

        # cross-modal loss
        loss_cross = ((relation_rgb_valid - relation_depth_valid).pow(2).sum(dim=1)).mean()
        loss_contrastive = loss_rgb + loss_depth + loss_cross
        pred['loss_contrastive'] = loss_contrastive


        return pred



