import os
import torch
import numpy as np
from datasets.relation_feature_data import Relation_Feature_Data
import constants
from utils.imutils import crop, flip_img
from utils.FileLoaders import load_pkl

class Relation_Pose_Data(Relation_Feature_Data):
    def __init__(self, train=True, dtype=torch.float32, data_folder='', name='', smpl=None, h36m_depth_folder=None):
        super(Relation_Pose_Data, self).__init__(train=train, dtype=dtype, data_folder=data_folder, name=name, smpl=smpl)
        self.h36m_depth_folder = h36m_depth_folder

        # depth 文件路径
        self.depth_names = []
        for f in self.imnames:
            depth_path = os.path.join(
                self.dataset_dir,
                f.replace('images', 'depth').replace('.jpg', '.pkl')
            )
            self.depth_names.append(depth_path)

        # pose 文件路径（annot pkl 已经给出）
        self.pose_names = []
        for f in self.imnames:
            pose_path = os.path.join(
                self.dataset_dir,
                f.replace('images', 'pose').replace('.jpg', '_pose.pkl')
            )
            self.pose_names.append(pose_path)

    @staticmethod
    def ensure_dir(path):
        if not os.path.exists(path):
            os.makedirs(path)

    def depth_processing(self, depth_img, center, scale, rot, flip, pn):
        depth_img, ul, br, new_shape, new_x, new_y, old_x, old_y = crop(
            depth_img, center, scale, [constants.IMG_RES, constants.IMG_RES], rot=rot
        )
        if flip:
            depth_img = flip_img(depth_img)
        depth_img = np.clip(depth_img, 0, 1000)
        depth_img = depth_img / 1000.
        depth_img = depth_img.astype(np.float32)
        return depth_img, ul, br, new_shape, new_x, new_y, old_x, old_y

    @staticmethod
    def pose3d_processing(kp3d, center, scale, flip=False, pn=None):
        kp3d = kp3d.copy().astype(np.float32)

        # 1. XY normalization (RGB-like)
        kp3d[:, 0] = (kp3d[:, 0] - center[0]) / (200.0 * scale)
        kp3d[:, 1] = (kp3d[:, 1] - center[1]) / (200.0 * scale)

        # 2. Z normalization (Depth-like)
        kp3d[:, 2] = np.clip(kp3d[:, 2], 0, 1000) / 1000.0

        # 3. Horizontal flip
        if flip:
            kp3d[:, 0] = -kp3d[:, 0]
            # 如果有左右关节点，可在这里 swap 左右

        return torch.from_numpy(kp3d).float()

    def create_data(self, index=0):
        depth_path = self.depth_names[index]
        pose_path = self.pose_names[index]

        if not os.path.exists(depth_path) or not os.path.exists(pose_path):
            return None

        load_data = {}
        imgname = os.path.join(self.dataset_dir, self.imnames[index])
        img_h, img_w = self.img_size[index]
        num_people = len(self.features[index])

        # ---------------- Depth ----------------
        raw_depth = load_pkl(depth_path)['depth_image']
        crop_size = constants.IMG_RES
        depth_imgs = torch.zeros((self.max_people, crop_size, crop_size)).float()
        valid_mask = [False] * self.max_people

        # ---------------- Pose3D ----------------
        pose_data_list = load_pkl(pose_path)
        pose3d = torch.zeros((self.max_people, 18, 3)).float()
        pose_mask = torch.zeros((self.max_people, 18)).float()
        pose_Tz = torch.zeros((self.max_people)).float()

        # ---------------- 其他变量 ----------------
        imgnames = ['empty'] * self.max_people
        valid = np.zeros((self.max_people), dtype=np.float32)
        has_3d = np.zeros(self.max_people, dtype=np.float32)
        has_smpls = np.zeros(self.max_people, dtype=np.float32)
        poses = torch.zeros((self.max_people, 72)).float()
        shapes = torch.zeros((self.max_people, 10)).float()
        vertss = torch.zeros((self.max_people, 6890, 3)).float()
        if self.dataset_name in ['Panoptic']:
            gt_joints = torch.zeros((self.max_people, 17, 4)).float()
        else:
            gt_joints = torch.zeros((self.max_people, 26, 4)).float()
        gt_trans = torch.zeros((self.max_people, 3)).float()
        pose2d_gt = torch.zeros((self.max_people, 26, 3)).float()
        img_features = torch.zeros((self.max_people, 2048)).float()
        centers = torch.zeros((self.max_people, 2)).float()
        scales = torch.zeros((self.max_people)).float()
        img_hs = np.zeros((self.max_people), dtype=np.float32)
        img_ws = np.zeros((self.max_people), dtype=np.float32)
        focal_lengthes = np.ones((self.max_people), dtype=np.float32)

        # ---------------- 遍历每个人 ----------------
        for idx in range(num_people):
            if idx >= self.max_people:
                break
            valid[idx] = 1.

            features = self.features[index][idx].copy()
            center = self.centers[index][idx].copy()
            scale = self.scales[index][idx].copy()
            focal_length = self.intris[index][idx].copy()[0][0]
            keypoints = self.pose2ds[index][idx].copy().astype(np.float32)

            # ---------------- Pose3D ----------------
            if idx < len(pose_data_list):
                pdata = pose_data_list[idx]
                kp3d = np.array(pdata['keypoints_3d'], dtype=np.float32)
                mask = np.array(pdata['mask'], dtype=np.float32)
                tz = float(pdata['Tz'])

                kp3d_norm = self.pose3d_processing(kp3d, center=center, scale=scale, flip=False)
                pose3d[idx] = kp3d_norm
                pose_mask[idx] = torch.from_numpy(mask)
                pose_Tz[idx] = tz / 1000.0  # 归一化到 [0,1]

            # ---------------- Depth processing ----------------
            bbox = self.bboxs[index][idx]
            c = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
            s = 1.0 * max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 200.0
            depth_crop, *_ = self.depth_processing(raw_depth, center=c, scale=s, rot=0, flip=False, pn=None)
            depth_imgs[idx] = torch.from_numpy(depth_crop).float()
            valid_mask[idx] = True

            # ---------------- 原RGB+SMPL信息 ----------------
            has_3d[idx] = 1.
            img_features[idx] = torch.from_numpy(features).float()
            poses[idx] = torch.zeros(72).float()
            shapes[idx] = torch.zeros(10).float()
            vertss[idx] = torch.zeros((6890, 3)).float()
            gt_joints[idx] = torch.zeros_like(gt_joints[idx])
            gt_trans[idx] = torch.zeros(3).float()
            imgnames[idx] = imgname
            pose2d_gt[idx] = torch.from_numpy(keypoints)
            centers[idx] = torch.from_numpy(center)
            scales[idx] = scale
            img_hs[idx] = img_h
            img_ws[idx] = img_w
            focal_lengthes[idx] = focal_length

        # ---------------- 组装返回 ----------------
        load_data['depth'] = depth_imgs
        load_data['features'] = img_features
        load_data['valid'] = valid
        load_data['has_3d'] = has_3d
        load_data['has_smpl'] = has_smpls
        load_data['verts'] = vertss
        load_data['gt_joints'] = gt_joints
        load_data['pose'] = poses
        load_data['betas'] = shapes
        load_data['gt_cam_t'] = gt_trans
        load_data['imgname'] = imgnames
        load_data['keypoints'] = pose2d_gt
        load_data["center"] = centers
        load_data["scale"] = scales
        load_data["img_h"] = img_hs
        load_data["img_w"] = img_ws
        load_data["focal_length"] = focal_lengthes
        load_data["pose3d"] = pose3d
        load_data["pose_mask"] = pose_mask
        load_data["pose_Tz"] = pose_Tz
        load_data["depth_valid_mask"] = valid_mask

        return load_data

    def __getitem__(self, index):
        return self.create_data(index)

    def __len__(self):
        return self.len
