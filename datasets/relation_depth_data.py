import os
import cv2
import torch
import numpy as np
from datasets.relation_feature_data import Relation_Feature_Data
import matplotlib.pyplot as plt
import constants
from utils.imutils import crop, flip_img
from utils.FileLoaders import load_pkl

class Relation_Depth_Data(Relation_Feature_Data):
    def __init__(self, train=True, dtype=torch.float32, data_folder='', name='', smpl=None, h36m_depth_folder=None):
        super(Relation_Depth_Data, self).__init__(train=train, dtype=dtype, data_folder=data_folder, name=name, smpl=smpl)
        self.h36m_depth_folder = h36m_depth_folder

        self.depth_names = []
        for f in self.imnames:
            if "Human36M" in f and self.h36m_depth_folder is not None:
                # depth 直接在 E:\depth 里
                fname = os.path.basename(f).replace('.jpg', '.pkl').replace('.png', '.pkl')
                depth_path = os.path.join(self.h36m_depth_folder, fname)
            else:
                # 默认逻辑
                depth_path = os.path.join(
                    self.dataset_dir,
                    f.replace('images', 'depth').replace('.jpg', '.pkl').replace('.png', '.pkl')
                )
            self.depth_names.append(depth_path)        
        
        # 构造 depth 路径
        # self.depth_names = [
        #     os.path.join(
        #         self.dataset_dir,
        #         f.replace('images', 'depth').replace('.jpg', '.pkl').replace('.png', '.pkl')
        #     )
        #     for f in self.imnames
        # ]


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
        noise_factor = np.random.uniform(0.95, 1.05)
        depth_img *= noise_factor
        depth_img = np.clip(depth_img, 0, 255)
        depth_img = depth_img.astype(np.float32)
        return depth_img, ul, br, new_shape, new_x, new_y, old_x, old_y

    def create_data(self, index=0):
        depth_path = self.depth_names[index]
        # print("depth_path",depth_path)
        if not os.path.exists(depth_path):
            # 如果 pkl 文件不存在，直接跳过
            return None

        load_data = {}
        imgname = os.path.join(self.dataset_dir, self.imnames[index])
        # print("imgname:", imgname)
        img_h, img_w = self.img_size[index]
        num_people = len(self.features[index])

        # 读取深度图
        raw_depth = load_pkl(depth_path)['depth_image']
        crop_size = constants.IMG_RES
        depth_imgs = torch.zeros((self.max_people, crop_size, crop_size)).float()
        valid_mask = [False] * self.max_people

        # 初始化其他变量
        bbox = np.zeros(self.max_people, dtype=np.float32)
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

        flip, pn, rot, sc, gt_input = 0, np.ones(3), 0, 1, 0

        for idx in range(num_people):
            if idx >= self.max_people:
                break
            valid[idx] = 1.

            # Load image features
            features = self.features[index][idx].copy()
            center = self.centers[index][idx].copy()
            scale = self.scales[index][idx].copy()
            focal_length = self.intris[index][idx].copy()[0][0]
            keypoints = self.pose2ds[index][idx].copy().astype(np.float32)

            if self.dataset_name in self.joint_dataset:
                if index < len(self.joints) and idx < len(self.joints[index]):
                    joints = torch.from_numpy(self.joints[index][idx].copy()).float()
                else:
                    joints = torch.zeros((17, 4)).float()
                if joints.shape[1] == 3:
                    conf = (torch.abs(torch.sum(joints, dim=1)) > 0).float().reshape(-1,1)
                    joints = torch.cat([joints, conf], dim=1)
                pose = torch.zeros((72,), dtype=self.dtype)
                betas = torch.zeros((10,), dtype=self.dtype)
                trans = torch.zeros((3,), dtype=self.dtype)
                verts = torch.zeros((6890,3), dtype=self.dtype)
                has_smpl = np.zeros(1)
            else:
                pose = self.poses[index][idx].copy().reshape(72,)
                betas = self.shapes[index][idx].copy().reshape(10,)
                pose = torch.from_numpy(self.pose_processing(pose, 0, 0)).float()
                betas = torch.from_numpy(betas).float()

                temp_pose = pose.clone().reshape(-1, 72)
                temp_shape = betas.clone().reshape(-1, 10)
                temp_trans = torch.zeros((temp_pose.shape[0], 3), dtype=temp_pose.dtype, device=temp_pose.device)
                verts, joints = self.smpl(temp_shape, temp_pose, temp_trans, halpe=True)
                verts = verts.squeeze(0)
                joints = joints.squeeze(0)

                try:
                    trans = self.estimate_trans_cliff(joints, keypoints, center, focal_length, img_h, img_w)
                except Exception as e:
                    print(f"Translation estimation failed: {e}")
                    trans = np.zeros((3,), dtype=np.float32)

                trans = torch.from_numpy(trans).float()
                conf = torch.ones((len(joints), 1)).float()
                joints = torch.cat([joints, conf], dim=1)
                has_smpl = np.ones(1)

            keypoints[:,:2] = (keypoints[:,:2] - center) / 256
            keypoints = torch.from_numpy(keypoints).float()
            center = torch.from_numpy(np.array(center)).float()

            has_3d[idx] = 1.
            has_smpls[idx] = has_smpl
            img_features[idx] = torch.from_numpy(features).float()
            vertss[idx] = verts
            gt_joints[idx] = joints
            poses[idx] = pose
            shapes[idx] = betas
            gt_trans[idx] = trans
            imgnames[idx] = imgname
            pose2d_gt[idx] = keypoints
            centers[idx] = center
            scales[idx] = sc*scale
            img_hs[idx] = img_h
            img_ws[idx] = img_w
            focal_lengthes[idx] = focal_length

            bbox = self.bboxs[index][idx]
            center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
            scale = 1.0 * max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 200.0

            # 使用 depth_processing 进行裁剪
            depth_crop, *_ = self.depth_processing(raw_depth, center=center, scale=scale, rot=0, flip=False, pn=None)
            depth_imgs[idx] = torch.from_numpy(depth_crop).float()
            valid_mask[idx] = True

        # 组装返回字典
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

        # 可视化
        self.vis_raw_depth(raw_depth, index=index)
        self.vis_cropped_depth(depth_imgs.unsqueeze(1), valid_mask, index=index)

        return load_data

    # def vis_raw_depth(self, raw_depth, index=None, save_dir='output_depth_vis'):
    #     plt.figure(figsize=(5, 5))
    #     plt.imshow(raw_depth, cmap='gray')
    #     plt.title("Raw Depth Image")
    #     plt.colorbar()
    #     plt.axis('off')
    #     if index is not None:
    #         self.ensure_dir(save_dir)
    #         plt.savefig(os.path.join(save_dir, f'depth_{index}_raw.png'), bbox_inches='tight', pad_inches=0)
    #     plt.close()

    # def vis_cropped_depth(self, cropped_depths, valid_mask, index=None, save_dir='output_depth_vis'):
    #     for idx in range(cropped_depths.shape[0]):
    #         if valid_mask[idx]:
    #             plt.figure(figsize=(4, 4))
    #             plt.imshow(cropped_depths[idx, 0].numpy(), cmap='gray')
    #             plt.title(f"Cropped Depth Person {idx}")
    #             plt.colorbar()
    #             plt.axis('off')
    #             if index is not None:
    #                 self.ensure_dir(save_dir)
    #                 save_path = os.path.join(save_dir, f'depth_{index}_cropped_person_{idx}.png')
    #                 plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    #             plt.close()

    def __getitem__(self, index):
        return self.create_data(index)

    def __len__(self):
        return self.len

