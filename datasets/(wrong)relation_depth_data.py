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
    def __init__(self, train=True, dtype=torch.float32, data_folder='', name='', smpl=None):
        super(Relation_Depth_Data, self).__init__(train=train, dtype=dtype, data_folder=data_folder, name=name, smpl=smpl)
        # 这里初始化 depth_names
        depth_dir = os.path.join(self.dataset_dir, 'depth')  # 指向 depth 子目录
  

        # 保证每个 imname 对应一个 depth 文件
        self.depth_names = [
            os.path.join(depth_dir, os.path.splitext(os.path.basename(f))[0] + '.pkl') 
            for f in self.imnames
        ]
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
        depth_img = depth_img.astype(np.float32)  # 不扩展通道！
        return depth_img, ul, br, new_shape, new_x, new_y, old_x, old_y
    
    def create_data(self, index=0):
        load_data = {}
        flip, pn, rot, sc, gt_input = 0, np.ones(3), 0, 1, 0
        imgname = os.path.join(self.dataset_dir, self.imnames[index])
        try:
            origin_img = cv2.imread(imgname)
            origin_img = origin_img[:,:,::-1].copy().astype(np.float32)
        except TypeError:
            print(imgname)
        orig_shape = np.array(origin_img.shape)[:2]
        img_h, img_w = orig_shape
        img_h, img_w = self.img_size[index]
        num_people = len(self.features[index])

        # 读取原始 RGB 图像
        raw_img = cv2.imread(imgname)
        raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)

        # 读取深度图
        raw_depth = load_pkl(self.depth_names[index])['depth_image']
        crop_size = constants.IMG_RES

        depth_imgs = torch.zeros((self.max_people, crop_size, crop_size)).float()
        rgb_imgs = torch.zeros((self.max_people, 3, crop_size, crop_size)).float()
        valid_mask = [False] * self.max_people  # 用于可视化

        # 初始化一堆变量（和原来一致）
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

        for idx in range(num_people):
            if idx >= self.max_people:
                break
            valid[idx] = 1.

            # Load meta
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

            # 归一化 2D keypoints
            keypoints[:,:2] = (keypoints[:,:2] - center) / 256
            keypoints = torch.from_numpy(keypoints).float()
            center = torch.from_numpy(np.array(center)).float()

            # === 图像裁剪 (新增部分) ===
            bbox = self.bboxs[index][idx]
            center_crop = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
            scale_crop = 1.0 * max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 200.0

            # 裁剪 RGB
            rgb_crop, *_ = crop(
                raw_img, center=center_crop, scale=scale_crop, 
                res=[constants.IMG_RES, constants.IMG_RES], rot=0
            )
            rgb_crop = rgb_crop.astype(np.float32) / 255.0
            rgb_crop = torch.from_numpy(rgb_crop.transpose(2,0,1)).float()  # HWC->CHW

            # 裁剪 Depth
            depth_crop, *_ = self.depth_processing(
                raw_depth, center=center_crop, scale=scale_crop, rot=0, flip=False, pn=None
            )

            # 存储
            rgb_imgs[idx] = rgb_crop
            depth_imgs[idx] = torch.from_numpy(depth_crop).float()
            img_features[idx] = torch.from_numpy(features).float()
            valid_mask[idx] = True

        load_data['rgb'] = rgb_imgs  
        load_data['depth'] = depth_imgs  # [max_people, H, W]
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


    def vis_raw_depth(self, raw_depth, index=None, save_dir='output_depth_vis'):
        plt.figure(figsize=(5, 5))
        plt.imshow(raw_depth, cmap='gray')
        plt.title("Raw Depth Image")
        plt.colorbar()
        plt.axis('off')

        # 保存图像
        if index is not None:
            self.ensure_dir(save_dir)
            plt.savefig(os.path.join(save_dir, f'depth_{index}_raw.png'), bbox_inches='tight', pad_inches=0)

        plt.close()  # 清理图像资源

    def vis_cropped_depth(self, cropped_depths, valid_mask, index=None, save_dir='output_depth_vis'):
        for idx in range(cropped_depths.shape[0]):
            if valid_mask[idx]:
                plt.figure(figsize=(4, 4))
                plt.imshow(cropped_depths[idx, 0].numpy(), cmap='gray')
                plt.title(f"Cropped Depth Person {idx}")
                plt.colorbar()
                plt.axis('off')

                # 保存图像
                if index is not None:
                    self.ensure_dir(save_dir)
                    save_path = os.path.join(save_dir, f'depth_{index}_cropped_person_{idx}.png')
                    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)

                plt.close()  # 避免内存泄漏

    def __getitem__(self, index):
        return self.create_data(index)

    def __len__(self):
        return self.len
