'''
 @FileName    : dataset.py
 @EditTime    : 2022-09-27 16:03:55
 @Author      : Buzhen Huang
 @Email       : hbz@seu.edu.cn
 @Description : 
'''

import os
#debug
#import sys
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import cv2
from utils.geometry import estimate_translation_np
from utils.imutils import get_crop, keyp_crop2origin, surface_projection, img_crop2origin
from datasets.base import base
import constants
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，专门用于生成图片文件，无窗口
import matplotlib.pyplot as plt
import os
import torch
import numpy as np
import pickle
import cv2
from tqdm import tqdm
import constants
from utils.FileLoaders import load_pkl

class Relation_Feature_Data(base):
    def __init__(self, train=True, dtype=torch.float32, data_folder='', name='', smpl=None):
        super(Relation_Feature_Data, self).__init__(train=train, dtype=dtype, data_folder=data_folder, name=name, smpl=smpl)

        self.max_people = 8
        self.dataset_name = name
        self.joint_dataset = ['Panoptic', 'JTA']

        if self.is_train:
            dataset_annot = os.path.join(self.dataset_dir, 'annot/new_train.pkl')
        else:
            self.eval = True
            dataset_annot = os.path.join(self.dataset_dir,'annot/new_test.pkl')

        params = self.load_pkl(dataset_annot)
        self.features, self.poses, self.shapes, self.imnames, self.masks, self.img_size, self.bboxs, self.intris, self.centers, self.scales, self.pose2ds, self.joints = [], [], [], [], [], [], [], [], [], [], [], []
        #scales, self.pose2ds, self.joints, self.depth_names = [], [], [], [], [], [], [], [], [], [], [], [], []
        for seq in tqdm(params, total=len(params)):
            if len(seq) < 1:
                continue
            for i, frame in enumerate(seq):
                # if i > 10:
                #     break
                features, pose2ds, poses, shapes, bboxs, intris, centers, scales, joints = [], [], [], [], [], [], [], [], []
                for key in frame.keys():
                    if key in ['img_path', 'h_w', 'depth_path']:
                        continue
            

                    pose2ds.append(np.array(frame[key]['halpe_joints_2d'], dtype=self.np_type))
                    poses.append(np.array(frame[key]['pose'], dtype=self.np_type))
                    shapes.append(np.array(frame[key]['betas'], dtype=self.np_type))
                    intris.append(np.array(frame[key]['intri'], dtype=self.np_type).reshape(3,3))

                    bboxs.append(np.array(frame[key]['bbox'], dtype=self.np_type).reshape(-1,))
                    features.append(np.array(frame[key]['gt_box_cliff_features_hr48'], dtype=self.np_type).reshape(-1,))
                    centers.append(np.array(frame[key]['gt_center'], dtype=self.np_type))
                    scales.append(np.array(frame[key]['gt_patch_scale'], dtype=self.np_type))

                    if self.dataset_name in ['Panoptic']:
                        joints.append(np.array(frame[key]['h36m_joints_3d'], dtype=self.np_type))
                    elif self.dataset_name in self.joint_dataset:
                        joints.append(np.array(frame[key]['halpe_joints_3d'], dtype=self.np_type))

                    if len(features) >= self.max_people:
                        self.img_size.append(frame['h_w'])
                        self.imnames.append(frame['img_path'])
                        #self.depth_names.append(frame['depth_path'])

                        self.features.append(features)
                        self.centers.append(centers)
                        self.scales.append(scales)
                        self.pose2ds.append(pose2ds)
                        self.poses.append(poses)
                        self.shapes.append(shapes)
                        self.bboxs.append(bboxs)
                        self.intris.append(intris)
                        self.joints.append(joints)
                        features, pose2ds, poses, shapes, bboxs, intris, centers, scales, joints = [], [], [], [], [], [], [], [], []

                if len(features) > 0:
                    self.img_size.append(frame['h_w'])
                    self.imnames.append(frame['img_path'])
                    #self.depth_names.append(frame['depth_path'])

                    self.features.append(features)
                    self.centers.append(centers)
                    self.scales.append(scales)
                    self.pose2ds.append(pose2ds)
                    self.poses.append(poses)
                    self.shapes.append(shapes)
                    self.bboxs.append(bboxs)
                    self.intris.append(intris)
                    self.joints.append(joints)
                
        del frame
        del params

        self.len = len(self.features)


    def vis_input(self, image, pred_keypoints, keypoints, pose, betas, trans, valids, new_shapes, new_xs, new_ys, old_xs, old_ys, focal_length, img_h, img_w):
        # Show image
        image = image.copy()
        self.vis_img('img', image)

        # Show keypoints
        for key, valid, new_shape, new_x, new_y, old_x, old_y in zip(keypoints, valids, new_shapes, new_xs, new_ys, old_xs, old_ys):
            if valid == 1:
                key = keyp_crop2origin(key.clone(), new_shape, new_x, new_y, old_x, old_y)
                # keypoints = keypoints[:,:-1].detach().numpy() * constants.IMG_RES + center.numpy()
                key = key[:,:-1].astype(np.int)
                for k in key:
                    image = cv2.circle(image, tuple(k), 3, (0,0,255), -1)
        # self.vis_img('keyp', image)

        # Show keypoints
        for key, valid, new_shape, new_x, new_y, old_x, old_y in zip(pred_keypoints, valids, new_shapes, new_xs, new_ys, old_xs, old_ys):
            if valid == 1:
                key = keyp_crop2origin(key.clone(), new_shape, new_x, new_y, old_x, old_y)
                # keypoints = keypoints[:,:-1].detach().numpy() * constants.IMG_RES + center.numpy()
                key = key[:,:-1].astype(np.int)
                for k in key:
                    image = cv2.circle(image, tuple(k), 3, (0,255,0), -1)
        self.vis_img('keyp', image)
        

        # Show SMPL
        pose = pose.reshape(-1, 72)[valids==1]
        betas = betas.reshape(-1, 10)[valids==1]
        trans = trans.reshape(-1, 3)[valids==1]
        extri = np.eye(4)
        intri = np.eye(3)
        intri[0][0] = focal_length
        intri[1][1] = focal_length
        intri[0][2] = img_w / 2
        intri[1][2] = img_h / 2
        verts, joints = self.smpl(betas, pose, trans)
        for vert in verts:
            vert = vert.detach().numpy()
            projs, image = surface_projection(vert, self.smpl.faces, extri, intri, image.copy(), viz=False)
        self.vis_img('smpl', image)
        output_path = 'output_folder/smpl_result.jpg'
        cv2.imwrite(output_path, image)


    def estimate_trans_cliff(self, joints, keypoints, center, focal_length, img_h, img_w):
        joints = joints.detach().numpy()
        # keypoints[:,:-1] = keypoints[:,:-1] * constants.IMG_RES + np.array(center)
        
        gt_cam_t = estimate_translation_np(joints, keypoints[:,:2], keypoints[:,2], focal_length=focal_length, center=[img_w/2, img_h/2])
        return gt_cam_t
    

    def create_data(self, index=0):
        load_data = {}
        flip, pn, rot, sc, gt_input = 0, np.ones(3), 0, 1, 0

        imgname = os.path.join(self.dataset_dir, self.imnames[index])
        img_h, img_w = self.img_size[index]

        num_people = len(self.features[index])
        crop_size = 224 # 定义裁剪尺寸

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
        #depths = torch.zeros((self.max_people, crop_size, crop_size)).float()
        
        #depth = load_pkl(os.path.join(self.dataset_dir, self.depth_names[index]))['depth_image']

        # vis depth

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

            # 获取当前人的bbox，bbox格式 [xmin, ymin, xmax, ymax]
            bbox = self.bboxs[index][idx]
            xmin, ymin, xmax, ymax = bbox

            # 获取当前人的bbox，bbox格式 [xmin, ymin, xmax, ymax]
            bbox = self.bboxs[index][idx]
            xmin, ymin, xmax, ymax = bbox

            # 强制转为整数并确保在图像范围内
            xmin = int(max(0, xmin))
            ymin = int(max(0, ymin))
            xmax = int(min(img_w, xmax))
            ymax = int(min(img_h, ymax))

            # 如果bbox有效，进行裁剪并resize（避免畸变）
            # if xmin < xmax and ymin < ymax:
            #     cropped_depth = depth[ymin:ymax, xmin:xmax]

            #     # 原始尺寸
            #     h, w = cropped_depth.shape
            #     scale = crop_size / max(h, w)
            #     new_w, new_h = int(w * scale), int(h * scale)
                
            #     # 缩放后图像
            #     resized = cv2.resize(cropped_depth, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                
            #     # 创建目标图像，初始化为0（padding）
            #     canvas = np.zeros((crop_size, crop_size), dtype=resized.dtype)
                
            #     # 计算居中的位置（pad）
            #     top = (crop_size - new_h) // 2
            #     left = (crop_size - new_w) // 2
                
            #     canvas[top:top+new_h, left:left+new_w] = resized
            #     depths[idx] = torch.from_numpy(canvas).float()
            # else:
            #     depths[idx] = torch.zeros((crop_size, crop_size)).float()


            # vis crop depth

        # 将所有depth拼成一个 tensor 后返回
        #load_data['depth'] = depths  # 返回统一尺寸的深度图
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
    
        return load_data


    def __getitem__(self, index):
        data = self.create_data(index)
        return data

    def __len__(self):
        return self.len
