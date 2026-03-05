'''
 @FileName    : process.py
 @EditTime    : 2022-09-27 16:18:51
 @Author      : Buzhen Huang
 @Email       : hbz@seu.edu.cn
'''

import torch
import numpy as np
import cv2
from tqdm import tqdm
import time
import os
import pickle

def save_params_individual(results, frame_dir, start_index=0):
    os.makedirs(frame_dir, exist_ok=True)
    num_people = len(results['pred_trans'])
    for i in range(num_people):
        person_result = {
            'pose': results['pred_pose'][i],
            'betas': results['pred_shape'][i],
            'trans': results['pred_trans'][i],
            'imgname': results['imgs'][i],
            'img_h': results['img_h'][i],
            'img_w': results['img_w'][i]
        }
        img_basename = os.path.basename(results['imgs'][i]).split('.')[0]
        save_name = f"{img_basename}_{start_index + i:04d}.pkl"
        save_path = os.path.join(frame_dir, save_name)
        with open(save_path, 'wb') as f:
            pickle.dump(person_result, f)

def extract_valid(data):
    batch_size, agent_num, d = data['keypoints'].shape[:3]
    valid = data['valid'].reshape(-1,)

    data['center'] = data['center'] 
    data['scale'] = data['scale'] 
    data['img_h'] = data['img_h'] 
    data['img_w'] = data['img_w'] 
    data['focal_length'] = data['focal_length'] 

    data['valid_img_h'] = data['img_h'].reshape(batch_size*agent_num,)[valid == 1]
    data['valid_img_w'] = data['img_w'].reshape(batch_size*agent_num,)[valid == 1]
    data['valid_focal_length'] = data['focal_length'].reshape(batch_size*agent_num,)[valid == 1]
    data['has_3d'] = data['has_3d'].reshape(batch_size*agent_num,1)[valid == 1]
    data['has_smpl'] = data['has_smpl'].reshape(batch_size*agent_num,1)[valid == 1]
    data['verts'] = data['verts'].reshape(batch_size*agent_num, 6890, 3)[valid == 1]
    data['gt_joints'] = data['gt_joints'].reshape(batch_size*agent_num, -1, 4)[valid == 1]
    data['pose'] = data['pose'].reshape(batch_size*agent_num, 72)[valid == 1]
    data['betas'] = data['betas'].reshape(batch_size*agent_num, 10)[valid == 1]
    data['keypoints'] = data['keypoints'].reshape(batch_size*agent_num, 26, 3)[valid == 1]
    data['gt_cam_t'] = data['gt_cam_t'].reshape(batch_size*agent_num, 3)[valid == 1]

    data['ori_imgname'] = data['imgname']
    imgname = (np.array(data['imgname']).T).reshape(batch_size*agent_num,)[valid.detach().cpu().numpy() == 1]
    data['imgname'] = imgname.tolist()
    return data

def extract_valid_demo(data):
    batch_size, agent_num, _, _, _ = data['img'].shape
    valid = data['valid'].reshape(-1,)
    data['center'] = data['center']
    data['scale'] = data['scale']
    data['img_h'] = data['img_h']
    data['img_w'] = data['img_w']
    data['focal_length'] = data['focal_length']
    data['valid_focal_length'] = data['focal_length'].reshape(batch_size*agent_num,)[valid == 1]
    return data

def to_device(data, device):
    imnames = {'imgname':data['imgname']} 
    data = {k:v.to(device).float() for k, v in data.items() if k not in ['imgname']}
    data = {**imnames, **data}
    return data

def relation_train(model, loss_func, train_loader, epoch, num_epoch, device=torch.device('cpu')):
    print('model training')
    len_data = len(train_loader)
    model.model.train(mode=True)
    if model.scheduler is not None:
        model.scheduler.step()

    train_loss = 0.
    for i, data in enumerate(train_loader):
        if data is None: continue        
        data = to_device(data, device)
        data = extract_valid(data)
        pred = model.model(data)
        loss, cur_loss_dict = loss_func.calcul_trainloss(pred, data)
        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()
        if model.scheduler is not None:
            model.scheduler.batch_step()
        loss_batch = loss.detach()
        print('epoch: %d/%d, batch: %d/%d, loss: %.6f' %(epoch, num_epoch, i, len_data, loss_batch), cur_loss_dict)
        train_loss += loss_batch
    return train_loss/len_data

# Standard Test Version (Commented out)
def relation_test(model, loss_func, loader, device=torch.device('cpu')):
    print('model testing')
    loss_all = 0.
    model.model.eval()
    with torch.no_grad():
        for i, data in enumerate(loader):
            if data is None: continue
            batchsize = data['keypoints'].shape[0]
            data = to_device(data, device)
            data = extract_valid(data)
            pred = model.model(data)
            loss, cur_loss_dict = loss_func.calcul_testloss(pred, data)
            if True:
                results = {}
                results.update(imgs=data['ori_imgname'])
                results.update(pred_trans=pred['pred_cam_t'].detach().cpu().numpy().astype(np.float32))
                results.update(gt_trans=data['gt_cam_t'].detach().cpu().numpy().astype(np.float32))
                results.update(focal_length=data['valid_focal_length'].detach().cpu().numpy().astype(np.float32))
                results.update(valid=data['valid'].detach().cpu().numpy().astype(np.float32))
                if 'MPJPE_instance' in cur_loss_dict.keys() or 'MPJPE_H36M_instance' in cur_loss_dict.keys():
                    results.update(MPJPE=loss.detach().cpu().numpy().astype(np.float32))
                if 'pred_verts' not in pred.keys():
                    results.update(pred_joints=pred['pred_joints'].detach().cpu().numpy().astype(np.float32))
                    results.update(gt_joints=data['gt_joints'].detach().cpu().numpy().astype(np.float32))
                    model.save_joint_results(results, i, batchsize)
                else:
                    results.update(pred_verts=pred['pred_verts'].detach().cpu().numpy().astype(np.float32))
                    results.update(gt_verts=data['verts'].detach().cpu().numpy().astype(np.float32))
                    model.save_test_results(results, i, batchsize)
            loss_batch = loss.detach().mean()
            print('batch: %d/%d, loss: %.6f ' %(i, len(loader), loss_batch), cur_loss_dict)
            loss_all += loss_batch
        model.finalize_test_results(iter=len(loader))
        loss_all = loss_all / len(loader)
        return loss_all

# Giga Metrics Version (Active)
# def relation_test(model, loss_func, loader, device=torch.device('cpu')):
#     output_root = r"/FinalGiga"
#     os.makedirs(output_root, exist_ok=True)
#     print('model testing (Giga)')
#     loss_all = 0.
#     model.model.eval()
#     frame_person_count = {}
#     with torch.no_grad():
#         for i, data in enumerate(loader):
#             if data is None: continue
#             data = to_device(data, device)
#             data = extract_valid(data)
#             pred = model.model(data)
#             loss, cur_loss_dict = loss_func.calcul_testloss(pred, data)
#             img_names = data['imgname']
#             results = {
#                 'imgs': img_names,
#                 'pred_trans': pred['pred_cam_t'].detach().cpu().numpy().astype(np.float32),
#                 'pred_pose': pred['pred_pose'].detach().cpu().numpy().astype(np.float32),
#                 'pred_shape': pred['pred_shape'].detach().cpu().numpy().astype(np.float32),
#                 'img_h': data['valid_img_h'].detach().cpu().numpy().astype(np.float32),
#                 'img_w': data['valid_img_w'].detach().cpu().numpy().astype(np.float32)
#             }
#             for j in range(len(img_names)):
#                 img_name = img_names[j]
#                 frame_name = os.path.basename(img_name).split('.')[0]
#                 frame_dir = os.path.join(output_root, frame_name)
#                 os.makedirs(frame_dir, exist_ok=True)
#                 if frame_name not in frame_person_count: frame_person_count[frame_name] = 0
#                 person_idx = frame_person_count[frame_name]
#                 frame_person_count[frame_name] += 1
#                 save_path = os.path.join(frame_dir, f"{frame_name}_{person_idx:04d}.pkl")
#                 person_result = {
#                     'pose': results['pred_pose'][j], 'betas': results['pred_shape'][j],
#                     'trans': results['pred_trans'][j], 'imgname': img_name,
#                     'img_h': results['img_h'][j], 'img_w': results['img_w'][j]
#                 }
#                 with open(save_path, 'wb') as f: pickle.dump(person_result, f)
#                 cam_file = os.path.join(frame_dir, 'camparams.txt')
#                 if not os.path.exists(cam_file):
#                     focal = float(data['valid_focal_length'][0].cpu().numpy())
#                     h, w = float(data['valid_img_h'][0].cpu().numpy()), float(data['valid_img_w'][0].cpu().numpy())
#                     intri = np.array([[focal, 0, w/2], [0, focal, h/2], [0, 0, 1]], dtype=np.float32)
#                     extri = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32)
#                     with open(cam_file, 'w') as f_cam:
#                         f_cam.write("0\n")
#                         for row in intri: f_cam.write(f"{row[0]:.6f} {row[1]:.6f} {row[2]:.6f}\n")
#                         f_cam.write("0 0\n")
#                         for row in extri: f_cam.write(f"{row[0]:.6f} {row[1]:.6f} {row[2]:.6f} {row[3]:.6f}\n")
#             loss_batch = loss.detach().mean()
#             print(f'batch: {i}, loss: {loss_batch:.6f}', cur_loss_dict)
#             loss_all += loss_batch
#         model.finalize_test_results(iter=len(loader))
#         return loss_all / len(loader)

def relation_demo(model, loader, device=torch.device('cpu')):
    print('model demo')
    model.model.eval()
    with torch.no_grad():
        for i, data in tqdm(enumerate(loader), total=len(loader)):
            batchsize = data['img'].shape[0]
            data = to_device(data, device)
            data = extract_valid_demo(data)
            pred = model.model(data)
            results = {'imgs': data['imgname']}
            results.update(pred_trans=pred['pred_cam_t'].detach().cpu().numpy().astype(np.float32))
            results.update(focal_length=data['valid_focal_length'].detach().cpu().numpy().astype(np.float32))
            if 'pred_verts' not in pred.keys():
                results.update(pred_joints=pred['pred_joints'].detach().cpu().numpy().astype(np.float32))
                model.save_demo_joint_results(results, i, batchsize)
            else:
                results.update(pred_verts=pred['pred_verts'].detach().cpu().numpy().astype(np.float32))
                model.save_demo_results(results, i, batchsize)