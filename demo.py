"""
Multi-modal demo: relation_multimodal inference + SMPL mesh visualisation.

Usage:
    python demo.py --config=cfg_files/demo.yaml

Demo folder layout  (set demo_dir in demo.yaml):
    demo/
      images/  demo_1.jpg  demo_2.jpg  ...
      depth/   demo_1.pkl  demo_2.pkl  ...   {'depth_image': ndarray(H,W)}
      pose/    demo_1.pkl  demo_2.pkl  ...   [{'keypoints_3d': (18,3), 'mask': (18,), 'Tz': float}, ...]
                                             keypoints_3d X/Y are pixel coords in original image
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # headless: no display
os.environ.setdefault("MPLBACKEND", "Agg")             # matplotlib headless

import torch
import numpy as np
import pickle
import cv2
from torch import nn
from tqdm import tqdm

if hasattr(torch, '_dynamo'):
    torch._dynamo.disable()

from cmd_parser import parse_config
from utils.module_utils import set_seed
from modules import init, ModelLoader
from utils.imutils import crop as imutils_crop, transform as imutils_transform
import constants

MAX_PEOPLE = 8


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def bbox_from_keypoints(kp3d, mask, img_h, img_w, pad_ratio=0.25):
    """Derive [x1,y1,x2,y2] bbox from pixel-space keypoints_3d[:, :2]."""
    valid = kp3d[mask > 0]
    if len(valid) == 0:
        return None
    x1, x2 = valid[:, 0].min(), valid[:, 0].max()
    y1, y2 = valid[:, 1].min(), valid[:, 1].max()
    pad = max(x2 - x1, y2 - y1) * pad_ratio
    x1 = max(0,     x1 - pad)
    y1 = max(0,     y1 - pad)
    x2 = min(img_w, x2 + pad)
    y2 = min(img_h, y2 + pad)
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def depth_processing(depth_img, center, scale):
    """Crop depth map to [IMG_RES × IMG_RES] and normalise to [0, 1]."""
    depth_crop, *_ = imutils_crop(
        depth_img, center, scale,
        [constants.IMG_RES, constants.IMG_RES], rot=0
    )
    depth_crop = np.clip(depth_crop, 0, 1000) / 1000.0
    return depth_crop.astype(np.float32)


def pose3d_processing(kp3d, center, scale):
    """Transform pixel-space XY into cropped-image space; normalise Z to [0,1]."""
    res = [constants.IMG_RES, constants.IMG_RES]
    kp3d_proc  = kp3d.copy().astype(np.float32)
    kp3d_trans = np.zeros_like(kp3d_proc)
    for i in range(kp3d_proc.shape[0]):
        kp3d_trans[i, :2] = imutils_transform(
            pt=kp3d_proc[i, :2] + 1,
            center=center, scale=scale, res=res, rot=0, invert=0
        )
    kp3d_trans[:, 2] = np.clip(kp3d_proc[:, 2], 0, 1000) / 1000.0
    return kp3d_trans


# ─── RGB feature extractor (ImageNet-pretrained ResNet-50 → 2048-d) ──────────

class RGBFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        import torchvision.models as tvm
        backbone = tvm.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(backbone.children())[:-1])

    def forward(self, x):
        return self.features(x).flatten(1)   # (N, 2048)


# ─── Build one batch dict for a single image ──────────────────────────────────

def build_batch(img_path, pose_data, depth_data, focal, feat_extractor, device):
    img_bgr = cv2.imread(img_path)
    img_rgb = img_bgr[:, :, ::-1].copy().astype(np.float32)
    img_h, img_w = img_rgb.shape[:2]

    raw_depth = depth_data['depth_image'].astype(np.float32)

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    norm_imgs   = torch.zeros((MAX_PEOPLE, 3, 256, 192))
    features    = torch.zeros((MAX_PEOPLE, 2048))
    depth_imgs  = torch.zeros((MAX_PEOPLE, constants.IMG_RES, constants.IMG_RES))
    pose3d_t    = torch.zeros((MAX_PEOPLE, 18, 3))
    pose_mask_t = torch.zeros((MAX_PEOPLE, 18))
    pose_Tz_t   = torch.zeros((MAX_PEOPLE,))
    valid_t     = torch.zeros((MAX_PEOPLE,))
    # All slots get valid img/focal values to avoid division-by-zero in bbox_info
    centers_t   = torch.tensor([[img_w / 2., img_h / 2.]] * MAX_PEOPLE)
    scales_t    = torch.ones((MAX_PEOPLE,))
    img_hs_t    = torch.full((MAX_PEOPLE,), float(img_h))
    img_ws_t    = torch.full((MAX_PEOPLE,), float(img_w))
    focals_t    = torch.full((MAX_PEOPLE,), float(focal))

    use_cliff_feat = 'gt_box_cliff_features_hr48' in pose_data[0]
    crops_for_feat = []
    num_valid = 0

    for i, pd in enumerate(pose_data[:MAX_PEOPLE]):
        kp3d = np.array(pd['keypoints_3d'], dtype=np.float32)
        mask = np.array(pd['mask'], dtype=np.float32)
        tz   = float(pd['Tz'])

        # Prefer GT center/scale/focal from pkl; fall back to keypoint-derived bbox
        if 'gt_center' in pd and 'gt_patch_scale' in pd:
            center = list(np.array(pd['gt_center'], dtype=np.float32))
            scale  = float(pd['gt_patch_scale'])
        else:
            bbox = bbox_from_keypoints(kp3d, mask, img_h, img_w)
            if bbox is None:
                continue
            cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
            center = [cx, cy]
            scale  = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 200.0

        focal_i = float(pd['intri'][0][0]) if 'intri' in pd else focal

        valid_t[i]   = 1.
        centers_t[i] = torch.tensor(center)
        scales_t[i]  = scale
        img_hs_t[i]  = img_h
        img_ws_t[i]  = img_w
        focals_t[i]  = focal_i
        num_valid   += 1

        # Depth crop
        depth_imgs[i] = torch.from_numpy(
            depth_processing(raw_depth, center, scale)
        )

        # RGB crop (256×192)
        rgb_crop, *_ = imutils_crop(img_rgb, center, scale, [256, 256], rot=0)
        rgb_crop = np.clip(rgb_crop, 0, 255) / 255.0
        rgb_crop = (rgb_crop - mean) / std
        rgb_crop = cv2.resize(rgb_crop, (192, 256))
        norm_imgs[i] = torch.from_numpy(rgb_crop.transpose(2, 0, 1)).float()

        # CLIFF features from pkl (preferred) or ResNet-50 fallback
        if use_cliff_feat:
            features[i] = torch.from_numpy(
                np.array(pd['gt_box_cliff_features_hr48'], dtype=np.float32)
            )
        else:
            rgb_sq, *_ = imutils_crop(img_rgb, center, scale, [224, 224], rot=0)
            rgb_sq = np.clip(rgb_sq, 0, 255) / 255.0
            rgb_sq = (rgb_sq - mean) / std
            crops_for_feat.append(torch.from_numpy(rgb_sq.transpose(2, 0, 1)).float())

        # 3-D pose
        pose3d_t[i]    = torch.from_numpy(pose3d_processing(kp3d, center, scale))
        pose_mask_t[i] = torch.from_numpy(mask)
        pose_Tz_t[i]   = np.clip(tz, 0, 1000) / 1000.0

    if not use_cliff_feat and crops_for_feat:
        with torch.no_grad():
            feats = feat_extractor(torch.stack(crops_for_feat).to(device))
        features[:num_valid] = feats.cpu()

    data = {
        'imgname':      img_path,
        'img':          norm_imgs.unsqueeze(0),
        'features':     features.unsqueeze(0),
        'depth':        depth_imgs.unsqueeze(0),
        'keypoints_3d': pose3d_t.unsqueeze(0),
        'mask':         pose_mask_t.unsqueeze(0),
        'Tz':           pose_Tz_t.unsqueeze(0),
        'valid':        valid_t.unsqueeze(0),
        'center':       centers_t.unsqueeze(0),
        'scale':        scales_t.unsqueeze(0),
        'img_h':        img_hs_t.unsqueeze(0),
        'img_w':        img_ws_t.unsqueeze(0),
        'focal_length': focals_t.unsqueeze(0),
        'gt_joints':    torch.zeros((1, 26, 4)),
    }
    return data, num_valid


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(**args):
    set_seed(7)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    demo_dir   = os.path.normpath(os.path.join(script_dir, args['demo_dir']))
    print(f'Demo folder : {demo_dir}')

    dtype  = torch.float32
    device = (torch.device(index=args.get('gpu_index'), type='cuda')
              if torch.cuda.is_available() else torch.device('cpu'))

    out_dir, logger, smpl = init(dtype=dtype, **args)

    model = ModelLoader(dtype=dtype, device=device, output=out_dir, **args)
    model.data_folder = demo_dir
    model.model.eval()

    feat_ext = RGBFeatureExtractor().to(device)
    feat_ext.eval()

    images_dir = os.path.join(demo_dir, 'images')
    depth_dir  = os.path.join(demo_dir, 'depth')
    pose_dir   = os.path.join(demo_dir, 'pose')

    img_files = sorted(
        f for f in os.listdir(images_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    )
    print(f'Found {len(img_files)} image(s): {img_files}')

    with torch.no_grad():
        for img_file in tqdm(img_files, desc='Demo'):
            img_path   = os.path.join(images_dir, img_file)
            stem       = os.path.splitext(img_file)[0]
            depth_path = os.path.join(depth_dir, stem + '.pkl')
            pose_path  = os.path.join(pose_dir,  stem + '.pkl')

            if not os.path.exists(pose_path):
                print(f'  [skip] pose file missing: {pose_path}')
                continue
            if not os.path.exists(depth_path):
                print(f'  [skip] depth file missing: {depth_path}')
                continue

            pose_data  = load_pkl(pose_path)
            depth_data = load_pkl(depth_path)

            img_bgr = cv2.imread(img_path)
            h, w    = img_bgr.shape[:2]
            focal   = (h ** 2 + w ** 2) ** 0.5

            data, num_valid = build_batch(
                img_path, pose_data, depth_data, focal, feat_ext, device
            )
            if num_valid == 0:
                print(f'  [skip] no valid persons in {img_file}')
                continue
            print(f'  {img_file}: {num_valid} person(s)')

            imgname = data.pop('imgname')
            data    = {k: v.to(device).float() for k, v in data.items()}
            data['imgname'] = imgname

            valid = data['valid'].reshape(-1,)
            data['valid_focal_length'] = (
                data['focal_length'].reshape(MAX_PEOPLE)[valid == 1]
            )

            pred = model.model(data)

            # ── debug ──
            print(f'    pred keys: {list(pred.keys())}')
            if 'pred_verts' in pred:
                v = pred['pred_verts']
                print(f'    pred_verts shape={v.shape}  min={v.min():.3f}  max={v.max():.3f}')
            if 'pred_cam_t' in pred:
                t = pred['pred_cam_t']
                print(f'    pred_cam_t shape={t.shape}  val={t[0].detach().cpu().numpy()}')
            print(f'    focal={data["valid_focal_length"].detach().cpu().numpy()}')
            print(f'    out_dir={out_dir}')
            # ── end debug ──

            if 'pred_verts' not in pred:
                print(f'  [skip] model did not return pred_verts for {img_file}')
                continue

            results = {
                'imgs':         imgname,
                'pred_trans':   pred['pred_cam_t'].detach().cpu().numpy().astype(np.float32),
                'focal_length': data['valid_focal_length'].detach().cpu().numpy().astype(np.float32),
                'pred_verts':   pred['pred_verts'].detach().cpu().numpy().astype(np.float32),
            }
            model.save_demo_results(results, 0, 1)
            print(f'    saved: {imgname}')

    logger.close()
    print(f'\nResults saved to: {out_dir}')


if __name__ == '__main__':
    args = parse_config()
    main(**args)
