"""
Extract gt_box_cliff_features_hr48 (and gt_center / gt_patch_scale / intri)
from the CrowdPose training pkl for the three demo images, and write them
into the corresponding demo/pose/*.pkl files.

Run once before demo.py:
    python extract_demo_features.py
"""

import pickle
import numpy as np
import os

TRAIN_PKL  = r'F:/grouprec/data/datasets/CrowdPose/annot/train.pkl'
DEMO_DIR   = r'C:/Users/Sun06/Desktop/demo'

# demo stem → image id in CrowdPose
IMG_MAP = {
    'demo_1': '100459',
    'demo_2': '100544',
    'demo_3': '100899',
}


def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_pkl(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def extract_frame(frames, img_id):
    """Return the frame dict whose img_path contains img_id."""
    for frame in frames:
        if img_id in frame.get('img_path', ''):
            return frame
    return None


def main():
    print('Loading training pkl ...')
    train_data = load_pkl(TRAIN_PKL)
    # structure: list[list[frame_dict]]
    if isinstance(train_data, list) and isinstance(train_data[0], list):
        frames = train_data[0]
    else:
        frames = train_data

    pose_dir = os.path.join(DEMO_DIR, 'pose')

    for stem, img_id in IMG_MAP.items():
        pose_path = os.path.join(pose_dir, stem + '.pkl')
        if not os.path.exists(pose_path):
            print(f'[skip] {pose_path} not found')
            continue

        frame = extract_frame(frames, img_id)
        if frame is None:
            print(f'[skip] image {img_id} not found in training pkl')
            continue

        # Person keys: everything except 'img_path' and 'h_w'
        person_keys = [k for k in frame if k not in ('img_path', 'h_w')]
        print(f'{stem} ({img_id}): {len(person_keys)} persons → {person_keys}')

        # Load existing demo pose pkl
        demo_poses = load_pkl(pose_path)

        if len(demo_poses) != len(person_keys):
            print(f'  WARNING: demo has {len(demo_poses)} persons but train has '
                  f'{len(person_keys)} — using min, check order!')

        n = min(len(demo_poses), len(person_keys))
        for i, pk in enumerate(person_keys[:n]):
            person = frame[pk]
            feat = np.array(person['gt_box_cliff_features_hr48'], dtype=np.float32)
            demo_poses[i]['gt_box_cliff_features_hr48'] = feat

            # Also store gt_center / gt_patch_scale / focal for accurate cropping
            if 'gt_center' in person:
                demo_poses[i]['gt_center']      = np.array(person['gt_center'],      dtype=np.float32)
            if 'gt_patch_scale' in person:
                demo_poses[i]['gt_patch_scale'] = float(person['gt_patch_scale'])
            if 'intri' in person:
                demo_poses[i]['intri']          = np.array(person['intri'],           dtype=np.float32)
            if 'bbox' in person:
                demo_poses[i]['bbox']           = np.array(person['bbox'],            dtype=np.float32)

        save_pkl(demo_poses, pose_path)
        print(f'  Saved {n} persons with CLIFF features → {pose_path}')

    print('Done.')


if __name__ == '__main__':
    main()
