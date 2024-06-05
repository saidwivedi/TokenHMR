
import os
import torch
import argparse
from os.path import join, isfile
import joblib as jl
import numpy as np

moyo_data_root = '/ps/project/datasets/MOYO/20220923_20220926_with_hands/mosh_smpl'

def get_body_params(body_pkl, device, index=0, num_betas=10, num_dmpls=8):
    if isinstance(index, int): index = [index]
    body_params = {
        'root_orient': torch.Tensor(body_pkl['global_orient'][index]).to(device), # controls the global root orientation
        'pose_body': torch.Tensor(body_pkl['body_pose'][index, :63]).to(device), # controls the body
        'trans': torch.Tensor(body_pkl['transl'][index]).to(device), # controls the global body position
        'betas': torch.Tensor(body_pkl['betas'][index, :10]).to(device),
        # 'betas': torch.Tensor(np.repeat(body_pkl['betas'][:num_betas][np.newaxis], repeats=len(body_pkl['transl'][index]), axis=0)).to(device), # controls the body shape. Body shape is static
        'gender': 'neutral'
    }
    return body_params

def get_all_paths(moyo_data_path):
    return [os.path.join(moyo_data_path, f) for f in os.listdir(moyo_data_path) if f.endswith('pkl')]


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Preparing data for MOYO Pose Tokenisation')
    parser.add_argument('--output_dir', type=str, default='data', help='dataset directory')
    parser.add_argument('--split', type=str, default='val', help='dataset directory')
    parser.add_argument('--skip_rate', type=int, default=5, help='skipping the frame')
    args = parser.parse_args()

    paths = get_all_paths(join(moyo_data_root, args.split))
    total_pose_body = np.empty((0,63))
    trim_rate = 0.1

    os.makedirs(join(args.output_dir, args.split), exist_ok=True)
    dt_npz = join(args.output_dir, args.split, f'{args.split}_MOYO.npz')
    pose_body = np.empty((0,63))
    trans = np.empty((0,3))
    betas = np.empty((0,10))
    gender = np.empty((0), dtype=str)
    name = np.empty((0), dtype=str) 
    idx = np.empty((0,1), dtype=int)

    for n_seq in range(len(paths)):
        seq_name = os.path.basename(paths[n_seq])
        moyo_seq = jl.load(paths[n_seq])
        # N = moyo_seq['transl'].shape[0]
        # keep_idx = np.array(range(int(trim_rate * N), int((1-trim_rate) * N), args.skip_rate))
        # data = get_body_params(moyo_seq, device='cpu', index=keep_idx)
        try:
            N = moyo_seq['transl'].shape[0]
            keep_idx = np.array(range(int(trim_rate * N), int((1-trim_rate) * N), args.skip_rate))
            data = get_body_params(moyo_seq, device='cpu', index=keep_idx)
            # for k, v in data.items():
            #     print(f'{k} --> {v.shape}')
        except:
            continue
        pose_body = np.append(pose_body, data['pose_body'], axis=0)
        betas = np.append(betas, data['betas'], axis=0)
        gender = np.append(gender, [data['gender']]*keep_idx.shape[0])
        name = np.append(name, [f'{seq_name}']*keep_idx.shape[0])
        print(f'{n_seq} | number of frames in {seq_name} -->  {keep_idx.shape}')
    np.savez(dt_npz,
            pose_body=pose_body,
            betas=betas,
            gender=gender,
            name=name,
            )
    total_pose_body = np.append(total_pose_body, pose_body, axis=0)

    print(f'Total pose samples --> {total_pose_body.shape}')

