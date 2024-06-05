import os
import torch
import argparse
from os.path import join, isfile
import numpy as np

amass_data_path = '/is/cluster/fast/scratch/sdwivedi/datasets/AMASS/'
amass_train = ['CMU', 'KIT', 'BMLrub', 'DanceDB', 'EyesJapan', 'BMLmovi', 'BMLhandball', 'TotalCapture', 'EKUT', 'ACCAD', 'TCDHands', 'MPI-Limits']
amass_val = ['HumanEva', 'HDM05', 'SFU', 'MPI-Mosh']
amass_test = ['Transitions', 'SSM']

def get_body_params(body_npz, device, index=0, num_betas=10, num_dmpls=8):
    if isinstance(index, int): index = [index]
    body_params = {
        'global_orient': torch.Tensor(body_npz['poses'][index][:,:3]).to(device), # controls the global root orientation
        'body_pose': torch.Tensor(body_npz['poses'][index][:,3:66]).to(device), # controls the body
        'pose_hand': torch.Tensor(body_npz['poses'][index][:,66:]).to(device), # controls the body
        'trans': torch.Tensor(body_npz['trans'][index]).to(device), # controls the global body position
        'betas': torch.Tensor(np.repeat(body_npz['betas'][:num_betas][np.newaxis], repeats=len(body_npz['trans'][index]), axis=0)).to(device), # controls the body shape. Body shape is static
        'gender': body_npz['gender']
    }
    return body_params

def get_all_paths():
    paths, folders, dataset_names = {}, [], []

    for root, dirs, files in os.walk(amass_data_path):
        folders.append(root)
        for name in files:
            dataset_name = root.split('/')[8]
            if dataset_name not in dataset_names and dataset_name:
                paths[f'{dataset_name}'] = []
                dataset_names.append(dataset_name)
            if name.endswith('npz'):
                paths[f'{dataset_name}'].append(os.path.join(root, name))
    return paths, folders, dataset_names



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Preparing data for AMASS Pose Tokenisation')
    parser.add_argument('--output_dir', type=str, default='data', help='dataset directory')
    parser.add_argument('--split', type=str, default='train', help='dataset directory')
    args = parser.parse_args()

    paths, _, _ = get_all_paths()
    total_pose_body = np.empty((0,63))
    trim_rate = 0.2
    skip_rate = 10

    for dt in eval(f'amass_{args.split}'):
        os.makedirs(join(args.output_dir, 'smplh', args.split), exist_ok=True)
        dt_npz = join(args.output_dir, 'smplh', args.split, f'{args.split}_{dt}.npz')
        pose_body = np.empty((0,63))
        trans = np.empty((0,3))
        betas = np.empty((0,10))
        gender = np.empty((0), dtype=str)
        name = np.empty((0), dtype=str) 
        idx = np.empty((0,1), dtype=int)

        if isfile(dt_npz):
            print(f'{dt_npz} file exists, loading...')
            if args.split == 'train':
                total_pose_body = np.append(total_pose_body, np.load(dt_npz)['body_pose'], axis=0)
        else:
            for n_seq in range(len(paths[dt])):
                seq_name = paths[dt][n_seq].split(dt)[-1]
                amass_seq = np.load(paths[dt][n_seq])
                try:
                    N = amass_seq['trans'].shape[0]
                    keep_idx = np.array(range(int(trim_rate * N), int((1-trim_rate) * N), skip_rate))
                    data = get_body_params(amass_seq, device='cpu', index=keep_idx)
                except:
                    continue
                pose_body = np.append(pose_body, data['body_pose'], axis=0)
                betas = np.append(betas, data['betas'], axis=0)
                gender = np.append(gender, [data['gender']]*keep_idx.shape[0])
                name = np.append(name, [f'{dt}{seq_name}']*keep_idx.shape[0])
                print(f'{dt} | {n_seq} | number of frames in {seq_name} -->  {keep_idx.shape}')
            np.savez(dt_npz,
                    pose_body=pose_body,
                    betas=betas,
                    gender=gender,
                    name=name,
                    )
            if args.split == 'train':
                total_pose_body = np.append(total_pose_body, pose_body, axis=0)

    if args.split == 'train':
        print(total_pose_body.shape, total_pose_body.mean(0).shape, total_pose_body.std(0).shape)

