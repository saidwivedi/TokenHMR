import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
from smplx import SMPLH, SMPLX
from utils.rotation_conversions import axis_angle_to_matrix
from utils.skeleton import get_smplx_body_parts

def get_dataloader(hparams, split, shuffle=True):

    batch_size = hparams.DATA.BATCH_SIZE
    debug = hparams.EXP.DEBUG
    data_root = hparams.DATA.DATA_ROOT
    mask_body_parts = hparams.DATA.MASK_BODY_PARTS
    rot_type = hparams.ARCH.ROT_TYPE
    debug = hparams.EXP.DEBUG
    smpl_type = hparams.ARCH.SMPL_TYPE
    num_workers = 8

    if split == 'train':
        ds_list = hparams.DATA.TRAINLIST.split('_')
        partition = [1] if len(ds_list) == 1 else hparams.DATA.TRAIN_PART.split('_')
        assert len(ds_list) == len(partition), "Number of datasets and parition does not match"
    elif split == 'val':
        ds_list = hparams.DATA.VALLIST.split('_')
        partition = [1/len(ds_list)]*len(ds_list)
    elif split == 'test':
        ds_list = hparams.DATA.TESTLIST.split('_')
        partition = [1/len(ds_list)]*len(ds_list)

    print(f'List of datasets for {split} --> {ds_list} with shuffle = {shuffle}')

    if len(ds_list) == 1:
        dataset = VQPoseDataset(ds_list[0], split, data_root, rot_type, smpl_type, mask_body_parts, debug)
    else:
        if split == 'train':
            dataset = MixedTrainDataset(ds_list, partition, split, data_root, rot_type, smpl_type, mask_body_parts, debug)
        else:
            dataset = ValDataset(ds_list, split, data_root, rot_type, smpl_type, debug)

    loader = torch.utils.data.DataLoader(dataset,
                                        batch_size,
                                        shuffle=shuffle,
                                        num_workers=num_workers,
                                        drop_last = True)
    if split == 'train':
        return cycle(loader)
    else:
        return loader

class MixedTrainDataset(data.Dataset):
    
    def __init__(self, ds_list, partition, split, data_root, rot_type, smpl_type, mask_body_parts, debug):

        self.ds_list = ds_list
        partition = [float(part) for part in partition]
        self.partition = np.array(partition).cumsum()
        
        self.datasets = [VQPoseDataset(ds, split, data_root, rot_type, smpl_type, mask_body_parts, debug) for ds in ds_list]
        self.length = max([len(ds) for ds in self.datasets])

    def __getitem__(self, index):
            p = np.random.rand()
            for i in range(len(self.ds_list)):
                if p <= self.partition[i]:
                    return self.datasets[i][index % len(self.datasets[i])]

    def __len__(self):
        return self.length

class VQPoseDataset(data.Dataset):
    def __init__(self, dt, split= 'train', data_root='', rot_type = 'rotmat', smpl_type= 'smplx', mask_body_parts = False, debug = False):

        self.data_root = pjoin(data_root, smpl_type, split)
        self.joints_num = 21
        self.smplx_body_parts = get_smplx_body_parts()
        self.mask_body_parts = mask_body_parts
        self.split = split
        self.smpl_type = smpl_type

        self.smpl_model = eval(f'{smpl_type.upper()}')(f'../data/body_models/{smpl_type}', num_betas=10, ext='pkl')
        data = np.load(pjoin(self.data_root, f'{split}_{dt}.npz'))
        total_samples = data['pose_body'].shape[0]
        
        random_idx = None
        if debug:
            debug_data_length = 600
            random_idx = np.random.choice(total_samples, size=debug_data_length, replace=False)
            print(f'In debug mode, processing with less data')

        self.pose_body = data['pose_body'][random_idx] if random_idx is not None else data['pose_body']
        self.betas = data['betas'][random_idx] if random_idx is not None else data['betas']
        self.gender = data['gender'][random_idx] if random_idx is not None else data['gender']
        self.name = data['name'][random_idx] if random_idx is not None else data['name']
        self.dataset_name = f'_{dt}'
        print(f"Processing {dt} for {split} with {self.pose_body.shape[0]} samples...")

        self.rot_type = rot_type

    def __len__(self):
        return self.pose_body.shape[0]

    def __getitem__(self, index):
        item = {}

        pose_body_aa = self.pose_body[index]
        
        pose_body_aa = torch.Tensor(pose_body_aa.reshape(-1)).float()
        item['pose_body_aa'] = pose_body_aa.clone()

        body_model = self.smpl_model(body_pose=pose_body_aa.view(-1, pose_body_aa.shape[0]))
        item['body_vertices'] = body_model.vertices[0].detach().float()
        item['body_joints'] = body_model.joints[0].detach().float()

        pose_body_rot = axis_angle_to_matrix(pose_body_aa.view(-1,3))
        item['gt_pose_body'] = pose_body_rot
        item['dataset_name'] = self.dataset_name

        return item

class ValDataset(data.Dataset):
    def __init__(self, dataset_list, split= 'val', data_root='', rot_type = 'rotmat', smpl_type = 'smplx', debug = False):

        self.data_root = pjoin(data_root, smpl_type, split)
        self.joints_num = 21
        self.smplx_body_parts = get_smplx_body_parts()
        self.split = split
        self.smpl_type = smpl_type

        self.smpl_model = eval(f'{smpl_type.upper()}')(f'../data/body_models/{smpl_type}', num_betas=10, ext='pkl')
        
        self.pose_body = np.empty((0,63))
        self.betas = np.empty((0,10))
        self.gender = np.empty((0), dtype=str)
        self.name = np.empty((0), dtype=str) 
        self.dataset_name = ''

        for dt in dataset_list:
            self.dataset_name += f'_{dt}'
            data = np.load(pjoin(self.data_root, f'{split}_{dt}.npz'))
            self.pose_body = np.append(self.pose_body, data['pose_body'], axis=0)
            self.betas = np.append(self.betas, data['betas'], axis=0)
            self.gender = np.append(self.gender, data['gender'], axis=0)
            self.name = np.append(self.name, data['name'], axis=0)
            print(f"Processing {dt} for {split} with {data['pose_body'].shape[0]} samples...")

        if debug:
            debug_data_length = 600
            random_idx = np.random.choice(self.pose_body.shape[0], size=debug_data_length, replace=False)
            print(f'In debug mode, processing with less data')
            self.pose_body = self.pose_body[random_idx]
            self.betas = self.betas[random_idx]
            self.gender = self.gender[random_idx]
            self.name = self.name[random_idx]

        self.rot_type = rot_type
        print(f"Total number of samples {self.pose_body.shape[0]}")

    def __len__(self):
        return self.pose_body.shape[0]

    def __getitem__(self, index):
        item = {}

        pose_body_aa = self.pose_body[index]
        
        pose_body_aa = torch.Tensor(pose_body_aa.reshape(-1)).float()
        item['pose_body_aa'] = pose_body_aa.clone()

        body_model = self.smpl_model(body_pose=pose_body_aa.view(-1, pose_body_aa.shape[0]))
        item['body_vertices'] = body_model.vertices[0].detach().float()
        item['body_joints'] = body_model.joints[0].detach().float()

        pose_body_rot = axis_angle_to_matrix(pose_body_aa.view(-1,3))
        item['gt_pose_body'] = pose_body_rot

        item['dataset_name'] = self.dataset_name

        return item

def cycle(iterable):
    while True:
        for x in iterable:
            yield x
