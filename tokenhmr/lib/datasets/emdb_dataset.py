import copy
import os
import numpy as np
import torch
from typing import Any, Dict, List
from yacs.config import CfgNode
import braceexpand
import cv2
import smplx
from .dataset import Dataset
from .utils import get_example, expand_to_aspect_ratio
from .smplh_prob_filter import poses_check_probable, load_amass_hist_smooth


body_permutation = [0, 1, 5, 6, 7, 2, 3, 4, 8, 12, 13, 14, 9, 10, 11, 16, 15, 18, 17, 22, 23, 24, 19, 20, 21]
extra_permutation = [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13, 14, 15, 16, 17, 18]
FLIP_KEYPOINT_PERMUTATION = body_permutation + [25 + i for i in extra_permutation]

DEFAULT_MEAN = 255. * np.array([0.485, 0.456, 0.406])
DEFAULT_STD = 255. * np.array([0.229, 0.224, 0.225])
DEFAULT_IMG_SIZE = 256

class EMDBDataset(Dataset):

    def __init__(self,
                 cfg: CfgNode,
                 dataset_file: str,
                 img_dir: str,
                 train: bool = True,
                 prune: Dict[str, Any] = {},
                 **kwargs):
        """
        Dataset class used for loading images and corresponding annotations.
        Args:
            cfg (CfgNode): Model config file.
            dataset_file (str): Path to npz file containing dataset info.
            img_dir (str): Path to image folder.
            train (bool): Whether it is for training or not (enables data augmentation).
        """
        super(EMDBDataset, self).__init__()
        self.train = train
        self.cfg = cfg

        self.img_size = cfg.MODEL.IMAGE_SIZE
        self.mean = 255. * np.array(self.cfg.MODEL.IMAGE_MEAN)
        self.std = 255. * np.array(self.cfg.MODEL.IMAGE_STD)

        self.img_dir = img_dir
        self.data = np.load(dataset_file, allow_pickle=True)

        self.imgname = self.data['imgname']
        self.personid = np.zeros(len(self.imgname), dtype=np.int32)
        self.extra_info = self.data.get('extra_info', [{} for _ in range(len(self.imgname))])
        self.flip_keypoint_permutation = copy.copy(FLIP_KEYPOINT_PERMUTATION)


        num_pose = 3 * (self.cfg.SMPL.NUM_BODY_JOINTS + 1)

        # Bounding boxes are assumed to be in the center and scale format
        self.center = self.data['center']
        self.scale = self.data['scale'].reshape(len(self.center), -1)
        if self.scale.shape[1] == 1:
            self.scale = np.tile(self.scale, (1, 2))
        assert self.scale.shape == (len(self.center), 2)


        self.body_pose = self.data['body_pose'].astype(np.float32)
        self.has_body_pose = self.data['has_body_pose'].astype(np.float32)

        self.betas = self.data['betas'].astype(np.float32)
        self.has_betas = self.data['has_betas'].astype(np.float32)

        self.keypoints_2d = self.data['keypoints_2d']

        # Not needed as we generate it later using vertices
        try:
            body_keypoints_3d = self.data['body_keypoints_3d'].astype(np.float32)
        except KeyError:
            body_keypoints_3d = np.zeros((len(self.center), 25, 4), dtype=np.float32)
        # Try to get extra 3d keypoints, if available
        try:
            extra_keypoints_3d = self.data['extra_keypoints_3d'].astype(np.float32)
        except KeyError:
            extra_keypoints_3d = np.zeros((len(self.center), 19, 4), dtype=np.float32)

        body_keypoints_3d[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], -1] = 0

        self.keypoints_3d = np.concatenate((body_keypoints_3d, extra_keypoints_3d), axis=1).astype(np.float32)

        smpl_cfg = {k.lower(): v for k,v in dict(self.cfg.SMPL).items()}
        for k, v in smpl_cfg.items():
            # print('${SMPL.DATA_DIR}' in v, isinstance(v,int))
            if not isinstance(v,int) and '${SMPL.DATA_DIR}' in v:
                smpl_cfg[k] = v.replace('${SMPL.DATA_DIR}', '')

        smpl_cfg_male = dict(smpl_cfg)
        smpl_cfg_female = dict(smpl_cfg)
        smpl_cfg_male['gender'] = 'male'
        smpl_cfg_female['gender'] = 'female'
        self.smpl_gt_male = smplx.SMPL(**smpl_cfg_male)
        self.smpl_gt_female = smplx.SMPL(**smpl_cfg_female)

        try:
            gender = self.data['gender']
            self.gender = np.array([0 if str(g) == 'm' or str(g)=='male'
                                    else 1 for g in gender]).astype(np.int32)
        except KeyError:
            self.gender = -1*np.ones(len(self.imgname)).astype(np.int32)

    def __len__(self) -> int:
        return len(self.scale)

    def __getitem__(self, idx: int) -> Dict:
        """
        Returns an example from the dataset.
        """
        try:
            image_file_rel = self.imgname[idx].decode('utf-8')
        except AttributeError:
            image_file_rel = self.imgname[idx]
        image_file = os.path.join(self.img_dir, image_file_rel)
        keypoints_2d = self.keypoints_2d[idx].copy()
        keypoints_3d = self.keypoints_3d[idx].copy()

        center = self.center[idx].copy()
        center_x = center[0]
        center_y = center[1]
        scale = self.scale[idx]
        BBOX_SHAPE = self.cfg.MODEL.get('BBOX_SHAPE', None)
        bbox_size = expand_to_aspect_ratio(scale*200, target_aspect_ratio=BBOX_SHAPE).max()
        bbox_expand_factor = bbox_size / ((scale*200).max())
        body_pose = self.body_pose[idx].copy().astype(np.float32)
        betas = self.betas[idx].copy().astype(np.float32)

        has_body_pose = self.has_body_pose[idx].copy()
        has_betas = self.has_betas[idx].copy()

        smpl_params = {'global_orient': body_pose[:3],
                       'body_pose': body_pose[3:],
                       'betas': betas
                      }
        has_smpl_params = {'global_orient': has_body_pose,
                           'body_pose': has_body_pose,
                           'betas': has_betas
                           }

        smpl_params_is_axis_angle = {'global_orient': True,
                                     'body_pose': True,
                                     'betas': False
                                    }

        augm_config = self.cfg.DATASETS.CONFIG
        # Crop image and (possibly) perform data augmentation
        img_patch, keypoints_2d, keypoints_3d, smpl_params, has_smpl_params, img_size = get_example(image_file,
                                                                                                    center_x, center_y,
                                                                                                    bbox_size, bbox_size,
                                                                                                    keypoints_2d, keypoints_3d,
                                                                                                    smpl_params, has_smpl_params,
                                                                                                    self.flip_keypoint_permutation,
                                                                                                    self.img_size, self.img_size,
                                                                                                    self.mean, self.std, self.train, augm_config)

        item = {}
        # These are the keypoints in the original image coordinates (before cropping)
        orig_keypoints_2d = self.keypoints_2d[idx].copy()

        item['img'] = img_patch
        item['keypoints_2d'] = keypoints_2d.astype(np.float32)
        item['orig_keypoints_2d'] = orig_keypoints_2d
        item['box_center'] = self.center[idx].copy()
        item['box_size'] = bbox_size
        item['bbox_expand_factor'] = bbox_expand_factor
        item['img_size'] = 1.0 * img_size[::-1].copy()
        item['smpl_params'] = smpl_params
        item['has_smpl_params'] = has_smpl_params
        item['smpl_params_is_axis_angle'] = smpl_params_is_axis_angle
        item['imgname'] = image_file
        item['imgname_rel'] = image_file_rel
        item['personid'] = int(self.personid[idx])
        item['extra_info'] = copy.deepcopy(self.extra_info[idx])
        item['idx'] = idx
        item['_scale'] = scale
        item['gender'] = self.gender[idx]
        if self.gender[idx] == 1:
            model = self.smpl_gt_female
            gt_smpl_out = self.smpl_gt_female(
                        global_orient=torch.from_numpy(item['smpl_params']['global_orient']).unsqueeze(0),
                        body_pose=torch.from_numpy(item['smpl_params']['body_pose']).unsqueeze(0),
                        betas=torch.from_numpy(item['smpl_params']['betas']).unsqueeze(0))
            gt_vertices = gt_smpl_out.vertices.detach()
        else:
            model = self.smpl_gt_male
            gt_smpl_out = self.smpl_gt_male(
                global_orient=torch.from_numpy(item['smpl_params']['global_orient']).unsqueeze(0),
                body_pose=torch.from_numpy(item['smpl_params']['body_pose']).unsqueeze(0),
                betas=torch.from_numpy(item['smpl_params']['betas']).unsqueeze(0))
            gt_vertices = gt_smpl_out.vertices.detach()
        item['keypoints_3d'] = torch.matmul(model.J_regressor, gt_vertices[0])
        item['vertices'] = gt_vertices[0].float()
        return item