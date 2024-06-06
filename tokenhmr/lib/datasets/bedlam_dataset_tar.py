import copy
import os
import glob
import numpy as np
import glob
import torch
from typing import Any, Dict, List
from yacs.config import CfgNode
import braceexpand
import cv2
from ..models.smpl_wrapper import SMPL

from .dataset import Dataset
from .utils import get_example, expand_to_aspect_ratio
from ..utils.rotation_utils import axis_angle_to_matrix

def expand(s):
    return os.path.expanduser(os.path.expandvars(s))
def expand_urls(urls: str|List[str]):
    if isinstance(urls, str):
        urls = [urls]
    urls = [u for url in urls for u in braceexpand.braceexpand(expand(url))]
    return urls

def project(points, cam_trans, cam_int):
    points = points + cam_trans
    cam_int = torch.tensor(cam_int).float()

    projected_points = points / points[:, -1].unsqueeze(-1)
    projected_points = torch.einsum('ij, kj->ki', cam_int, projected_points.float())

    return projected_points.detach().cpu().numpy()


body_permutation = [0, 1, 5, 6, 7, 2, 3, 4, 8, 12, 13, 14, 9, 10, 11, 16, 15, 18, 17, 22, 23, 24, 19, 20, 21]
extra_permutation = [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13, 14, 15, 16, 17, 18]
FLIP_KEYPOINT_PERMUTATION = body_permutation + [25 + i for i in extra_permutation]

DEFAULT_MEAN = 255. * np.array([0.485, 0.456, 0.406])
DEFAULT_STD = 255. * np.array([0.229, 0.224, 0.225])
DEFAULT_IMG_SIZE = 256

class BedlamDataset(Dataset):

    @staticmethod
    def load_tars_as_webdataset(cfg: CfgNode, urls: str|List[str], train: bool,
            resampled=False,
            epoch_size=None,
            cache_dir=None,
            **kwargs) -> Dataset:
        """
        Loads the dataset from a webdataset tar file.
        """

        IMG_SIZE = cfg.MODEL.IMAGE_SIZE
        BBOX_SHAPE = cfg.MODEL.get('BBOX_SHAPE', None)
        MEAN = 255. * np.array(cfg.MODEL.IMAGE_MEAN)
        STD = 255. * np.array(cfg.MODEL.IMAGE_STD)

        smpl_cfg = {k.lower(): v for k,v in dict(cfg.SMPL).items()}
        smpl = SMPL(**smpl_cfg)

        def split_data(source):
            for item in source:
                datas = item['data.pyd']
                for data in datas:
                    if 'detection.npz' in item:
                        det_idx = data['extra_info']['detection_npz_idx']
                        mask = item['detection.npz']['masks'][det_idx]
                    else:
                        mask = np.ones_like(item['jpg'][:,:,0], dtype=bool)
                    yield {
                        '__key__': item['__key__'],
                        'jpg': item['jpg'],
                        'data.pyd': data,
                        'mask': mask}
        # Load the dataset
        if epoch_size is not None:
            resampled = True
        import webdataset as wds
        folder = os.path.join(cfg.DATASETS.DATASET_DIR, urls)
        urls = []
        for fname in os.listdir(folder):
            if 'agora' in fname:
                continue
            ff = os.path.join(folder, fname)
            for tar_path in glob.glob(os.path.join(ff,'*.tar')):
                urls.append(tar_path)        
        
        dataset = wds.WebDataset(urls,
                                nodesplitter=wds.split_by_node,
                                shardshuffle=True,
                                resampled=resampled,
                                cache_dir=cache_dir,)
        if train:
            dataset = dataset.shuffle(100)
        dataset = dataset.decode('rgb8').rename(jpg='jpg;jpeg;png')

        # Process the dataset
        dataset = dataset.compose(split_data)

        use_skimage_antialias = cfg.DATASETS.get('USE_SKIMAGE_ANTIALIAS', False)
        border_mode = {
            'constant': cv2.BORDER_CONSTANT,
            'replicate': cv2.BORDER_REPLICATE,
        }[cfg.DATASETS.get('BORDER_MODE', 'constant')]

        # Process the dataset further
        dataset = dataset.map(lambda x: BedlamDataset.process_webdataset_tar_item(x, train,
                                                        augm_config=cfg.DATASETS.CONFIG,
                                                        MEAN=MEAN, STD=STD, IMG_SIZE=IMG_SIZE,
                                                        BBOX_SHAPE=BBOX_SHAPE,
                                                        use_skimage_antialias=use_skimage_antialias,
                                                        border_mode=border_mode,
                                                        smpl=smpl
                                                        ))
        if epoch_size is not None:
            dataset = dataset.with_epoch(epoch_size)

        return dataset

    @staticmethod
    def process_webdataset_tar_item(item, train, 
                                    augm_config=None, 
                                    MEAN=DEFAULT_MEAN, 
                                    STD=DEFAULT_STD, 
                                    IMG_SIZE=DEFAULT_IMG_SIZE,
                                    BBOX_SHAPE=None,
                                    use_skimage_antialias=False,
                                    border_mode=cv2.BORDER_CONSTANT,
                                    smpl=None,
                                    ):
        # Read data from item
        key = item['__key__']
        image = item['jpg']
        data = item['data.pyd']
        mask = item['mask']
        gender = data['gender']
        keypoints_2d_full = data['gtkps'].astype(np.float32)
        center = data['center']
        scale = data['scale']
        body_pose = data['pose_cam'].astype(np.float32)
        betas = data['shape'].astype(np.float32)
        cam_trans = data['trans_cam'].astype(np.float32) + data['cam_ext'][:3, 3].astype(np.float32)

        # Crop image and (possibly) perform data augmentation
        if 'closeup' in key:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            mask = np.transpose(mask,(1,0))

        if 'cam_int' in data.keys():
            CAM_INT = np.array(data['cam_int']).astype(np.float32)
        else:
            img_w = img_size[1]
            img_h = img_size[0]
            fl = (img_w * img_w + img_h * img_h) ** 0.5
            CAM_INT = np.array([[fl, 0, img_w/2], [0, fl, img_h / 2], [0, 0, 1]]).astype(np.float32)
        # Process data
        
        center_x = center[0]
        center_y = center[1]
        bbox_size = expand_to_aspect_ratio(scale*200, target_aspect_ratio=BBOX_SHAPE).max()
        if bbox_size < 1:
            breakpoint()
        smpl_params = {'global_orient': body_pose[:3],
                    'body_pose': body_pose[3:24*3],
                    'betas': betas[:10]}
        smpl_params_in = {'global_orient': axis_angle_to_matrix(torch.from_numpy(body_pose[:3][None])),
                    'body_pose': axis_angle_to_matrix(torch.from_numpy(body_pose[3:24*3][None]).reshape(1,-1,3)),
                    'betas': torch.from_numpy(betas[:10][None])}
        smpl_output = smpl(**{k: v for k,v in smpl_params_in.items()}, pose2rot=False)
        keypoints_3d = smpl_output.joints[0]
        keypoints_2d = project(keypoints_3d, torch.from_numpy(cam_trans), CAM_INT)
        keypoints_3d = torch.cat([keypoints_3d, torch.ones(keypoints_3d.shape[0], 1)], 1).numpy()
        orig_keypoints_2d = keypoints_2d.copy()

        has_smpl_params = {'global_orient': np.array(1.0),
                        'body_pose': np.array(1.0),
                        'betas': np.array(1.0)}

        smpl_params_is_axis_angle = {'global_orient': True,
                                    'body_pose': True,
                                    'betas': False}

        augm_config = copy.deepcopy(augm_config)
        
        img_rgba = np.concatenate([image, mask.astype(np.uint8)[:,:,None]*255], axis=2)
        
        img_patch_rgba=None
        img_patch_rgba, \
        keypoints_2d, \
        keypoints_3d, \
        smpl_params, \
        has_smpl_params, \
        img_size, cx, cy, bbox_w, bbox_h, trans = get_example(img_rgba,
                                                                center_x, center_y,
                                                                bbox_size, bbox_size,
                                                                keypoints_2d, keypoints_3d,
                                                                smpl_params, has_smpl_params,
                                                                FLIP_KEYPOINT_PERMUTATION,
                                                                IMG_SIZE, IMG_SIZE,
                                                                MEAN, STD, train, augm_config,
                                                                is_bgr=False, return_trans=True,
                                                                use_skimage_antialias=use_skimage_antialias,
                                                                border_mode=border_mode, return_newbox=True,
                                                                )
        new_center = np.array([cx, cy])
        img_patch = img_patch_rgba[:3, :, :]

        item = {}
        img_patch = img_patch_rgba[:3,:,:]
        mask_patch = (img_patch_rgba[3,:,:] / 255.0).clip(0,1)
        if (mask_patch < 0.5).all():
            mask_patch = np.ones_like(mask_patch)

        item['img'] = img_patch
        item['mask'] = mask_patch
        item['keypoints_2d'] = keypoints_2d.astype(np.float32)
        item['keypoints_3d'] = keypoints_3d.astype(np.float32)
        item['orig_keypoints_2d'] = orig_keypoints_2d
        item['box_center'] = new_center
        item['box_size'] = bbox_w
        item['img_size'] = 1.0 * img_size[::-1].copy()
        item['smpl_params'] = smpl_params
        item['has_smpl_params'] = has_smpl_params
        item['smpl_params_is_axis_angle'] = smpl_params_is_axis_angle
        item['_scale'] = np.array([scale, scale]).astype(np.float32)
        item['_trans'] = trans
        item['imgname'] = key
        item['dataset'] = 'BEDLAM'
        return item
