"""
Code adapted from: https://github.com/akanazawa/hmr/blob/master/src/benchmark/eval_util.py
"""
import cv2
import torch
import numpy as np
from typing import Optional, Dict, List, Tuple
from .rotation_utils import axis_angle_to_matrix, matrix_to_axis_angle, matrix_to_euler_angles
# keypoints 2D 44 = 25 Openpose + 14 lsp + 5?
JOINT44_NAMES = [
# 25 OpenPose joints (in the order provided by OpenPose)
'Nose',
'Neck',
'RShoulder',
'RElbow',
'RWrist',
'LShoulder',
'LElbow',
'LWrist',
'MidHip',
'RHip',
'RKnee',
'RAnkle',
'LHip',
'LKnee',
'LAnkle',
'REye',
'LEye',
'REar',
'LEar',
'LBigToe',
'LSmallToe',
'LHeel',
'RBigToe',
'RSmallToe',
'RHeel',
# 14 LSP joints
'R_Ankle',
'R_Knee',
'R_Hip',
'L_Hip',
'L_Knee',
'L_Ankle',
'R_Wrist',
'R_Elbow',
'R_Shoulder',
'L_Shoulder',
'L_Elbow',
'L_Wrist',
'Neck_LSP',
'HeadTop_LSP',

'Pelvis_MPII',
'Thorax_MPII',
'Spine_H36M',
'Jaw_H36M',
'Head_H36M',
'Nose_other',
]

def compute_similarity_transform(S1: torch.Tensor, S2: torch.Tensor) -> torch.Tensor:
    """
    Computes a similarity transform (sR, t) in a batched way that takes
    a set of 3D points S1 (B, N, 3) closest to a set of 3D points S2 (B, N, 3),
    where R is a 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    Args:
        S1 (torch.Tensor): First set of points of shape (B, N, 3).
        S2 (torch.Tensor): Second set of points of shape (B, N, 3).
    Returns:
        (torch.Tensor): The first set of points after applying the similarity transformation.
    """

    # Ensure that the input is of type float32
    S1 = S1.to(torch.float32)
    S2 = S2.to(torch.float32)

    batch_size = S1.shape[0]
    S1 = S1.permute(0, 2, 1)
    S2 = S2.permute(0, 2, 1)
    # 1. Remove mean.
    mu1 = S1.mean(dim=2, keepdim=True)
    mu2 = S2.mean(dim=2, keepdim=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = (X1**2).sum(dim=(1,2))

    # 3. The outer product of X1 and X2.
    K = torch.matmul(X1, X2.permute(0, 2, 1))

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are singular vectors of K.
    U, s, V = torch.svd(K)
    Vh = V.permute(0, 2, 1)

    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[1], device=U.device).unsqueeze(0).repeat(batch_size, 1, 1)
    Z[:, -1, -1] *= torch.sign(torch.linalg.det(torch.matmul(U, Vh)))

    # Construct R.
    R = torch.matmul(torch.matmul(V, Z), U.permute(0, 2, 1))

    # 5. Recover scale.
    trace = torch.matmul(R, K).diagonal(offset=0, dim1=-1, dim2=-2).sum(dim=-1)
    scale = (trace / var1).unsqueeze(dim=-1).unsqueeze(dim=-1)

    # 6. Recover translation.
    t = mu2 - scale*torch.matmul(R, mu1)

    # 7. Error:
    S1_hat = scale*torch.matmul(R, S1) + t

    return S1_hat.permute(0, 2, 1)

def reconstruction_error(S1, S2) -> np.array:
    """
    Computes the mean Euclidean distance of 2 set of points S1, S2 after performing Procrustes alignment.
    Args:
        S1 (torch.Tensor): First set of points of shape (B, N, 3).
        S2 (torch.Tensor): Second set of points of shape (B, N, 3).
    Returns:
        (np.array): Reconstruction error.
    """
    S1_hat = compute_similarity_transform(S1, S2)
    re = torch.sqrt( ((S1_hat - S2)** 2).sum(dim=-1)).mean(dim=-1)
    return re

def eval_pose(pred_joints, gt_joints) -> Tuple[np.array, np.array]:
    """
    Compute joint errors in mm before and after Procrustes alignment.
    Args:
        pred_joints (torch.Tensor): Predicted 3D joints of shape (B, N, 3).
        gt_joints (torch.Tensor): Ground truth 3D joints of shape (B, N, 3).
    Returns:
        Tuple[np.array, np.array]: Joint errors in mm before and after alignment.
    """
    # Absolute error (MPJPE)
    mpjpe = torch.sqrt(((pred_joints - gt_joints) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()

    # Reconstruction_error
    r_error = reconstruction_error(pred_joints, gt_joints).cpu().numpy()
    return 1000 * mpjpe, 1000 * r_error

class Evaluator:

    def __init__(self,
                 dataset_length: int,
                 keypoint_list: List,
                 pelvis_ind: int,
                 metrics: List = ['mode_mpjpe', 'mode_re', 'model_pve'],
                 J_regressor_24_SMPL = None,
                 dataset=''):
        """
        Class used for evaluating trained models on different 3D pose datasets.
        Args:
            dataset_length (int): Total dataset length.
            keypoint_list [List]: List of keypoints used for evaluation.
            pelvis_ind (int): Index of pelvis keypoint; used for aligning the predictions and ground truth.
            metrics [List]: List of evaluation metrics to record.
        """
        self.dataset_length = dataset_length
        self.keypoint_list = keypoint_list
        self.pelvis_ind = pelvis_ind
        self.metrics = metrics
        self.J_regressor_24_SMPL = J_regressor_24_SMPL
        self.dataset = dataset
        for metric in self.metrics:
            setattr(self, metric, np.zeros((dataset_length,)))
        self.counter = 0

        self.imgnames = []

    def log(self):
        """
        Print current evaluation metrics
        """
        if self.counter == 0:
            print('Evaluation has not started')
            return
        print(f'{self.counter} / {self.dataset_length} samples')
        for metric in self.metrics:
            if metric in ['mode_mpjpe', 'mode_re', 'mode_pve']:
                unit = 'mm'
            else:
                unit = ''
            print(f'{metric}: {getattr(self, metric)[:self.counter].mean(0)} {unit}')
        print('***')

    def get_metrics_dict(self) -> Dict:
        """
        Returns:
            Dict: Dictionary of evaluation metrics.
        """
        d1 = {metric: getattr(self, metric)[:self.counter].mean() for metric in self.metrics}
        return d1
    
    def get_imgnames(self):
        return self.imgnames

    def __call__(self, output: Dict, batch: Dict):
        """
        Evaluate current batch.
        Args:
            output (Dict): Regression output.
            batch (Dict): Dictionary containing images and their corresponding annotations.
        """
        imgnames = batch['imgname']
        self.imgnames += imgnames
        if 'EMDB' in self.dataset:
            gt_vertices = batch['vertices']
            gt_keypoints_3d = torch.matmul(self.J_regressor_24_SMPL, gt_vertices)
            gt_pelvis = (gt_keypoints_3d[:, [1], :] + gt_keypoints_3d[:, [2], :]) / 2.0
            gt_keypoints_3d = gt_keypoints_3d - gt_pelvis
            gt_vertices = gt_vertices - gt_pelvis

            pred_vertices = output['pred_vertices']
            pred_keypoints_3d = torch.matmul(self.J_regressor_24_SMPL, pred_vertices)
            pred_pelvis = (pred_keypoints_3d[:, [1], :] + pred_keypoints_3d[:, [2], :]) / 2.0
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis
            pred_vertices = pred_vertices - pred_pelvis
            batch_size = pred_keypoints_3d.shape[0]
            num_samples = 1
        else:
            pred_keypoints_3d = output['pred_keypoints_3d'].detach()
            pred_keypoints_3d = pred_keypoints_3d[:,None,:,:]
            batch_size = pred_keypoints_3d.shape[0]
            num_samples = pred_keypoints_3d.shape[1]
            gt_keypoints_3d = batch['keypoints_3d'][:, :, :-1].unsqueeze(1).repeat(1, num_samples, 1, 1)
            gt_vertices = batch['vertices'][:,None,:,:]

            # Align predictions and ground truth such that the pelvis location is at the origin
            pred_pelvis = pred_keypoints_3d[:, :, [self.pelvis_ind]]
            gt_pelvis = gt_keypoints_3d[:, :, [self.pelvis_ind]]
            pred_keypoints_3d -= pred_pelvis
            gt_keypoints_3d -= gt_pelvis
            pred_keypoints_3d = pred_keypoints_3d.reshape(batch_size * num_samples, -1, 3)
            gt_keypoints_3d =  gt_keypoints_3d.reshape(batch_size * num_samples, -1 ,3)
            pred_vertices = output['pred_vertices'][:,None,:,:]
            gt_vertices = gt_vertices - gt_pelvis
            pred_vertices = pred_vertices - pred_pelvis

        mpjpe, re = eval_pose(pred_keypoints_3d[:, self.keypoint_list],gt_keypoints_3d[:, self.keypoint_list])

        if hasattr(self, 'mode_pve'):
            pve = torch.sqrt(((pred_vertices - gt_vertices) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy() * 1000.
            pve = pve.reshape(batch_size, num_samples)
        
        mpjpe = mpjpe.reshape(batch_size, num_samples)
        re = re.reshape(batch_size, num_samples)

        # The 0-th sample always corresponds to the mode
        if hasattr(self, 'mode_mpjpe'):
            mode_mpjpe = mpjpe[:, 0]
            self.mode_mpjpe[self.counter:self.counter+batch_size] = mode_mpjpe
        if hasattr(self, 'mode_re'):
            mode_re = re[:, 0]
            self.mode_re[self.counter:self.counter+batch_size] = mode_re
        if hasattr(self, 'mode_pve'):
            mode_pve = pve[:, 0]
            self.mode_pve[self.counter:self.counter+batch_size] = mode_pve

        self.counter += batch_size

        if hasattr(self, 'mode_mpjpe') and hasattr(self, 'mode_re'):
            return {
                'mode_mpjpe': mode_mpjpe,
                'mode_re': mode_re,
            }
        if hasattr(self, 'mode_mpjpe') and hasattr(self, 'mode_re') and hasattr(self, 'mode_pve'):
            return {
                'mode_mpjpe': mode_mpjpe,
                'mode_re': mode_re,
                'mode_pve': mode_pve,
            }

