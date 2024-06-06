import torch
import numpy as np
import pytorch_lightning as pl
from typing import Dict, Tuple
from torch.autograd import Variable as V
from datetime import datetime

from yacs.config import CfgNode

import sys, os
sys.path.append(os.path.join(__file__.replace(os.path.basename(__file__), ''), '..', '..', '..'))

from ..utils import SkeletonRenderer, MeshRenderer
from ..utils.geometry import aa_to_rotmat, perspective_projection
from ..utils.pylogger import get_pylogger
from ..utils.misc import load_pretrained
from .backbones import create_backbone
from .heads import build_smpl_head

from .discriminator import Discriminator
from .losses import Keypoint3DLoss, Keypoint2DLoss, ParameterLoss, VerticesLoss, \
    TokenLoss, Keypoint2DLossPCKT, Keypoint3DLossPCKT, ParameterLossPCKT
from .losses import joint_angle_error, angle_valid_thresh, kp2D_err_valid_thresh
from .smpl_wrapper import SMPL

log = get_pylogger(__name__)
H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]
H36M_TO_J14 = H36M_TO_J17[:14]


class TokenHMR(pl.LightningModule):

    def __init__(self, cfg: CfgNode, init_renderer: bool = True, is_train_state = False, is_demo = False):
        """
        Setup TokenHMR model
        Args:
            cfg (CfgNode): Config file as a yacs CfgNode
        """
        super().__init__()

        # Save hyperparameters
        self.save_hyperparameters(logger=False, ignore=['init_renderer'])
        self.cfg = cfg
        self.is_demo = is_demo

        # Model in evaluation state
        if not is_train_state:
            print(f'Model is in evaluation state')
            self.backbone, self.smpl_head = create_backbone(cfg, load_weights=False), build_smpl_head(cfg)
            self.backbone, self.smpl_head = load_pretrained(cfg, self.backbone, self.smpl_head, is_train_state)

        # Model in training state
        else:
            print(f'Model is in training state')

            # Create backbone feature extractor
            self.backbone = create_backbone(cfg)

            # Create SMPL head
            self.smpl_head = build_smpl_head(cfg)

            # Create discriminator
            if self.cfg.LOSS_WEIGHTS.ADVERSARIAL > 0:
                self.discriminator = Discriminator()

            # Define loss functions
            if self.cfg.MODEL.LOOSE_SUP:
                self.keypoint_3d_loss = Keypoint3DLossPCKT(loss_type='l1')
                self.keypoint_2d_loss = Keypoint2DLossPCKT(loss_type='l1')
                self.smpl_parameter_loss = ParameterLossPCKT()
            else:
                self.keypoint_3d_loss = Keypoint3DLoss(loss_type='l1')
                self.keypoint_2d_loss = Keypoint2DLoss(loss_type='l1')
                self.smpl_parameter_loss = ParameterLoss()
            
            self.vertices_loss = VerticesLoss(loss_type='l1')
            self.token_classification_loss = TokenLoss()

            if self.cfg.MODEL.SMPL_HEAD.TYPE == 'token':
                if self.cfg.MODEL.FROZEN_LEARNED:
                    self.frozen_backbone()

        # Instantiate SMPL model
        smpl_cfg = {k.lower(): v for k,v in dict(cfg.SMPL).items()}
        self.smpl = SMPL(**smpl_cfg)

        # Buffer that shows whetheer we need to initialize ActNorm layers
        self.register_buffer('initialized', torch.tensor(False))
        # Setup renderer for visualization
        if init_renderer:
            self.renderer = SkeletonRenderer(self.cfg)
            self.mesh_renderer = MeshRenderer(self.cfg, faces=self.smpl.faces)
        else:
            self.renderer = None
            self.mesh_renderer = None

        # Disable automatic optimization since we use adversarial training
        self.automatic_optimization = False
        if is_train_state:
            self.validation_step_outputs = []
            self.best_validation_loss = cfg.MODEL.get('VAL_LOSS_SAVE_THRESH', 5.0)
            self.best_MPJPE, self.MPJPE = 75.0, []
            self.best_PAMPJPE, self.PAMPJPE = 50.0, []
            self.best_PVE, self.PVE = 87.0, []
            
    def get_parameters(self):
        all_params = list(self.smpl_head.parameters())
        all_params += list(self.backbone.parameters())
        return all_params

    def frozen_backbone(self):
        self.backbone.eval()
        for name, params in self.backbone.named_parameters():
            params.requires_grad = False

    def configure_optimizers(self) -> Tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
        """
        Setup model and distriminator Optimizers
        Returns:
            Tuple[torch.optim.Optimizer, torch.optim.Optimizer]: Model and discriminator optimizers
        """
        param_groups = [{'params': filter(lambda p: p.requires_grad, self.get_parameters()), 'lr': self.cfg.TRAIN.LR}]

        optimizer = torch.optim.AdamW(params=param_groups,
                                        # lr=self.cfg.TRAIN.LR,
                                        weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)
        if self.cfg.LOSS_WEIGHTS.ADVERSARIAL > 0:
            optimizer_disc = torch.optim.AdamW(params=self.discriminator.parameters(),
                                                lr=self.cfg.TRAIN.LR,
                                                weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)

            return optimizer, optimizer_disc
        return optimizer

    def forward_step(self, batch: Dict, train: bool = False) -> Dict:
        """
        Run a forward step of the network
        Args:
            batch (Dict): Dictionary containing batch data
            train (bool): Flag indicating whether it is training or validation mode
        Returns:
            Dict: Dictionary containing the regression output
        """

        # Use RGB image as input
        x = batch['img']
        batch_size = x.shape[0]

        # Compute conditioning features using the backbone
        # if using ViT backbone, we need to use a different aspect ratio
        conditioning_feats = self.backbone(x)

        pred_smpl_params, pred_cam, pred_smpl_params_list = self.smpl_head(conditioning_feats)

        # Store useful regression outputs to the output dict
        output = {}
        if self.cfg.MODEL.SMPL_HEAD.TYPE == 'token':
            output['cls_logits_softmax'] = pred_smpl_params_list['cls_logits_softmax']
        output['pred_cam'] = pred_cam
        output['pred_smpl_params'] = {k: v.clone() for k,v in pred_smpl_params.items()}

        # Compute camera translation
        device = pred_smpl_params['body_pose'].device
        dtype = pred_smpl_params['body_pose'].dtype
        focal_length = self.cfg.EXTRA.FOCAL_LENGTH * torch.ones(batch_size, 2, device=device, dtype=dtype)
        pred_cam_t = torch.stack([pred_cam[:, 1],
                                  pred_cam[:, 2],
                                  2*focal_length[:, 0]/(self.cfg.MODEL.IMAGE_SIZE * pred_cam[:, 0] +1e-9)],dim=-1)
        output['pred_cam_t'] = pred_cam_t
        output['focal_length'] = focal_length

        # Compute model vertices, joints and the projected joints
        pred_smpl_params['global_orient'] = pred_smpl_params['global_orient'].reshape(batch_size, -1, 3, 3)
        pred_smpl_params['body_pose'] = pred_smpl_params['body_pose'].reshape(batch_size, -1, 3, 3)
        pred_smpl_params['betas'] = pred_smpl_params['betas'].reshape(batch_size, -1)
        smpl_output = self.smpl(**{k: v for k,v in pred_smpl_params.items()}, pose2rot=False)
        pred_keypoints_3d = smpl_output.joints
        pred_vertices = smpl_output.vertices
        output['pred_keypoints_3d'] = pred_keypoints_3d.reshape(batch_size, -1, 3)
        output['pred_vertices'] = pred_vertices.reshape(batch_size, -1, 3)
        pred_cam_t = pred_cam_t.reshape(-1, 3)
        focal_length = focal_length.reshape(-1, 2)
        pred_keypoints_2d = perspective_projection(pred_keypoints_3d,
                                                   translation=pred_cam_t,
                                                   focal_length=focal_length / self.cfg.MODEL.IMAGE_SIZE)

        output['pred_keypoints_2d'] = pred_keypoints_2d.reshape(batch_size, -1, 2)
        return output

    def compute_loss(self, batch: Dict, output: Dict, train: bool = True) -> torch.Tensor:
        """
        Compute losses given the input batch and the regression output
        Args:
            batch (Dict): Dictionary containing batch data
            output (Dict): Dictionary containing the regression output
            train (bool): Flag indicating whether it is training or validation mode
        Returns:
            torch.Tensor : Total loss for current batch
        """

        pred_smpl_params = output['pred_smpl_params']
        pred_keypoints_2d = output['pred_keypoints_2d']
        pred_keypoints_3d = output['pred_keypoints_3d']

        batch_size = pred_smpl_params['body_pose'].shape[0]

        # Get annotations
        gt_keypoints_2d = batch['keypoints_2d']
        gt_keypoints_3d = batch['keypoints_3d']
        gt_smpl_params = batch['smpl_params']
        has_smpl_params = batch['has_smpl_params']
        is_axis_angle = batch['smpl_params_is_axis_angle']

        if self.cfg.MODEL.LOOSE_SUP and train:
            dataset_names = batch['dataset']
            batch_size = pred_keypoints_2d.shape[0]

            kp2D_err = gt_keypoints_2d[:, :, -1] * torch.nn.functional.mse_loss(
                pred_keypoints_2d, gt_keypoints_2d[:, :, :-1], reduction='none').sum(dim=2)
            valid_mask2D = kp2D_err > kp2D_err_valid_thresh[None].repeat(batch_size,1).to(kp2D_err.device)
            weak_mask = gt_keypoints_2d[:, :, -1] * (~valid_mask2D).float()
            # Compute 3D keypoint loss
            gt_keypoints_2d[:,:,-1] = gt_keypoints_2d[:,:,-1] * valid_mask2D
            loss_keypoints_2d = self.keypoint_2d_loss(pred_keypoints_2d, gt_keypoints_2d, weak_mask, self.cfg.MODEL.LOOSE_WEIGHT)

            valid_3D_mask = torch.Tensor([name in ['H36M-TRAIN-WMASK', 'BEDLAM'] for name in dataset_names]).float().to(gt_keypoints_3d.device)
            gt_keypoints_3d[:, :, -1] = gt_keypoints_3d[:, :, -1] * ((valid_3D_mask.unsqueeze(-1) + gt_keypoints_2d[:,:,-1]) > 0.5)
            loss_keypoints_3d = self.keypoint_3d_loss(pred_keypoints_3d, gt_keypoints_3d, pelvis_id=25+14)

            # Compute loss on SMPL parameters
            loss_smpl_params = {}
            for k, pred in pred_smpl_params.items():
                gt = gt_smpl_params[k].view(batch_size, -1)
                if is_axis_angle[k].all():
                    gt = aa_to_rotmat(gt.reshape(-1, 3)).view(batch_size, -1, 3, 3)
                has_gt = has_smpl_params[k]
                if k in ['betas']:
                    valid_mask3D = None
                    weak_mask = None
                    has_gt *= valid_3D_mask
                    #print(k, has_gt, valid_3D_mask)
                elif k in ['body_pose', 'global_orient']:
                    angle_error = joint_angle_error(pred, gt) 
                    valid_mask3D = angle_error > angle_valid_thresh[k][None].repeat(batch_size,1).to(angle_error.device)
                    valid_mask3D = (valid_mask3D * has_gt.unsqueeze(1) + valid_3D_mask.unsqueeze(1)).bool()
                    weak_mask = (~valid_mask3D * has_gt.unsqueeze(1)).float()
                    valid_mask3D = valid_mask3D.float()
                    #print(k, angle_error, valid_mask3D)
                loss_smpl_params[k] = self.smpl_parameter_loss(pred, gt, has_gt, valid_mask3D, weak_mask, self.cfg.MODEL.LOOSE_WEIGHT)
        else:
            # Compute 3D keypoint loss
            loss_keypoints_2d = self.keypoint_2d_loss(pred_keypoints_2d, gt_keypoints_2d)
            loss_keypoints_3d = self.keypoint_3d_loss(pred_keypoints_3d, gt_keypoints_3d, pelvis_id=25+14)

            # Compute loss on SMPL parameters
            loss_smpl_params = {}
            for k, pred in pred_smpl_params.items():
                gt = gt_smpl_params[k].view(batch_size, -1)
                if is_axis_angle[k].all():
                    gt = aa_to_rotmat(gt.reshape(-1, 3)).view(batch_size, -1, 3, 3)
                has_gt = has_smpl_params[k]
                loss_smpl_params[k] = self.smpl_parameter_loss(pred.reshape(batch_size, -1), gt.reshape(batch_size, -1), has_gt)

        loss = self.cfg.LOSS_WEIGHTS['KEYPOINTS_3D'] * loss_keypoints_3d +\
               self.cfg.LOSS_WEIGHTS['KEYPOINTS_2D'] * loss_keypoints_2d +\
               sum([loss_smpl_params[k] * self.cfg.LOSS_WEIGHTS[k.upper()] for k in loss_smpl_params])

        losses = dict(loss=loss.detach(),
                      loss_keypoints_2d=loss_keypoints_2d.detach(),
                      loss_keypoints_3d=loss_keypoints_3d.detach())

        for k, v in loss_smpl_params.items():
            losses['loss_' + k] = v.detach()

        output['losses'] = losses

        return loss

    # Tensoroboard logging should run from first rank only
    @pl.utilities.rank_zero.rank_zero_only
    def tensorboard_logging(self, batch: Dict, output: Dict, step_count: int, train: bool = True, write_to_summary_writer: bool = True) -> None:
        """
        Log results to Tensorboard
        Args:
            batch (Dict): Dictionary containing batch data
            output (Dict): Dictionary containing the regression output
            step_count (int): Global training step count
            train (bool): Flag indicating whether it is training or validation mode
        """

        mode = 'train' if train else 'val'
        batch_size = batch['keypoints_2d'].shape[0]
        images = batch['img']
        images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1,3,1,1)
        images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1,3,1,1)
        #images = 255*images.permute(0, 2, 3, 1).cpu().numpy()

        pred_vertices = output['pred_vertices'].detach().reshape(batch_size, -1, 3)
        focal_length = output['focal_length'].detach().reshape(batch_size, 2)
        gt_keypoints_2d = batch['keypoints_2d']
        losses = output['losses']
        pred_cam_t = output['pred_cam_t'].detach().reshape(batch_size, 3)
        pred_keypoints_2d = output['pred_keypoints_2d'].detach().reshape(batch_size, -1, 2)

        if write_to_summary_writer:
            summary_writer = self.logger.experiment
            for loss_name, val in losses.items():
                summary_writer.add_scalar(mode +'/' + loss_name, val.detach().item(), step_count)
        num_images = min(batch_size, self.cfg.EXTRA.NUM_LOG_IMAGES)

        if pred_vertices.dtype != torch.float32:
            pred_vertices = pred_vertices.float()
            pred_cam_t = pred_cam_t.float()
            images = images.float()
            pred_keypoints_2d = pred_keypoints_2d.float()
            gt_keypoints_2d = gt_keypoints_2d.float()
            focal_length = focal_length.float()
            
        predictions = self.mesh_renderer.visualize_tensorboard(pred_vertices[:num_images].cpu().numpy(),
                                                               pred_cam_t[:num_images].cpu().numpy(),
                                                               images[:num_images].cpu().numpy(),
                                                               pred_keypoints_2d[:num_images].cpu().numpy(),
                                                               gt_keypoints_2d[:num_images].cpu().numpy(),
                                                               focal_length=focal_length[:num_images].cpu().numpy())
        if write_to_summary_writer:
            summary_writer.add_image('%s/predictions' % mode, predictions, step_count)

        return predictions

    def forward(self, batch: Dict) -> Dict:
        """
        Run a forward step of the network in val mode
        Args:
            batch (Dict): Dictionary containing batch data
        Returns:
            Dict: Dictionary containing the regression output
        """
        return self.forward_step(batch, train=False)

    def training_step_discriminator(self, batch: Dict,
                                    body_pose: torch.Tensor,
                                    betas: torch.Tensor,
                                    optimizer: torch.optim.Optimizer) -> torch.Tensor:
        """
        Run a discriminator training step
        Args:
            batch (Dict): Dictionary containing mocap batch data
            body_pose (torch.Tensor): Regressed body pose from current step
            betas (torch.Tensor): Regressed betas from current step
            optimizer (torch.optim.Optimizer): Discriminator optimizer
        Returns:
            torch.Tensor: Discriminator loss
        """
        batch_size = body_pose.shape[0]
        gt_body_pose = batch['body_pose']
        gt_betas = batch['betas']
        gt_rotmat = aa_to_rotmat(gt_body_pose.view(-1,3)).view(batch_size, -1, 3, 3)
        disc_fake_out = self.discriminator(body_pose.detach(), betas.detach())
        loss_fake = ((disc_fake_out - 0.0) ** 2).sum() / batch_size
        disc_real_out = self.discriminator(gt_rotmat, gt_betas)
        loss_real = ((disc_real_out - 1.0) ** 2).sum() / batch_size
        loss_disc = loss_fake + loss_real
        loss = self.cfg.LOSS_WEIGHTS.ADVERSARIAL * loss_disc
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()
        return loss_disc.detach()

    def training_step(self, joint_batch: Dict, batch_idx: int) -> Dict:
        """
        Run a full training step
        Args:
            joint_batch (Dict): Dictionary containing image and mocap batch data
            batch_idx (int): Unused.
            batch_idx (torch.Tensor): Unused.
        Returns:
            Dict: Dictionary containing regression output.
        """
        batch = joint_batch['img']
        mocap_batch = joint_batch['mocap']
        optimizer = self.optimizers(use_pl_optimizer=True)
        if self.cfg.LOSS_WEIGHTS.ADVERSARIAL > 0:
            optimizer, optimizer_disc = optimizer

        batch_size = batch['img'].shape[0]
        output = self.forward_step(batch, train=True)
        pred_smpl_params = output['pred_smpl_params']
        if self.cfg.get('UPDATE_GT_SPIN', False):
            self.update_batch_gt_spin(batch, output)
        loss = self.compute_loss(batch, output, train=True)
        if self.cfg.LOSS_WEIGHTS.ADVERSARIAL > 0:
            disc_out = self.discriminator(pred_smpl_params['body_pose'].reshape(batch_size, -1), pred_smpl_params['betas'].reshape(batch_size, -1))
            loss_adv = ((disc_out - 1.0) ** 2).sum() / batch_size
            loss = loss + self.cfg.LOSS_WEIGHTS.ADVERSARIAL * loss_adv
            print('dis gan loss:'+'{:.2f}'.format(self.cfg.LOSS_WEIGHTS.ADVERSARIAL * loss_adv))

        # Error if Nan
        if torch.isnan(loss):
            raise ValueError('Loss is NaN')

        optimizer.zero_grad()
        self.manual_backward(loss)
        # Clip gradient
        if self.cfg.TRAIN.get('GRAD_CLIP_VAL', 0) > 0:
            gn = torch.nn.utils.clip_grad_norm_(self.get_parameters(), self.cfg.TRAIN.GRAD_CLIP_VAL, error_if_nonfinite=True)
            self.log('train/grad_norm', gn, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        optimizer.step()
        if self.cfg.LOSS_WEIGHTS.ADVERSARIAL > 0:
            loss_disc = self.training_step_discriminator(mocap_batch, pred_smpl_params['body_pose'].reshape(batch_size, -1), pred_smpl_params['betas'].reshape(batch_size, -1), optimizer_disc)
            output['losses']['loss_gen'] = loss_adv
            output['losses']['loss_disc'] = loss_disc

        if self.global_step > 0 and self.global_step % self.cfg.GENERAL.LOG_STEPS == 0:
            self.tensorboard_logging(batch, output, self.global_step, train=True)

        if self.global_step % 6 == 0:
            self.log('train/loss', output['losses']['loss'], on_step=True, on_epoch=True, prog_bar=True, logger=False)

        return output

    def validation_step(self, batch: Dict, batch_idx: int, dataloader_idx=0) -> Dict:
        """
        Run a validation step and log to Tensorboard
        Args:
            batch (Dict): Dictionary containing batch data
            batch_idx (int): Unused.
        Returns:
            Dict: Dictionary containing regression output.
        """
        output = self.forward_step(batch, train=False)
        loss = self.compute_loss(batch, output, train=False)
        output['loss'] = loss
        self.validation_step_outputs.append(loss)

        if self.global_step > 0 and batch_idx % self.cfg.GENERAL.LOG_STEPS == 0:
            self.tensorboard_logging(batch, output, self.global_step, train=False)

        output['loss'] = loss

        return output