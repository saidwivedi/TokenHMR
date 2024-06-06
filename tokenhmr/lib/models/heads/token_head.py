import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops

from ...utils.geometry import rot6d_to_rotmat, aa_to_rotmat
from ..components.pose_transformer import TransformerDecoder
from .token_classifier import TokenClassfier

def frozing_module(module):
    module.eval()
    for name, params in module.named_parameters():
        params.requires_grad = False

class SMPLTokenDecoderHead(nn.Module):
    """ Cross-attention based SMPL Transformer decoder
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.joint_rep_type = cfg.MODEL.SMPL_HEAD.get('JOINT_REP', '6d')
        self.joint_rep_dim = {'6d': 6, 'aa': 3}[self.joint_rep_type]
        self.tokenizer_type = cfg.MODEL.SMPL_HEAD.TOKENIZER.get('TOKENIZER_TYPE', 'parts')
        self.token_code_dim = cfg.MODEL.SMPL_HEAD.TOKENIZER.get('TOKEN_CODE_DIM', None)
        npose = self.joint_rep_dim * (cfg.SMPL.NUM_BODY_JOINTS + 1)
        self.npose = npose
        self.input_is_mean_shape = cfg.MODEL.SMPL_HEAD.get('TRANSFORMER_INPUT', 'zero') == 'mean_shape'
        transformer_args = dict(
            num_tokens=1,
            token_dim=(npose + 10 + 3) if self.input_is_mean_shape else 1,
            dim=1024,
        )
        transformer_args = dict(**transformer_args, **dict(cfg.MODEL.SMPL_HEAD.TRANSFORMER_DECODER))
        self.transformer = TransformerDecoder(
            **transformer_args
        )
        dim=transformer_args['dim']
        self.decpose_grot = nn.Linear(dim, 6)
        self.decshape = nn.Linear(dim, 10)
        self.deccam = nn.Linear(dim, 3)
        self.decpose_hands = nn.Linear(dim, 6*2)

        self.decpose = TokenClassfier(dim, tokenizer_checkpoint_path=cfg.MODEL.TOKENIZER_CHECKPOINT_PATH, \
                                    token_num=cfg.MODEL.SMPL_HEAD.TOKENIZER.TOKEN_NUM, \
                                    token_class_num=cfg.MODEL.SMPL_HEAD.TOKENIZER.TOKEN_CLASS_NUM,\
                                    token_code_dim=self.token_code_dim,\
                                    tokenizer_type=self.tokenizer_type)

        if cfg.MODEL.SMPL_HEAD.get('INIT_DECODER_XAVIER', False):
            # True by default in MLP. False by default in Transformer
            nn.init.xavier_uniform_(self.decpose_grot.weight, gain=0.01)
            nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
            nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        mean_params = np.load(cfg.SMPL.MEAN_PARAMS)
        init_body_pose = torch.from_numpy(mean_params['pose'].astype(np.float32)).unsqueeze(0)
        init_betas = torch.from_numpy(mean_params['shape'].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam'].astype(np.float32)).unsqueeze(0)
        self.register_buffer('init_body_pose', init_body_pose)
        self.register_buffer('init_betas', init_betas)
        self.register_buffer('init_cam', init_cam)

    def forward(self, x, **kwargs):

        batch_size = x.shape[0]
        # vit pretrained backbone is channel-first. Change to token-first
        x = einops.rearrange(x, 'b c h w -> b (h w) c')

        init_body_pose = self.init_body_pose.expand(batch_size, -1)
        init_betas = self.init_betas.expand(batch_size, -1)
        init_cam = self.init_cam.expand(batch_size, -1)

        # TODO: Convert init_body_pose to aa rep if needed
        if self.joint_rep_type == 'aa':
            raise NotImplementedError

        pred_body_pose = init_body_pose
        pred_betas = init_betas
        pred_cam = init_cam
        pred_body_pose_list = []
        pred_betas_list = []
        pred_cam_list = []
        cls_logits_softmax_list = []
        for i in range(self.cfg.MODEL.SMPL_HEAD.get('IEF_ITERS', 1)):
            # Input token to transformer is zero token
            if self.input_is_mean_shape:
                token = torch.cat([pred_body_pose, pred_betas, pred_cam], dim=1)[:,None,:]
            else:
                token = torch.zeros(batch_size, 1, 1).to(x.device)

            token = token.to(x.dtype)
            # Pass through transformer
            token_out = self.transformer(token, context=x)
            token_out = token_out.squeeze(1) # (B, C) 1024

            # Readout from token_out
            pred_grot = self.decpose_grot(token_out)
            pred_bpose, cls_logits_softmax = self.decpose(token_out) 
            pred_handpose = self.decpose_hands(token_out)
            # pred_body_pose = torch.cat([pred_grot,pred_bpose,torch.zeros_like(pred_grot),torch.zeros_like(pred_grot)], -1) + pred_body_pose
            pred_body_pose = torch.cat([pred_grot,pred_bpose,pred_handpose], -1) + pred_body_pose
            pred_betas = self.decshape(token_out) + pred_betas
            pred_cam = self.deccam(token_out) + pred_cam

            cls_logits_softmax_list.append(cls_logits_softmax)
            pred_body_pose_list.append(pred_body_pose)
            pred_betas_list.append(pred_betas)
            pred_cam_list.append(pred_cam)

        # Convert self.joint_rep_type -> rotmat
        joint_conversion_fn = {
            '6d': rot6d_to_rotmat,
            'aa': lambda x: aa_to_rotmat(x.view(-1, 3).contiguous())
        }[self.joint_rep_type]

        pred_smpl_params_list = {}
        pred_smpl_params_list['body_pose'] = torch.cat([joint_conversion_fn(pbp).view(batch_size, -1, 3, 3)[:, 1:, :, :] for pbp in pred_body_pose_list], dim=0)
        pred_smpl_params_list['betas'] = torch.cat(pred_betas_list, dim=0)
        pred_smpl_params_list['cam'] = torch.cat(pred_cam_list, dim=0)
        pred_smpl_params_list['cls_logits_softmax'] = torch.cat(cls_logits_softmax_list, dim=0)
        pred_body_pose = joint_conversion_fn(pred_body_pose).view(batch_size, self.cfg.SMPL.NUM_BODY_JOINTS+1, 3, 3)

        pred_smpl_params = {'global_orient': pred_body_pose[:, [0]],
                            'body_pose': pred_body_pose[:, 1:],
                            'betas': pred_betas}
        return pred_smpl_params, pred_cam, pred_smpl_params_list
