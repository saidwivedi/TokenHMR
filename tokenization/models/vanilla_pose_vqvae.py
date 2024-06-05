import re
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from .resnet import Resnet1D
from .quantize_cnn import QuantizeEMAReset
from .rotation_utils import matrix_to_rotation_6d, rotation_6d_to_matrix, matrix_to_axis_angle
from smplx import SMPLHLayer, SMPLXLayer

smpl_type='smplh'
import os
current_dir = os.path.dirname(os.path.realpath(__file__))
body_model_path = os.path.join(current_dir, '..', '..', 'data/body_models', smpl_type)
body_model = eval(f'{smpl_type.upper()}Layer')(body_model_path, num_betas=10, ext='pkl')
body_model = body_model.cuda() if torch.cuda.is_available() else body_model

def step_multiplier_mapping():
    return {
        0: 1e-2, 1: 5e-2, 2: 1e-1, 3: 1e-1, 4: 5e-1, 5: 5e-1
    }

def prepare_statedict(model, full_state_dict, partname, ignore_partname=' '):
    part_statedict = {}
    new_part_statedict = OrderedDict()

    # Load only the part given by sel_partname
    for key in full_state_dict.keys():
        if key.startswith(f'{partname}') and ignore_partname not in key:
            part_statedict[key] = full_state_dict[key]

    # Replace mismatch names
    for name, param in part_statedict.items():
        if re.match(f'^{partname}', name):
            name = name.replace(f'{partname}.', '', 1)
        new_part_statedict[name] = param

    model.load_state_dict(new_part_statedict, strict=True)
    return model

class PoseSPEncoderV1(nn.Module):
    def __init__(self,
                 rot_type = 'rotmat',
                 output_emb_width = 512,
                 down_t = 1,
                 stride_t = 2,
                 token_size_mul = 1,
                 width = 512,
                 depth = 2,
                 input_dim = 9,
                 dilation_growth_rate = 3,
                 inp_preprocess = True,
                 add_noise = False):
        super(PoseSPEncoderV1, self).__init__()

        encoder_layers = []
        num_joints = 21
        filter_t, pad_t = stride_t * 2, stride_t // 2
        self.inp_preprocess = inp_preprocess
        self.add_noise = add_noise
        self.step_multiplier_mapping = step_multiplier_mapping()
        if self.add_noise:
            from utils.skeleton import get_smplx_body_parts
            self.smplx_body_parts = get_smplx_body_parts()
        encoder_layers.append(nn.Conv1d(input_dim, width, 3, 1, 1))
        encoder_layers.append(nn.ReLU())

        # Make num of tokens in multiple of 10
        encoder_layers.append(nn.Upsample(((num_joints*2)//10)*10))
        encoder_layers.append(nn.Conv1d(width, width, 3, 1, 1))
        encoder_layers.append(nn.ReLU())
        
        for _ in range(token_size_mul-1):
            encoder_layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
            encoder_layers.append(nn.Conv1d(width, width, 3, 1, 1))
            encoder_layers.append(nn.ReLU())

        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
                Resnet1D(width, depth, dilation_growth_rate, activation='relu', norm=False),
            )
            encoder_layers.append(block)
        
        encoder_layers.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        self.encoder = nn.Sequential(*encoder_layers)

    def preprocess(self, x):
        # (bs, num_joints, 3, 3) -> (bs, num_joints, 9) -> (bs, 9, num_joints)
        x = x.view(x.shape[0], x.shape[1], -1)
        x = x.permute(0,2,1)
        return x

    def forward(self, x, global_step=None):
        if self.add_noise and global_step is not None:
            step = global_step // 5000
            noise_multiplier = float(self.step_multiplier_mapping[step]) if step <=5 else 0.5
            batch_size = x.shape[0]
            noised_samples = np.random.randint(low=0, high=batch_size-1, size=batch_size//2)
            mask_part = np.random.randint(len(self.smplx_body_parts.keys()))
            masked_joints = self.smplx_body_parts[mask_part]
            x[noised_samples][:,masked_joints] += (torch.cuda.FloatTensor(1).uniform_() * noise_multiplier)
        if self.inp_preprocess:
            x = self.preprocess(x)
        x = self.encoder(x)
        # for layer in self.encoder:
        #     print(f'{layer} --> {x.shape} --> {layer(x).shape}')
        #     x = layer(x)
        return x

class PoseSPDecoderV1(nn.Module):
    def __init__(self,
                 rot_type='rotmat',
                 output_emb_width = 512,
                 down_t = 1,
                 width = 512,
                 depth = 2,
                 token_size_div = 1,
                 num_tokens = 10,
                 dilation_growth_rate = 3,
                 num_joints=21,
                 output_dim = 6,
                 mesh_inference = True,
                 out_postprocess = True):
        super(PoseSPDecoderV1, self).__init__()

        decoder_layers = []
        self.rot_type = rot_type
        self.num_joints = num_joints
        self.mesh_inference = mesh_inference
        self.out_postprocess = out_postprocess

        decoder_layers.append(nn.Conv1d(output_emb_width, width, 3, 1, 1))
        decoder_layers.append(nn.ReLU())
        
        print(f'Num of tokens --> {num_tokens}')
        for i in list(np.linspace(self.num_joints, num_tokens, token_size_div, endpoint=False, dtype=int)[::-1]):
            decoder_layers.append(nn.Upsample(i))
            decoder_layers.append(nn.Conv1d(width, width, 3, 1, 1))
            decoder_layers.append(nn.ReLU())

        for i in range(down_t):
            out_dim = width
            block = nn.Sequential(
                Resnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation='relu', norm=False),
                nn.Conv1d(width, out_dim, 3, 1, 1)
            )
            decoder_layers.append(block)

        decoder_layers.append(nn.Conv1d(width, output_dim, 3, 1, 1))

        self.decoder = nn.Sequential(*decoder_layers)

    def postprocess(self, x):
        # (bs, 6, num_joints) -> (bs, num_joints, 6)
        x = x.permute(0,2,1)
        return x

    def forward(self, x):
        output = {}
        batch_size = x.shape[0]

        x = self.decoder(x)
        
        if not self.out_postprocess:
            return x
        pred_pose = self.postprocess(x)
        pred_pose_6d, pred_pose_rotmat = None, None
        if self.rot_type == 'rot6d':
            pred_pose_6d = pred_pose
            pred_pose_rotmat = rotation_6d_to_matrix(pred_pose.reshape(-1, 6)).view(batch_size, self.num_joints, 3, 3)
        elif self.rot_type == 'rotmat':
            NotImplementedError()
        
        output.update({
            'pred_pose_body_6d': pred_pose_6d,
            'pred_pose_body_rotmat': pred_pose_rotmat,
        })
        
        if self.mesh_inference:
            pred_pose_aa = matrix_to_axis_angle(pred_pose_rotmat.view(-1, 3, 3)).view(batch_size, 3*self.num_joints)
            pred_body_mesh = body_model(body_pose=pred_pose_rotmat)

            output.update({
                'pred_pose_body_aa': pred_pose_aa,
                'pred_body_mesh': pred_body_mesh,
                'pred_body_vertices': pred_body_mesh.vertices,
                'pred_body_joints': pred_body_mesh.joints
            })
        
        return output

class VanillaTokenizer(nn.Module):
    def __init__(self, arch_params=None, input_joint_dim=6, output_joint_dim=6, mesh_inference=True, add_noise=False):
        
        super().__init__()
        self.num_joints = 21
        self.code_dim = arch_params.CODE_DIM
        self.num_code = arch_params.NB_CODE
        self.down_t = arch_params.DOWN_T
        self.depth = arch_params.DEPTH
        self.width = arch_params.WIDTH
        self.quant = arch_params.QUANTIZER
        self.rot_type = arch_params.ROT_TYPE
        self.dilation_growth_rate = arch_params.DILATION_RATE
        self.token_size_mul = arch_params.TOKEN_SIZE_MUL 
        self.token_size_div = arch_params.TOKEN_SIZE_DIV
        self.input_joint_dim = input_joint_dim
        self.num_tokens = (((self.num_joints//10)*10) * (2**(self.token_size_mul)) / (2**self.down_t))
        self.encoder = PoseSPEncoderV1(rot_type=self.rot_type,
                                       input_dim=input_joint_dim,
                                       output_emb_width=self.code_dim,
                                       down_t=self.down_t,
                                       token_size_mul=self.token_size_mul,
                                       depth=self.depth,
                                       width=self.width,
                                       dilation_growth_rate=self.dilation_growth_rate,
                                       add_noise=add_noise)
        self.decoder = PoseSPDecoderV1(rot_type=self.rot_type,
                                       output_dim=output_joint_dim,
                                       output_emb_width=self.code_dim,
                                       down_t=self.down_t,
                                       depth=self.depth,
                                       width=self.width,
                                       token_size_div=self.token_size_div,
                                       num_tokens=self.num_tokens,
                                       dilation_growth_rate=self.dilation_growth_rate,
                                       num_joints=arch_params.NB_JOINTS,
                                       mesh_inference=mesh_inference)
        self.quantizer = QuantizeEMAReset(self.num_code, self.code_dim)

    def encode(self, x):
        batch_size, num_joints, rot_dim = x.shape[:3]
        if rot_dim == 3 and self.input_joint_dim == 6:
            x = matrix_to_rotation_6d(x)
        x_encoder = self.encoder(x)
        x_encoder = x_encoder.contiguous()
        code_idx = self.quantizer.quantize(x_encoder)
        code_idx = code_idx.view(batch_size, -1)
        return code_idx

    def forward(self, x, global_step=None):
        # Transform to 6D rotation if needed
        batch_size, num_joints, rot_dim = x.shape[:3]
        if rot_dim == 3 and self.input_joint_dim == 6:
            x = matrix_to_rotation_6d(x)
        # Encode
        x_encoder = self.encoder(x, global_step)
        ## quantization
        x_quantized, loss, perplexity  = self.quantizer(x_encoder)
        ## decoder
        x_decoder = self.decoder(x_quantized)
        return x_decoder, loss, perplexity


class DecodeTokens(nn.Module):
    def __init__(self,
                 ckpt_path='',
                 mesh_inference = False):
        super(DecodeTokens, self).__init__()
        
        num_joints = 21
        ckpt = torch.load(ckpt_path, map_location='cpu')
        pretrained_hparams = ckpt['hparams']
        arch = pretrained_hparams.ARCH
        rot_type = arch.ROT_TYPE
        code_dim = arch.CODE_DIM
        nb_code = arch.NB_CODE
        output_emb_width = code_dim
        down_t = arch.DOWN_T
        width = arch.WIDTH
        depth = arch.DEPTH
        dilation_growth_rate = arch.DILATION_RATE
        token_size_div = arch.TOKEN_SIZE_DIV
        token_size_mul = arch.TOKEN_SIZE_MUL
        num_tokens = (((num_joints//10)*10) * (2**(token_size_mul)) / (2**down_t))
        
        self.decoder = PoseSPDecoderV1(rot_type=rot_type,
                                       output_dim=6,
                                       output_emb_width=output_emb_width,
                                       down_t=down_t,
                                       width=width,
                                       depth=depth,
                                       token_size_div=token_size_div,
                                       num_tokens=num_tokens,
                                       dilation_growth_rate=dilation_growth_rate,
                                       num_joints=num_joints,
                                       mesh_inference=mesh_inference)
        self.quantizer = QuantizeEMAReset(nb_code, code_dim)
        self.load_weights(ckpt)

    def forward(self, logits):
        decode_feat = self.quantizer.dequantize_logits(logits)
        pose_out = self.decoder(decode_feat.permute(0,2,1))
        return pose_out['pred_pose_body_6d']

    def load_weights(self, ckpt):
        prepare_statedict(self.decoder, ckpt['net'], 'decoder', 'body_model')
        prepare_statedict(self.quantizer, ckpt['net'], 'quantizer', 'body_model')
        

class EncodeTokens(nn.Module):
    def __init__(self,
                 ckpt_path=''):
        super(EncodeTokens, self).__init__()
        
        ckpt = torch.load(ckpt_path, map_location='cpu')
        pretrained_hparams = ckpt['hparams']
        arch = pretrained_hparams.ARCH
        rot_type = arch.ROT_TYPE
        code_dim = arch.CODE_DIM
        nb_code = arch.NB_CODE
        output_emb_width = code_dim
        down_t = arch.DOWN_T
        width = arch.WIDTH
        depth = arch.DEPTH
        dilation_growth_rate = arch.DILATION_RATE
        token_size_mul = arch.TOKEN_SIZE_MUL
        
        self.encoder = PoseSPEncoderV1(rot_type=rot_type,
                                       input_dim=6,
                                       output_emb_width=output_emb_width,
                                       down_t=down_t,
                                       width=width,
                                       depth=depth,
                                       token_size_mul=token_size_mul,
                                       dilation_growth_rate=dilation_growth_rate)
        self.quantizer = QuantizeEMAReset(nb_code, code_dim)

        self.load_weights(ckpt)

    def forward(self, x):
        # Encoder
        x_encoder = self.encoder(x)

        # Quantize
        x_encoder = self.quantizer.preprocess(x_encoder)
        code_idx = self.quantizer.quantize(x_encoder)

        return code_idx

    def load_weights(self, ckpt):
        prepare_statedict(self.encoder, ckpt['net'], 'encoder')
        prepare_statedict(self.quantizer, ckpt['net'], 'quantizer')