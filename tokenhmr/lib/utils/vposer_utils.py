import numpy as np
import re
import torch
from collections import OrderedDict
from torch import nn
import torch.nn.functional as F

from .geometry import matrix_to_rotation_6d

def prepare_statedict(model, full_state_dict, partname, remove_name):
    part_statedict = {}
    new_part_statedict = OrderedDict()

    words = '{}'.format(remove_name)
    full_state_dict = {k.replace(words, '') if k.startswith(words) else k: v for k, v in full_state_dict.items()}
    
    # Load only the part given by sel_partname
    for key in full_state_dict.keys():
        if key.startswith(f'{partname}'):
            part_statedict[key] = full_state_dict[key]

    # Replace mismatch names
    for name, param in part_statedict.items():
        if re.match(f'^{partname}', name):
            name = name.replace(f'{partname}.', '', 1)
        new_part_statedict[name] = param

    model.load_state_dict(new_part_statedict, strict=True)
    return model

class ContinousRotReprDecoder(nn.Module):
    def __init__(self):
        super(ContinousRotReprDecoder, self).__init__()

    def forward(self, module_input):
        reshaped_input = module_input.view(-1, 3, 2)

        b1 = F.normalize(reshaped_input[:, :, 0], dim=1)

        dot_prod = torch.sum(b1 * reshaped_input[:, :, 1], dim=1, keepdim=True)
        b2 = F.normalize(reshaped_input[:, :, 1] - dot_prod * b1, dim=-1)
        b3 = torch.cross(b1, b2, dim=1)

        return torch.stack([b1, b2, b3], dim=-1)

class VPoserDecoder(nn.Module):
    def __init__(self, ckpt_path):
        super().__init__()
        
        self.num_neurons = 512
        self.latentD = 32
        self.num_joints = 21
        self.set_gpu = False

        self.decoder_net = nn.Sequential(
            nn.Linear(self.latentD, self.num_neurons),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.num_neurons, self.num_neurons),
            nn.LeakyReLU(),
            nn.Linear(self.num_neurons, self.num_joints * 6),
            ContinousRotReprDecoder(),
        )
        ckpt = torch.load(f'{ckpt_path}/snapshots/V02_05_epoch=13_val_loss=0.03.ckpt', map_location='cpu')['state_dict']
        prepare_statedict(self.decoder_net, ckpt, 'decoder_net', 'vp_model.')

    def forward(self, mu, logvar):
        batch_size = mu.shape[0]

        if not self.set_gpu:
            self.decoder_net = self.decoder_net.to(mu.device)
            self.set_gpu = True
        
        x = torch.distributions.normal.Normal(mu, F.softplus(logvar))
        decoder_x = x.rsample()
        smpl_rotmat = self.decoder_net(decoder_x)
        smpl_thetas6D = matrix_to_rotation_6d(smpl_rotmat).view(batch_size, -1)
        return smpl_thetas6D
