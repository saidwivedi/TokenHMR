from .vit import vit
import os
import torch
from collections import OrderedDict
import re

# log = pylogger.get_pylogger(__name__)

def prepare_statedict(model, full_state_dict, partname, strict=True):
    part_statedict = {}
    new_part_statedict = OrderedDict()

    # Load only the part given by sel_partname
    for key in full_state_dict.keys():
        if key.startswith(f'{partname}'):
            part_statedict[key] = full_state_dict[key]

    # Replace mismatch names
    for name, param in part_statedict.items():
        if re.match(f'^{partname}', name):
            name = name.replace(f'{partname}.', '')
        new_part_statedict[name] = param

    try:
        model.load_state_dict(new_part_statedict, strict=True)
    except Exception as e:
        # log.warning(f'Mismatch in statedict of {partname}!!!')
        # log.warning(f'{e}')
        if not strict:
            # log.warning(f'Partially Initializing {partname}...')
            model.load_state_dict(new_part_statedict, strict=False)
    return model

def create_backbone(cfg, load_weights=True):
    if cfg.MODEL.BACKBONE.TYPE == 'vit':
        backbone = vit(cfg)
        if cfg.MODEL.BACKBONE.get('PRETRAINED_WEIGHTS', None) and load_weights:
            PRETRAINED_WEIGHTS_path = cfg.MODEL.BACKBONE.PRETRAINED_WEIGHTS
            pt_model = torch.load(PRETRAINED_WEIGHTS_path, map_location='cpu')['state_dict']
            print(f'Loading backbone weights from {PRETRAINED_WEIGHTS_path}')
            try:
                backbone.load_state_dict(pt_model)
            except:
                print(f'Could not load {PRETRAINED_WEIGHTS_path} in strict mode!!!, trying to load partially...')
                backbone = prepare_statedict(backbone, pt_model, 'backbone')
        return backbone
    else:
        raise NotImplementedError('Backbone type is not implemented')
