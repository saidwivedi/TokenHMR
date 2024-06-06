from .smpl_head import SMPLTransformerDecoderHead
from .token_head import SMPLTokenDecoderHead

def build_smpl_head(cfg, *args, **kwargs):
    smpl_head_type = cfg.MODEL.SMPL_HEAD.get('TYPE', 'hmr')
    if smpl_head_type == 'transformer_decoder':
        print('using hmr2 head!!!')
        return SMPLTransformerDecoderHead(cfg, *args, **kwargs)
    elif smpl_head_type == 'token':
        print('using token head!!!')
        return SMPLTokenDecoderHead(cfg, *args, **kwargs)
    else:
        raise ValueError('Unknown SMPL head type: {}'.format(smpl_head_type))