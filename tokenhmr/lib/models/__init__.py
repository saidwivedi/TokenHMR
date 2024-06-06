from .tokenhmr import TokenHMR

def load_tokenhmr(checkpoint_path='', model_cfg=f'', dataset_dir='', is_train_state=False, is_demo=False):
    from pathlib import Path
    from ..configs import get_config
    model_cfg = get_config(model_cfg)

    # overide model config
    model_cfg.defrost()

    # Update checkpoint path
    model_cfg.ckpt_path = checkpoint_path

    # Override some config values, to crop bbox correctly
    if (model_cfg.MODEL.BACKBONE.TYPE == 'vit') and ('BBOX_SHAPE' not in model_cfg.MODEL):
        assert model_cfg.MODEL.IMAGE_SIZE == 256, f"MODEL.IMAGE_SIZE ({model_cfg.MODEL.IMAGE_SIZE}) should be 256 for ViT backbone"
        model_cfg.MODEL.BBOX_SHAPE = [192,256]

    if dataset_dir != '':
        model_cfg.DATASETS.DATASET_DIR = dataset_dir

    # freeze model config
    model_cfg.freeze()

    model = TokenHMR(cfg=model_cfg, is_train_state=is_train_state, is_demo=is_demo)
    return model, model_cfg
