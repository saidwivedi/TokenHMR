from typing import Optional, Tuple
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import pyrootutils

root_dir = __file__.replace(os.path.basename(__file__), '')
root = pyrootutils.setup_root(root_dir, dotenv=True, pythonpath=True)

from pathlib import Path
import hydra
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.plugins.environments import SLURMEnvironment

from yacs.config import CfgNode
from lib.configs import dataset_config
from lib.datasets import TokenHMRDataModule
from lib.models.tokenhmr import TokenHMR
from lib.utils.pylogger import get_pylogger
from lib.utils.misc import task_wrapper, log_hyperparameters, get_grid_search_configs

# HACK reset the signal handling so the lightning is free to set it
# Based on https://github.com/facebookincubator/submitit/issues/1709#issuecomment-1246758283
import signal
signal.signal(signal.SIGUSR1, signal.SIG_DFL)

log = get_pylogger(__name__)


@pl.utilities.rank_zero.rank_zero_only
def save_configs(model_cfg: CfgNode, dataset_cfg: CfgNode, rootdir: str):
    """Save config files to rootdir."""
    Path(rootdir).mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=model_cfg, f=os.path.join(rootdir, 'model_config.yaml'))
    with open(os.path.join(rootdir, 'dataset_config.yaml'), 'w') as f:
        f.write(dataset_cfg.dump())

@task_wrapper
def train(cfg: DictConfig) -> Tuple[dict, dict]:
    # Load dataset config
    dataset_cfg = dataset_config()
    
    # Do a grid search of params mentioned in list
    exp, hyper_params = get_grid_search_configs(cfg)
    print(f'Number of experiments --> {len(exp)} || Grid search params: {hyper_params}')

    # From the number of experiments, choose the experiment defined by `cls_id`: default is 0
    cfg = OmegaConf.create(exp[cfg.cls_id])

    # Save configs
    save_configs(cfg, dataset_cfg, cfg.paths.output_dir)

    # Setup training and validation datasets
    datamodule = TokenHMRDataModule(cfg, dataset_cfg)

    # Setup model
    model = TokenHMR(cfg, is_train_state=True)

    # Setup loggers
    logger = TensorBoardLogger(os.path.join(cfg.paths.output_dir, 'tensorboard'), name='', version='', default_hp_metric=False)
    loggers = [logger]

    # Setup checkpoint saving
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(cfg.paths.output_dir, 'checkpoints'), 
        every_n_train_steps=cfg.GENERAL.CHECKPOINT_STEPS, 
        save_last=True,
        save_top_k=cfg.GENERAL.CHECKPOINT_SAVE_TOP_K,
        save_weights_only=True,
    )
    rich_callback = pl.callbacks.RichProgressBar()
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
    callbacks = [
        checkpoint_callback, 
        lr_monitor,
        # rich_callback
    ]

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, 
        callbacks=callbacks, 
        logger=loggers, 
        plugins=(SLURMEnvironment(requeue_signal=signal.SIGUSR2) if (cfg.get('launcher',None) is not None) else None), # Submitit uses SIGUSR2
    )
    #accumulate_grad_batches=32,

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    # Train the model
    checkpoint_path = cfg.get('resume_path', None)
    trainer.fit(model, datamodule=datamodule, ckpt_path=checkpoint_path)
    log.info("Fitting done")


@hydra.main(version_base="1.2", config_path=os.path.join(root, "lib", "configs_hydra"), config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    # train the model
    train(cfg)


if __name__ == "__main__":
    main()
