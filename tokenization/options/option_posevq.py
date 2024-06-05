
import os
import time
import argparse
import yaml
import shutil
import operator
import itertools

from yacs.config import CfgNode as CN
from functools import reduce
from typing import Dict, List, Union, Any
from flatten_dict import flatten, unflatten

from utils.cluster import execute_task_on_cluster

hparams = CN()

hparams.DATA = CN()
hparams.DATA.DATA_ROOT = 'data'
hparams.DATA.DATASET = 'amass_pose' # deprecated
hparams.DATA.TRAINLIST = 'CMU_KIT_BMLrub_BMLmovi_MOYO_TotalCapture_EKUT_ACCAD_TCDHands'
hparams.DATA.TRAIN_PART = '0.2_0.2_0.2_0.1_0.1_0.1_0.025_0.025_0.025_0.025'
hparams.DATA.VALLIST = 'HumanEva_HDM05_SFU_MOYO'
hparams.DATA.TESTLIST = 'Transitions_SSM'
hparams.DATA.MASK_BODY_PARTS = False
hparams.DATA.ADD_NOISE = False
hparams.DATA.BATCH_SIZE = 256
hparams.DATA.NUM_WORKERS = 8

hparams.OPT = CN()
hparams.OPT.TOTAL_ITER = 200000
hparams.OPT.WARM_UP_ITER = 2
hparams.OPT.LR = 2e-4
hparams.OPT.LR_SCHEDULER = '75000_100000'
hparams.OPT.GAMMA = 0.05
hparams.OPT.WEIGHT_DECAY = 0.0

hparams.LOSS = CN()
hparams.LOSS.POSE_LOSS_WT = 1.0
hparams.LOSS.MESH_LOSS_WT = 1.0
hparams.LOSS.JNT_LOSS_WT = 1.0
hparams.LOSS.COMMIT_LOSS_WT = 0.02
hparams.LOSS.LOSS_WT = 5.0
hparams.LOSS.ONLY_VALID_JNT = True
hparams.LOSS.POSE_LOSS = 'l2'
hparams.LOSS.MESH_LOSS = 'l1'
hparams.LOSS.JNT_LOSS = 'l2'

hparams.ARCH = CN()
hparams.ARCH.MODEL_NAME = 'vanila'
hparams.ARCH.CODE_DIM = 512
hparams.ARCH.NB_CODE = 512
hparams.ARCH.DOWN_T = 1
hparams.ARCH.WIDTH = 512
hparams.ARCH.DEPTH = 2
hparams.ARCH.DILATION_RATE = 3
hparams.ARCH.TOKEN_SIZE_MUL = 2
hparams.ARCH.TOKEN_SIZE_DIV = 1
hparams.ARCH.N_ENCODER_LAYERS = 3
hparams.ARCH.N_DECODER_LAYERS = 2
hparams.ARCH.NUM_TOKENS = 10
hparams.ARCH.NB_JOINTS = 21
hparams.ARCH.ROT_TYPE = 'rotmat'
hparams.ARCH.QUANTIZER = 'ema_reset' # ema, orig, ema_reset, reset
hparams.ARCH.SMPL_TYPE = 'smplh'
hparams.ARCH.CB_SCALE_DOWN = 2
hparams.ARCH.BETA = 1.0

hparams.EXP = CN()
hparams.EXP.ID = ''
hparams.EXP.NUM_GPUS = 1
hparams.EXP.NAME = 'debug'
hparams.EXP.OUT_DIR = 'output'
hparams.EXP.DEBUG = False
hparams.EXP.PRINT_ITER = 100
hparams.EXP.EVAL_ITER = 500
hparams.EXP.TR_DISP_ITER = 1000
hparams.EXP.VAL_DISP_ITER = 500
hparams.EXP.SEED = 123
hparams.EXP.EVAL_ONLY = False
hparams.EXP.EVAL_DS = 'test'
hparams.EXP.RESUME_PTH = ''
hparams.EXP.RESUME_TRAINING = False
hparams.EXP.LOG_TB = False

def get_hparams_defaults():
    """Get a yacs hparamsNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return hparams.clone()


def update_hparams(hparams_file):
    hparams = get_hparams_defaults()
    hparams.merge_from_file(hparams_file)
    return hparams.clone()


def update_hparams_from_dict(cfg_dict):
    hparams = get_hparams_defaults()
    cfg = hparams.load_cfg(str(cfg_dict))
    hparams.merge_from_other_cfg(cfg)
    return hparams.clone()

def CfgNode_to_dict(cfgnode):
    if isinstance(cfgnode, CN):
        cfg_dict = dict(cfgnode)
        for key in cfgnode.keys():
            cfg_dict[key] = CfgNode_to_dict(cfgnode[key])
        return cfg_dict
    elif isinstance(cfgnode, list):
        return [CfgNode_to_dict(item) for item in cfgnode]
    else:
        return cfgnode

def flatten_cfgnode(cfgnode, parent_key='', separator='/'):
    cfg_dict = {}
    for key, value in cfgnode.items():
        full_key = f"{parent_key}{separator}{key}" if parent_key else key
        if isinstance(value, CN):
            cfg_dict.update(flatten_cfgnode(value, parent_key=full_key, separator=separator))
        else:
            cfg_dict[full_key] = value

    return cfg_dict


def get_grid_search_configs(config, excluded_keys=[]):
    """
    :param config: dictionary with the configurations
    :return: The different configurations
    """

    def bool_to_string(x: Union[List[bool], bool]) -> Union[List[str], str]:
        """
        boolean to string conversion
        :param x: list or bool to be converted
        :return: string converted thinghat
        """
        if isinstance(x, bool):
            return [str(x)]
        for i, j in enumerate(x):
            x[i] = str(j)
        return x

    # exclude from grid search

    flattened_config_dict = flatten(config, reducer='path')
    hyper_params = []

    for k,v in flattened_config_dict.items():
        if isinstance(v,list):
            if k in excluded_keys:
                flattened_config_dict[k] = ['+'.join(v)]
            elif len(v) > 1:
                hyper_params += [k]

        if isinstance(v, list) and isinstance(v[0], bool) :
            flattened_config_dict[k] = bool_to_string(v)

        if not isinstance(v,list):
            if isinstance(v, bool):
                flattened_config_dict[k] = bool_to_string(v)
            else:
                flattened_config_dict[k] = [v]

    keys, values = zip(*flattened_config_dict.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for exp_id, exp in enumerate(experiments):
        for param in excluded_keys:
            exp[param] = exp[param].strip().split('+')
        for param_name, param_value in exp.items():
            # print(param_name,type(param_value))
            if isinstance(param_value, list) and (param_value[0] in ['True', 'False']):
                exp[param_name] = [True if x == 'True' else False for x in param_value]
            if param_value in ['True', 'False']:
                if param_value == 'True':
                    exp[param_name] = True
                else:
                    exp[param_name] = False


        experiments[exp_id] = unflatten(exp, splitter='path')

    return experiments, hyper_params

def run_grid_search_experiments(
        cfg_id,
        cfg_file,
        use_cluster,
        bid,
        memory,
        exclude_nodes,
        script='train_poseVQ.py',
        gpu_min_mem=10000,
):
    cfg = yaml.full_load(open(cfg_file))

    # parse config file to get a list of configs and related hyperparameters
    different_configs, hyperparams = get_grid_search_configs(
        cfg,
        excluded_keys=[],
    )
    print(f'Grid search hparams: \n {hyperparams}')

    different_configs = [update_hparams_from_dict(c) for c in different_configs]
    print(f'======> Number of experiment configurations is {len(different_configs)}')
    config_to_run = CN(different_configs[cfg_id])

    print(f'===> Number of GPUs {config_to_run.EXP.NUM_GPUS}')

    if use_cluster:
        cls_run_folder = 'scripts/cluster'
        new_cfg_file = os.path.join(cls_run_folder, f'{config_to_run.EXP.NAME}_config.yaml')
        os.makedirs(cls_run_folder, exist_ok=True)
        shutil.copy(src=cfg_file, dst=new_cfg_file)
        execute_task_on_cluster(
            script=script,
            exp_name=config_to_run.EXP.NAME,
            num_exp=len(different_configs),
            cfg_file=new_cfg_file,
            bid_amount=bid,
            num_workers=config_to_run.DATA.NUM_WORKERS * config_to_run.EXP.NUM_GPUS,
            memory=memory,
            exclude_nodes=exclude_nodes,
            gpu_min_mem=gpu_min_mem,
            num_gpus=config_to_run.EXP.NUM_GPUS,
        )
        exit()

    # ==== create logdir using hyperparam settings
    logtime = time.strftime('%d-%m-%Y_%H-%M-%S')
    logdir = f'{config_to_run.EXP.NAME}_ID{cfg_id:02d}_{logtime}'
    config_to_run.EXP.ID += f'{config_to_run.EXP.NAME}_ID{cfg_id:02d}'

    def get_from_dict(dict, keys):
        return reduce(operator.getitem, keys, dict)

    exp_id = ''
    for hp in hyperparams:
        v = get_from_dict(different_configs[cfg_id], hp.split('/'))
        exp_id += f'{hp.replace("/", ".").replace("_", "").lower()}-{v}'
    exp_id = exp_id.replace('/', '.')

    if exp_id:
        logdir += f'_{exp_id}'
        config_to_run.EXP.ID += f'/{exp_id}'

    # config_to_run.EXP.ID += f'{logtime}'

    logdir = os.path.join(config_to_run.EXP.OUT_DIR, config_to_run.EXP.NAME, logdir)
    os.makedirs(logdir, exist_ok=True)
    shutil.copy(src=cfg_file, dst=os.path.join(config_to_run.EXP.OUT_DIR, 'config.yaml'))

    config_to_run.EXP.OUT_DIR = logdir

    def save_dict_to_yaml(obj, filename, mode='w'):
        with open(filename, mode) as f:
            yaml.dump(obj, f, default_flow_style=False)

    # save config
    save_dict_to_yaml(
        unflatten(flatten(config_to_run)),
        os.path.join(config_to_run.EXP.OUT_DIR, 'config_to_run.yaml')
    )

    return config_to_run
