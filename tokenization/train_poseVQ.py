import os
import json
import tqdm
import argparse
import warnings
from os.path import join, isdir
warnings.filterwarnings('ignore')

import torch
import torch.optim as optim

from dataset.dataset_poseVQ import get_dataloader
import utils.losses as losses 
import utils.utils_model as utils_model
from utils.pose_visualize import visualize_from_mesh
from utils.eval_poseVQ import eval_pose_vqvae, reset_err_list, init_best_scores, set_random_seed, get_loggers
from options.option_posevq import run_grid_search_experiments

def update_lr_warm_up(optimizer, nb_iter, warm_up_iter, lr):

    current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    return optimizer, current_lr

def get_model(hparams, add_noise=False):
    if hparams.ARCH.MODEL_NAME == 'vanilla':
        from models.vanilla_pose_vqvae import VanillaTokenizer
        net = VanillaTokenizer(hparams.ARCH, add_noise=add_noise)
    else:
        raise NotImplementedError(f'{hparams.ARCH.MODEL_NAME} not implemented yet')
    return net

def main(hparams):
    ##### ---- Exp dirs ---- #####
    torch.manual_seed(hparams.EXP.SEED)

    hparams.EXP.OUT_DIR = os.path.join(hparams.EXP.OUT_DIR, f'{hparams.EXP.NAME}')
    save_dir = join(hparams.EXP.OUT_DIR, 'train_render')
    os.makedirs(hparams.EXP.OUT_DIR, exist_ok = True)
    os.makedirs(save_dir, exist_ok = True)
    writer = None
    
    ##### ---- Logger ---- #####
    logger = utils_model.get_logger(hparams.EXP.OUT_DIR)
    logger.info(f'Training on {hparams.DATA.DATASET}, with {hparams.ARCH.NB_JOINTS} joints')

    ##### ---- Dataloader ---- #####
    if hparams.EXP.EVAL_ONLY:
        eval_loader = get_dataloader(hparams, split=hparams.EXP.EVAL_DS, shuffle=False)
    else:
        train_loader_iter = get_dataloader(hparams, split='train')
        val_loader = get_dataloader(hparams, split='val', shuffle=False)

    ##### ------ eval-only ------- #####
    if hparams.EXP.EVAL_ONLY: 
        set_random_seed(0)
        best_scores = init_best_scores()
        writer = get_loggers(hparams)
        logger.info('EVAL-ONLY: loading checkpoint from {}'.format(hparams.EXP.RESUME_PTH))
        ckpt_file = f'{hparams.EXP.RESUME_PTH}/best_net.pth' if isdir(hparams.EXP.RESUME_PTH) else hparams.EXP.RESUME_PTH
        ckpt = torch.load(ckpt_file, map_location='cpu')
        pretrained_hparams = ckpt['hparams']
        net = get_model(pretrained_hparams)
        net.load_state_dict(ckpt['net'], strict=True)
        net.cuda()
        eval_pose_vqvae(hparams, eval_loader, net, logger, writer, 0, hparams.EXP.OUT_DIR, hparams.EXP.VAL_DISP_ITER, best_scores)
        exit()

    ##### ------ resume training ------- #####
    if hparams.EXP.RESUME_TRAINING:
        print(f'RESUME TRAINING: loading checkpoint from {hparams.EXP.RESUME_PTH}. Overiding architecture...')
        ckpt = torch.load(hparams.EXP.RESUME_PTH, map_location='cpu')
        pretrained_hparams = ckpt['hparams']
        hparams.ARCH = pretrained_hparams.ARCH
        writer = get_loggers(hparams)
        net = get_model(pretrained_hparams)
        net.load_state_dict(ckpt['net'], strict=True)

    ##### ---- training from scratch ---- #####
    else:
        print('train params:', hparams.ARCH)
        writer = get_loggers(hparams)
        net = get_model(hparams, hparams.DATA.ADD_NOISE)

    net.train()
    net.cuda()
    
    ##### ---- Optimizer & Scheduler ---- #####
    optimizer = optim.AdamW(net.parameters(), lr=hparams.OPT.LR, betas=(0.9, 0.99), weight_decay=hparams.OPT.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=hparams.OPT.LR_SCHEDULER.split('_'), gamma=hparams.OPT.GAMMA)
    
    Loss = losses.PoseReConsLoss(hparams.LOSS, hparams.ARCH.NB_JOINTS, hparams.ARCH.ROT_TYPE, hparams.ARCH.SMPL_TYPE)
    best_scores = init_best_scores()


    ##### ------ warm-up ------- #####
    avg_recons, avg_perplexity, avg_commit = 0., 0., 0.

    for nb_iter in range(1, hparams.OPT.WARM_UP_ITER):
        
        optimizer, current_lr = update_lr_warm_up(optimizer, nb_iter, hparams.OPT.WARM_UP_ITER, hparams.OPT.LR)
        
        batch = next(train_loader_iter)
        gt_pose = batch['gt_pose_body'].cuda().float() # (bs, 63)
        gt_mesh = batch['body_vertices'].cuda().float()
        gt_jnts = batch['body_joints'].cuda().float()
        
        output, loss_commit, perplexity = net(gt_pose)
        loss_pose = Loss.forward_pose(gt_pose, output)
        loss_mesh = Loss.forward_mesh(gt_mesh, output)
        loss_jnts = Loss.forward_joints(gt_jnts, output)

        loss = hparams.LOSS.POSE_LOSS_WT * loss_pose + \
               hparams.LOSS.MESH_LOSS_WT * loss_mesh + \
               hparams.LOSS.JNT_LOSS_WT * loss_jnts + \
               hparams.LOSS.COMMIT_LOSS_WT * loss_commit

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_recons += loss_pose.item()
        avg_perplexity += perplexity.item()
        avg_commit += loss_commit.item()
        
        if nb_iter % hparams.EXP.PRINT_ITER ==  0 :
            avg_recons /= hparams.EXP.PRINT_ITER
            avg_perplexity /= hparams.EXP.PRINT_ITER
            avg_commit /= hparams.EXP.PRINT_ITER
            
            logger.info(f"Warmup. Iter {nb_iter} :  lr {current_lr:.5f} \t Commit. {avg_commit:.5f} \t PPL. {avg_perplexity:.2f} \t Recons.  {avg_recons:.5f}")
            
            avg_recons, avg_perplexity, avg_commit = 0., 0., 0.

    ##### ---- Training ---- #####
    err_list = reset_err_list('tr')

    for nb_iter in tqdm.tqdm(range(1, hparams.OPT.TOTAL_ITER + 1)):
        
        batch = next(train_loader_iter)
        gt_pose = batch['gt_pose_body'].cuda().float() # (bs, 63)
        gt_mesh = batch['body_vertices'].cuda().float()
        gt_jnts = batch['body_joints'].cuda().float()
        
        output, loss_commit, perplexity = net(gt_pose, nb_iter)
        loss_pose = Loss.forward_pose(gt_pose, output)
        loss_mesh = Loss.forward_mesh(gt_mesh, output)
        loss_jnts = Loss.forward_joints(gt_jnts, output)

        loss = hparams.LOSS.POSE_LOSS_WT * loss_pose + \
               hparams.LOSS.MESH_LOSS_WT * loss_mesh + \
               hparams.LOSS.JNT_LOSS_WT * loss_jnts + \
               hparams.LOSS.COMMIT_LOSS_WT * loss_commit
        loss *= hparams.LOSS.LOSS_WT

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        err_list['tr/curr_pose_recons'] += hparams.LOSS.POSE_LOSS_WT * loss_pose.item()
        err_list['tr/curr_mesh_recons'] += hparams.LOSS.MESH_LOSS_WT * loss_mesh.item()
        err_list['tr/curr_jnt_recons'] += hparams.LOSS.JNT_LOSS_WT * loss_jnts.item()
        err_list['tr/curr_perplexity'] += perplexity.item()
        err_list['tr/curr_commit'] += hparams.LOSS.COMMIT_LOSS_WT * loss_commit.item()
        err_list['tr/curr_loss'] += hparams.LOSS.LOSS_WT * loss.item()

        if nb_iter % hparams.EXP.PRINT_ITER ==  0 :
            for key, value in err_list.items():
                err_list[key] /= hparams.EXP.PRINT_ITER
            
            if writer is not None:
                for key, value in err_list.items():
                    writer.add_scalar(f'{key}', err_list[key], nb_iter)

            print_str = f'Train. Iter {nb_iter}: lr: {scheduler.get_last_lr()[0]:.5f}'
            for key, value in err_list.items():
                print_str += f'\t{key[7:]}: {value:.5f}'
            logger.info(print_str)

            err_list = reset_err_list('tr')

        if nb_iter % hparams.EXP.TR_DISP_ITER == 0:
            visualize_from_mesh(hparams.ARCH.SMPL_TYPE, batch, output, nb_iter, save_dir) 

        if nb_iter % hparams.EXP.EVAL_ITER == 0:
            best_scores = eval_pose_vqvae(hparams, val_loader, net, logger, writer, nb_iter, hparams.EXP.OUT_DIR, hparams.EXP.VAL_DISP_ITER, best_scores)
            

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg', type=str, help='cfg file path')
    parser.add_argument('--cfg_id', type=int, default=0, help='cfg id to run when multiple experiments are spawned')
    parser.add_argument('--cluster', default=False, action='store_true', help='creates submission files for cluster')
    parser.add_argument('--bid', type=int, default=30, help='amount of bid for cluster')
    parser.add_argument('--memory', type=int, default=64000, help='memory amount for cluster')
    parser.add_argument('--gpu_min_mem', type=int, default=20000, help='minimum gpu mem')
    parser.add_argument('--exclude_nodes', type=str, default='', help='exclude the nodes from submitting')
    parser.add_argument('--num_cpus', type=int, default=8, help='num cpus for cluster')

    args = parser.parse_args()

    print(f'Input arguments: \n {args}')

    hparams = run_grid_search_experiments(
        cfg_id=args.cfg_id,
        cfg_file=args.cfg,
        bid=args.bid,
        use_cluster=args.cluster,
        memory=args.memory,
        exclude_nodes=args.exclude_nodes,
        script='train_poseVQ.py',
        gpu_min_mem=args.gpu_min_mem,
    )
    main(hparams)