import os
import random
from os.path import join
import tqdm
import numpy as np
import torch
import pickle as pkl

from utils.pose_visualize import visualize_from_mesh
from torch.utils.tensorboard import SummaryWriter

def reset_err_list(type='tr'):
    err_list = {f'{type}/curr_pose_recons': 0.,
                f'{type}/curr_mesh_recons': 0.,
                f'{type}/curr_jnt_recons': 0.,
                f'{type}/curr_perplexity': 0.,
                f'{type}/curr_commit': 0.}
    if type == 'tr':
        err_list.update({
            f'{type}/curr_loss': 0.
        })
    return err_list

def init_best_scores():
    best_scores = {
        f'val/best_iter': 0,
        f'val/best_val_score': 1e8,
        f'val/best_mesh_recons': 1e8,
        f'val/best_jnt_recons': 1e8,
    }
    return best_scores

def set_random_seed(random_seed=0):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def get_loggers(hparams):
    writer = None
    if hparams.EXP.LOG_TB:
        writer = SummaryWriter(hparams.EXP.OUT_DIR)
    return writer

def calculate_pose_reconstruction_error(gt_pose, pred_pose):
    return torch.sqrt(torch.pow(gt_pose-pred_pose, 2).sum(-1)).mean()

def calculate_mesh_reconstruction_error(gt_mesh, pred_mesh):
    return torch.sqrt(torch.pow(gt_mesh-pred_mesh, 2).sum(-1)).mean()

def calculate_jnts_reconstruction_error(gt_jnts, pred_jnts):
    valid_joints = [*range(1,22)] # only body joints
    return torch.sqrt(torch.pow(gt_jnts[:,valid_joints]-pred_jnts[:,valid_joints], 2).sum(-1)).mean()

def save_results_func(batch, output, results):
    valid_joints = [*range(1,22)]
    save_aa_gt = batch['pose_body_aa'].numpy()
    save_jnts_gt = batch['body_joints'][:,valid_joints].numpy()
    save_aa_pred = output['pred_pose_body_aa'].detach().cpu().numpy()
    save_jnts_pred = output['pred_body_joints'][:,valid_joints].detach().cpu().numpy()
    results['gt_jnts'].extend(save_jnts_gt)
    results['gt_aa'].extend(save_aa_gt)
    results['pred_jnts'].extend(save_jnts_pred)
    results['pred_aa'].extend(save_aa_pred)
    return results


def eval_pose_vqvae(hparams, val_loader, net, logger, writer, nb_iter, out_dir, val_disp_iter, best_scores):

    net.eval()
    save_dir = join(out_dir, 'val_render')
    os.makedirs(save_dir, exist_ok=True)

    save_results = True
    results = {'gt_jnts': [], 'gt_aa': [], 'pred_jnts': [], 'pred_aa': []}

    err_list = reset_err_list('val')
    dataset_name = ''

    for batch_idx, batch in enumerate(tqdm.tqdm(val_loader)):
        gt_pose = batch['gt_pose_body'].cuda().float() # (bs, 63)
        gt_mesh = batch['body_vertices'].cuda().float() # (bs, 10743)
        gt_jnts = batch['body_joints'].cuda().float()
        dataset_name = batch['dataset_name'][0]

        output, loss_commit, perplexity = net(gt_pose)
        pose_error = calculate_pose_reconstruction_error(gt_pose, output['pred_pose_body_rotmat'])
        mesh_error = calculate_mesh_reconstruction_error(gt_mesh, output['pred_body_vertices'])
        jnt_error = calculate_jnts_reconstruction_error(gt_jnts, output['pred_body_joints'])

        if save_results:
            results = save_results_func(batch, output, results)

        err_list['val/curr_pose_recons'] += pose_error.item()
        err_list['val/curr_mesh_recons'] += mesh_error.item()
        err_list['val/curr_jnt_recons'] += jnt_error.item()
        err_list['val/curr_perplexity'] += perplexity.item()
        err_list['val/curr_commit'] += loss_commit.item()
        
        if batch_idx % val_disp_iter == 0:
            visualize_from_mesh(hparams.ARCH.SMPL_TYPE, batch, output, f'eval_{nb_iter}_{batch_idx}', save_dir)

    if save_results:
        with open(f'results{dataset_name}.pkl', 'wb') as handle:
            pkl.dump(results, handle, protocol=pkl.HIGHEST_PROTOCOL)
    
    for key, value in err_list.items():
        err_list[key] /= batch_idx

    err_list['val/curr_jnt_recons'] *= 1000
    err_list['val/curr_mesh_recons'] *= 1000
    
    curr_score = err_list['val/curr_jnt_recons'] + err_list['val/curr_mesh_recons']
    if curr_score < best_scores['val/best_val_score']:
        best_scores['val/best_val_score'] = curr_score
        save_dict = {
            'net': net.state_dict(),
            'hparams': hparams,
        }
        best_scores['val/best_iter'] = nb_iter
        best_scores['val/best_jnt_recons']  = err_list['val/curr_jnt_recons']
        best_scores['val/best_mesh_recons'] = err_list['val/curr_mesh_recons']
        torch.save(save_dict, join(out_dir, 'best_net.pth'))
        logger.info(f"Eval. Iter {nb_iter}: !!---> BEST MODEL FOUND <---!! Validation Score - {best_scores['val/best_val_score']:.2f}")
    
    print_str = f'Eval. Iter: {nb_iter} | curr_score: {curr_score:.2f}'
    for key, value in err_list.items():
        print_str += f'\t {key[9:]}: {value:.5f}'
    for key, value in best_scores.items():
        print_str += f'\t {key[4:]}: {value:.5f}'
    logger.info(print_str)
    
    if writer is not None:
        for key, value in err_list.items():
            writer.add_scalar(f'{key}', err_list[key], nb_iter)
        for key, value in best_scores.items():
            writer.add_scalar(f'{key}', err_list[key], nb_iter)

    net.eval()

    return best_scores
    