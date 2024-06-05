import argparse
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl' #'osmesa'
from pathlib import Path
import traceback
from typing import Optional
from lib.utils import Evaluator, recursive_to
import pandas as pd
import cv2
import numpy as np
import torch
from filelock import FileLock
import smplx
from lib.configs import dataset_eval_config
from lib.datasets import create_dataset

from tqdm import tqdm
from lib.models import load_tokenhmr
from lib.utils import MeshRenderer
from lib.models.smpl_wrapper import SMPL

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained models')
    parser.add_argument('--checkpoint', type=str, default='', help='Path to pretrained model checkpoint')
    parser.add_argument('--model_config', type=str, default='model_config.yaml', help='Path to model config file')
    parser.add_argument('--results_file', type=str, default='eval_regression.csv', help='Path to results file.')
    parser.add_argument('--dataset', type=str, default='EMDB, 3DPW-TEST', help='Dataset to evaluate') 
    parser.add_argument('--dataset_dir', type=str, default='', help='Dataset folder')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference')
    parser.add_argument('--num_samples', type=int, default=1, help='Number of test samples to draw')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers used for data loading')
    parser.add_argument('--log_freq', type=int, default=10, help='How often to log results')
    parser.add_argument('--shuffle', dest='shuffle', action='store_true', default=False, help='Shuffle the dataset during evaluation')
    parser.add_argument('--exp_name', type=str, default=None, help='Experiment name')
    parser.add_argument('--render', action='store_true')

    args = parser.parse_args()

    exp_name = 'eval' if args.exp_name is None else args.exp_name
    results_dir = f'results/release/{exp_name}'
    render_dir = f'{results_dir}/render'
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(render_dir, exist_ok=True)
    args.render_dir = render_dir
    args.results_file = os.path.join(results_dir, args.results_file)

    # Download and load checkpoints
    model, model_cfg = load_tokenhmr(checkpoint_path=args.checkpoint, \
                                 model_cfg=args.model_config, dataset_dir=args.dataset_dir)

    # Setup model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()
    print('model loaded!')
    print('using', args.checkpoint)
    # Load config and run eval, one dataset at a time
    print('Evaluating on datasets: {}'.format(args.dataset), flush=True)
    for dataset in args.dataset.split(','):
        dataset_cfg = dataset_eval_config()[dataset]
        if 'DATASET_FILE' in dataset_cfg:
            dataset_cfg['DATASET_FILE'] = os.path.join(args.dataset_dir, dataset_cfg['DATASET_FILE'])
        if 'IMG_DIR' in dataset_cfg:
            dataset_cfg['IMG_DIR'] = os.path.join(args.dataset_dir, dataset_cfg['IMG_DIR'])
        args.dataset = dataset
        print(dataset)
        run_eval(model, model_cfg, dataset_cfg, device, args)

def render_predictions(args, dataset, batch, output, mesh_renderer, idx):
    batch_size = batch['keypoints_2d'].shape[0]
    images = batch['img']
    images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1,3,1,1)
    images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1,3,1,1)
    gt_keypoints_2d = batch['keypoints_2d']

    pred_vertices = output['pred_vertices'].detach().reshape(batch_size, -1, 3)
    focal_length = output['focal_length'].detach().reshape(batch_size, 2)
    
    pred_cam_t = output['pred_cam_t'].detach().reshape(batch_size, 3)
    pred_keypoints_2d = output['pred_keypoints_2d'].detach().reshape(batch_size, -1, 2)
    num_images = min(batch_size, 8)
    
    predictions = mesh_renderer.visualize_tensorboard(pred_vertices[:num_images].cpu().numpy(),
                                                      pred_cam_t[:num_images].cpu().numpy(),
                                                      images[:num_images].cpu().numpy(),
                                                      pred_keypoints_2d[:num_images].cpu().numpy(),
                                                      gt_keypoints_2d[:num_images].cpu().numpy(),
                                                      focal_length=focal_length[:num_images].cpu().numpy())
    
    predictions = predictions.cpu().numpy().transpose(1,2,0)*255
    predictions = np.clip(predictions, 0, 255).astype(np.uint8)
    
    if 'pred_vertices_gt' in output:
        pred_vertices = output['pred_vertices_gt'].detach().reshape(batch_size, -1, 3)
        focal_length = output['focal_length'].detach().reshape(batch_size, 2)
        pred_cam_t = output['pred_cam_t'].detach().reshape(batch_size, 3)
        pred_keypoints_2d = output['pred_keypoints_2d_gt'].detach().reshape(batch_size, -1, 2)
        num_images = min(batch_size, 8)
        
        predictions_gt = mesh_renderer.visualize_tensorboard(pred_vertices[:num_images].cpu().numpy(),
                                                        pred_cam_t[:num_images].cpu().numpy(),
                                                        images[:num_images].cpu().numpy(),
                                                        pred_keypoints_2d[:num_images].cpu().numpy(),
                                                        gt_keypoints_2d[:num_images].cpu().numpy(),
                                                        focal_length=focal_length[:num_images].cpu().numpy())
        predictions_gt = predictions_gt.cpu().numpy().transpose(1,2,0)*255
        predictions_gt = np.clip(predictions_gt, 0, 255).astype(np.uint8)
        predictions = np.concatenate([predictions, predictions_gt[:,256:256*4]],1)
    
    cv2.imwrite(os.path.join(args.render_dir, f'render_{dataset}_{idx}.png'),
                cv2.cvtColor(predictions, cv2.COLOR_BGR2RGB))
    return predictions


def run_eval(model, model_cfg, dataset_cfg, device, args):

    if args.render:
        smpl_cfg = {k.lower(): v for k,v in dict(model_cfg.SMPL).items()}
        smpl = SMPL(**smpl_cfg)
        mesh_renderer = MeshRenderer(model_cfg, smpl.faces)

    # Create dataset and data loader
    dataset = create_dataset(model_cfg, dataset_cfg, train=False)
    dataloader = torch.utils.data.DataLoader(dataset, args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers)

    J_regressor_24_SMPL = None
    # List of metrics to log
    if args.dataset in ['EMDB', '3DPW-TEST']:
        metrics = ['mode_re', 'mode_mpjpe', 'mode_pve']
        J_regressor_24_SMPL=smplx.SMPL(model_path=model_cfg.SMPL.MODEL_PATH).J_regressor.cuda().float()
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')

    # Setup evaluator object
    evaluator = Evaluator(
        dataset_length=int(1e8), 
        keypoint_list=dataset_cfg.KEYPOINT_LIST, 
        pelvis_ind=model_cfg.EXTRA.PELVIS_IND, 
        metrics=metrics,
        J_regressor_24_SMPL = J_regressor_24_SMPL,
        dataset=args.dataset,
    )

    for i, batch in enumerate(tqdm(dataloader)):
        batch = recursive_to(batch, device)
        with torch.no_grad():
            out = model(batch)

        evaluator(out, batch)
        if i % args.log_freq == args.log_freq - 1:
            evaluator.log()
            if args.render:
                predictions = render_predictions(args, args.dataset, batch, out, mesh_renderer, i)
    evaluator.log()
    error = None

    # Append results to file
    metrics_dict = evaluator.get_metrics_dict()
    save_eval_result(args.results_file, metrics_dict, args.checkpoint, args.dataset, error=error, iters_done=i, exp_name=args.exp_name)


def save_eval_result(
    csv_path: str,
    metric_dict: float,
    checkpoint_path: str,
    dataset_name: str,
    # start_time: pd.Timestamp,
    error: Optional[str] = None,
    iters_done=None,
    exp_name=None,
) -> None:
    """Save evaluation results for a single scene file to a common CSV file."""

    timestamp = pd.Timestamp.now()
    exists: bool = os.path.exists(csv_path)
    exp_name = exp_name or Path(checkpoint_path).parent.parent.name

    # save each metric as different row to the csv path
    metric_names = list(metric_dict.keys())
    metric_values = list(metric_dict.values())
    
    metric_values = [float('{:.2f}'.format(value)) for value in metric_values]
    N = len(metric_names)
    df = pd.DataFrame(
        dict(
            timestamp=[timestamp] * N,
            checkpoint_path=[checkpoint_path] * N,
            exp_name=[exp_name] * N,
            dataset=[dataset_name] * N,
            metric_name=metric_names,
            metric_value=metric_values,
            error=[error] * N,
            iters_done=[iters_done] * N,
        ),
        index=list(range(N)),
    )

    # Lock the file to prevent multiple processes from writing to it at the same time.
    # lock = FileLock(f"{csv_path}.lock", timeout=10)
    # with lock:
    df.to_csv(csv_path, mode="a", header=not exists, index=False)

if __name__ == '__main__':
    main()
