import os
import trimesh
import torch
import numpy as np
import cv2
from torchvision.utils import make_grid

from body_visualizer.tools.vis_tools import colors
from body_visualizer.mesh.mesh_viewer import MeshViewer
from body_visualizer.mesh.sphere import points_to_spheres
from body_visualizer.tools.vis_tools import show_image

c2c = lambda tensor: tensor.detach().cpu().numpy()

def overlay_text(image, txt_str, color=(0,0,255), str_id=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = image.shape[0]*0.0016
    thickness = int(image.shape[0]*0.005)
    bbox_offset = int(image.shape[0]*0.01)
    text_offset_x, text_offset_y = int(image.shape[1]*0.02), int(image.shape[0]*0.06*str_id)

    (text_width, text_height) = cv2.getTextSize(txt_str, font, fontScale=font_scale, thickness=thickness)[0]
    box_coords = ((text_offset_x, text_offset_y + bbox_offset), (text_offset_x + text_width + bbox_offset, text_offset_y - text_height - bbox_offset))

    cv2.rectangle(image, box_coords[0], box_coords[1], (255, 255, 255), cv2.FILLED)
    cv2.putText(image, txt_str, (text_offset_x, text_offset_y), font, font_scale, color, thickness)
    return image

def get_body_model(model_type, smpl_type='smplx'):
    from smplx import SMPLH, SMPLX
    return eval(f'{smpl_type.upper()}')(f'../data/body_models/{smpl_type}', num_betas=10, ext='pkl').to('cuda')

imw, imh = 500, 500

def visualize_body(vertices, faces, mesh_viewer, num_vertices=None, index=0, show_fig=False, side_view=False):
    num_vertices = vertices.shape[1]
    body_mesh = trimesh.Trimesh(vertices=c2c(vertices[index]), faces=c2c(faces), vertex_colors=np.tile(colors['grey'], (num_vertices, 1)))
    if side_view:
        rot = trimesh.transformations.rotation_matrix(np.radians(270), [0, 1, 0])
        body_mesh.apply_transform(rot)
    mesh_viewer.set_static_meshes([body_mesh])
    body_image = mesh_viewer.render(render_wireframe=False)
    if show_fig:
        show_image(body_image)
    else:
        return torch.FloatTensor(body_image.transpose(2,0,1).copy())
    
def visualize_from_pose(input_batch, output_batch, iter_str, save_dir=''):
    gt_pose_body = input_batch['pose_body_aa'].float().to('cuda')
    gt_betas = input_batch['betas'].float().to('cuda')
    gt_gender = input_batch['gender']
    pred_pose_body = output_batch['pred_pose_body_aa'].detach().clone()
    save_name = f'{save_dir}/results_{iter_str}.png'
    num_renders = 8
    imgs = []

    mesh_viewer = MeshViewer(width=imw, height=imh, use_offscreen=True)
    for pose_type in ['gt', 'pred']:
        for pose_idx in range(num_renders):
            pose_body = eval(f'{pose_type}_pose_body')[[pose_idx]]
            betas = gt_betas[[pose_idx]]
            gender = gt_gender[pose_idx]
            body_model = get_body_model(gender)
            smplx_body = body_model(pose_body=pose_body, betas=betas)
            imgs.append(visualize_body(smplx_body, mesh_viewer))

    rend_img = make_grid(imgs, nrow=num_renders)
    rend_img = rend_img.cpu().numpy().transpose(1, 2, 0)
    rend_img = np.clip(rend_img, 0, 255).astype(np.uint8)
    cv2.imwrite(save_name, cv2.cvtColor(rend_img, cv2.COLOR_BGR2RGB))
    mesh_viewer.viewer.delete()

def visualize_from_mesh(smpl_type, input_batch, output_batch, iter_str, save_dir=''):
    os.makedirs(save_dir, exist_ok=True)
    gt_pose_body = input_batch['pose_body_aa'].float().to('cuda')
    pred_vertices = output_batch['pred_body_vertices']
    save_name = f'{save_dir}/results_{iter_str}.png'
    num_renders = 8
    side_view = True
    imgs = []

    mesh_viewer = MeshViewer(width=imw, height=imh, use_offscreen=True)
    body_model = get_body_model('neutral', smpl_type)
    
    for pose_idx in range(num_renders):
        gt_pose = gt_pose_body[[pose_idx]]
        body = body_model(body_pose=gt_pose)
        vertices = body.vertices.detach()
        faces = body_model.faces
        imgs.append(visualize_body(vertices, faces, mesh_viewer))
    

    for mesh_idx in range(num_renders):
        imgs.append(visualize_body(pred_vertices, faces, mesh_viewer, index=mesh_idx))

    if side_view:
        for pose_idx in range(num_renders):
            gt_pose = gt_pose_body[[pose_idx]]
            body = body_model(body_pose=gt_pose, global_orient=torch.FloatTensor([[0,-1.4,0]]).cuda())
            vertices = body.vertices.detach()
            faces = body_model.faces
            imgs.append(visualize_body(vertices, faces, mesh_viewer))

        for mesh_idx in range(num_renders):
            imgs.append(visualize_body(pred_vertices, faces, mesh_viewer, index=mesh_idx, side_view=True))

    rend_img = make_grid(imgs, nrow=num_renders)
    rend_img = rend_img.cpu().numpy().transpose(1, 2, 0)
    rend_img = np.clip(rend_img, 0, 255).astype(np.uint8)
    rend_img = cv2.cvtColor(rend_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(save_name, rend_img)
    mesh_viewer.viewer.delete()

