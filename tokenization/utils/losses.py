import numpy as np
import torch
import torch.nn as nn

class Geodesic_Loss(nn.Module):
    def __init__(self, reduction='mean'):
        super(Geodesic_Loss, self).__init__()

        self.reduction = reduction
        self.eps = 1e-6

    # batch geodesic loss for rotation matrices
    def bgdR(self,m1,m2):
        batch = m1.shape[0]
        m = torch.bmm(m1, m2.transpose(1, 2))  # batch*3*3

        cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
        cos = torch.min(cos, m1.new(np.ones(batch)))
        cos = torch.max(cos, m1.new(np.ones(batch)) * -1)

        return torch.acos(cos)

    def forward(self, pred_rot, gt_rot):
        theta = self.bgdR(pred_rot.view(-1, 3, 3), gt_rot.view(-1, 3, 3))
        if self.reduction == 'mean':
            return torch.mean(theta)
        else:
            return theta

class WeightedMSE(nn.Module):
    def __init__(self, weights, reduction='mean'):
        super(WeightedMSE, self).__init__()

        self.register_buffer('weights', torch.tensor(weights, dtype=torch.float32).cuda())
        self.reduction = reduction

    def forward(self, input, target):
        if self.reduction == 'mean':
            return torch.mean(self.weights * (input - target) ** 2)


class RotDist(nn.Module):
    def __init__(self, reduction='mean'):
        super(RotDist, self).__init__()

        self.reduction = reduction
        self.epsilon = 1e-6

    def forward(self, gt_pose, pred_pose):
        '''
        pose type: rotation matrix
        http://www.boris-belousov.net/2016/12/01/quat-dist/#using-rotation-matrices
        '''
        tr = torch.einsum(
                'bij,bij->b',
                [pred_pose.reshape(-1, 3, 3),
                gt_pose.reshape(-1, 3, 3)])
        theta = (tr - 1) * 0.5
        loss = torch.acos(
            torch.clamp(theta, -1 + self.epsilon, 1 - self.epsilon))
        if self.reduction == 'mean':
            return loss.mean()
        return loss
        
class PoseReConsLoss(nn.Module):
    def __init__(self, loss_params, nb_joints, rot_type, smpl_type):
        super(PoseReConsLoss, self).__init__()
        
        self.nb_joints = nb_joints
        self.rot_type = rot_type
        self.smpl_type = smpl_type

        self.mesh_loss = self.get_loss(loss_params.MESH_LOSS, 'mesh')
        self.pose_loss = self.get_loss(loss_params.POSE_LOSS, 'pose')
        self.jnt_loss = self.get_loss(loss_params.JNT_LOSS, 'jnt')

        self.valid_joints = None
        if loss_params.ONLY_VALID_JNT:
            self.valid_joints = [*range(1,22)]  # only body joints

    def get_loss(self, loss, var):
        if loss == 'l1':
            return torch.nn.L1Loss()
        elif loss == 'l2':
            return torch.nn.MSELoss()
        elif loss == 'l1_smooth':
            return torch.nn.SmoothL1Loss()
        elif loss == 'geodesic' and var == 'pose':
            return Geodesic_Loss()
        elif loss == 'wt_l2' and var == 'mesh':
            vertex_weights = self.calculate_vertex_weights()
            return WeightedMSE(vertex_weights)
        elif loss == 'wt_l2' and var == 'jnt':
            weights = np.ones((self.nb_joints,3))
            #weights[:3] *= 5
            return WeightedMSE(weights)
        elif loss == 'wt_l2' and var == 'pose':
            weights = np.ones((self.nb_joints,3,3))
            #weights[:3] *= 5
            return WeightedMSE(weights)
        elif loss == 'rotdist' and var == 'pose':
            return RotDist()
        else:
            return torch.nn.MSELoss()
    
    def calculate_vertex_weights(self):
        from smplx import SMPLH, SMPLX
        body_model = eval(f'{self.smpl_type.upper()}')(f'../data/body_models/{self.smpl_type}', num_betas=10, ext='pkl')
        mesh = body_model()
        vertices, faces = mesh.vertices[0].detach().numpy(), body_model.faces
        v1, v2, v3 = vertices[faces[:,0]], vertices[faces[:,1]], vertices[faces[:,2]]
        cross_product = np.cross(v2 - v1, v3 - v1)
        tri_area = 0.5 * np.linalg.norm(cross_product, axis=1)
        norm_triangle_area = (tri_area - np.min(tri_area)) / (np.max(tri_area) - np.min(tri_area))
        vertex_weights = np.zeros((vertices.shape[0], 1))
        for i, face in enumerate(faces):
            for vertex_index in face:
                vertex_weights[vertex_index] += norm_triangle_area[i]
        return np.repeat(vertex_weights, 3, axis=1)

    def forward_pose(self, gt_pose_body, output_batch) : 
        return self.pose_loss(gt_pose_body, output_batch['pred_pose_body_rotmat'])

    def forward_mesh(self, gt_mesh, output_batch):
        return self.mesh_loss(gt_mesh, output_batch['pred_body_vertices'])
    
    def forward_joints(self, gt_jnts, output_batch):
        if self.valid_joints is None:
            return self.jnt_loss(gt_jnts, output_batch['pred_body_joints'])
        else:
            return self.jnt_loss(gt_jnts[:,self.valid_joints], output_batch['pred_body_joints'][:,self.valid_joints])
    
    