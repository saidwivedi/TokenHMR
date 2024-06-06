import torch
import torch.nn as nn

from ..utils.rotation_utils import matrix_to_axis_angle


kp2D_err_valid_thresh = torch.Tensor([0.0085024, 0.00648666, 0.00747825, 0.01103439, 0.01355629, 0.00741691,\
                                0.01096735, 0.01414461, 0.00974212, 0.01127469, 0.01663222, 0.00564927,\
                                0.01126335, 0.01615757, 0.00532595, 0.00829731, 0.00831497, 0.00737241,\
                                0.00743286, 0.00543739, 0.00550524, 0.00535504, 0.00565414, 0.00581685,\
                                0.00573041, 0.00554029, 0.01515258, 0.00986267, 0.00997563, 0.01519944,\
                                0.00511402, 0.01288267, 0.01105894, 0.00710525, 0.00709785, 0.01092387,\
                                0.01388091, 0.00648326, 0.00766487, 0.00931454, 0.00646622, 0.00677057,\
                                0.00744011, 0.00752381])
angle_valid_thresh = {
    'body_pose': torch.Tensor([0.273709, 0.26481161, 0.1838198, 0.41490657, 0.37521194, \
                    0.20793171, 0.24905021, 0.33887333, 0.14481062, 0.35632194, 0.34944217, \
                    0.30542146, 0.32835298, 0.33110567, 0.34813467, 0.36357761, 0.40062272, \
                    0.43493496, 0.4400709, 0.78017052, 0.7375746, 0.24927082, 0.24966981]) * 0.8, 
    'global_orient': torch.Tensor([0.46])}

def joint_angle_error(pred_mat, gt_mat):
    n_frames, joint_num = pred_mat.shape[:2]
    # Reshape the matrices into B x 3 x 3 arrays
    r1 = pred_mat.reshape(-1, 3, 3)
    r2 = gt_mat.reshape(-1, 3, 3)
    # compute R1 * R2.T, if prediction and target match, this will be the identity matrix
    r = r1 @ r2.permute(0, 2, 1)
    # Convert rotation matrix to axis angle representation and find the angle
    r_aa = matrix_to_axis_angle(r).reshape(n_frames, joint_num, 3)
    angles = torch.linalg.norm(r_aa, dim=-1).reshape(n_frames, joint_num)

    return angles


class Keypoint2DLoss(nn.Module):

    def __init__(self, loss_type: str = 'l1'):
        """
        2D keypoint loss module.
        Args:
            loss_type (str): Choose between l1 and l2 losses.
        """
        super(Keypoint2DLoss, self).__init__()
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError('Unsupported loss function')

    def forward(self, pred_keypoints_2d: torch.Tensor, gt_keypoints_2d: torch.Tensor) -> torch.Tensor:
        """
        Compute 2D reprojection loss on the keypoints.
        Args:
            pred_keypoints_2d (torch.Tensor): Tensor of shape [B, S, N, 2] containing projected 2D keypoints (B: batch_size, S: num_samples, N: num_keypoints)
            gt_keypoints_2d (torch.Tensor): Tensor of shape [B, S, N, 3] containing the ground truth 2D keypoints and confidence.
        Returns:
            torch.Tensor: 2D keypoint loss.
        """
        conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
        batch_size = conf.shape[0]
        loss = (conf * self.loss_fn(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])).sum(dim=(1,2))
        return loss.sum()


class Keypoint3DLoss(nn.Module):

    def __init__(self, loss_type: str = 'l1'):
        """
        3D keypoint loss module.
        Args:
            loss_type (str): Choose between l1 and l2 losses.
        """
        super(Keypoint3DLoss, self).__init__()
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError('Unsupported loss function')

    def forward(self, pred_keypoints_3d: torch.Tensor, gt_keypoints_3d: torch.Tensor, pelvis_id: int = 39):
        """
        Compute 3D keypoint loss.
        Args:
            pred_keypoints_3d (torch.Tensor): Tensor of shape [B, S, N, 3] containing the predicted 3D keypoints (B: batch_size, S: num_samples, N: num_keypoints)
            gt_keypoints_3d (torch.Tensor): Tensor of shape [B, S, N, 4] containing the ground truth 3D keypoints and confidence.
        Returns:
            torch.Tensor: 3D keypoint loss.
        """
        batch_size = pred_keypoints_3d.shape[0]
        gt_keypoints_3d = gt_keypoints_3d.clone()
        pred_keypoints_3d = pred_keypoints_3d - pred_keypoints_3d[:, pelvis_id, :].unsqueeze(dim=1)
        gt_keypoints_3d[:, :, :-1] = gt_keypoints_3d[:, :, :-1] - gt_keypoints_3d[:, pelvis_id, :-1].unsqueeze(dim=1)
        conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
        gt_keypoints_3d = gt_keypoints_3d[:, :, :-1]
        loss = (conf * self.loss_fn(pred_keypoints_3d, gt_keypoints_3d)).sum(dim=(1,2))
        return loss.sum()

class Keypoint2DLossPCKT(nn.Module):

    def __init__(self, loss_type: str = 'l1'):
        """
        2D keypoint loss module.
        Args:
            loss_type (str): Choose between l1 and l2 losses.
        """
        super(Keypoint2DLossPCKT, self).__init__()
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError('Unsupported loss function')

    def forward(self, pred_keypoints_2d: torch.Tensor, gt_keypoints_2d: torch.Tensor, weak_mask=None, LOOSE_WEIGHT=0.01) -> torch.Tensor:
        """
        Compute 2D reprojection loss on the keypoints.
        Args:
            pred_keypoints_2d (torch.Tensor): Tensor of shape [B, S, N, 2] containing projected 2D keypoints (B: batch_size, S: num_samples, N: num_keypoints)
            gt_keypoints_2d (torch.Tensor): Tensor of shape [B, S, N, 3] containing the ground truth 2D keypoints and confidence.
        Returns:
            torch.Tensor: 2D keypoint loss.
        """
        conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
        batch_size = conf.shape[0]
        loss = (conf * self.loss_fn(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])).sum()
        if weak_mask is not None:
            weak_loose_loss = LOOSE_WEIGHT * (weak_mask.unsqueeze(-1) * self.loss_fn(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])).sum()
            loss = loss + weak_loose_loss
        return loss


class Keypoint3DLossPCKT(nn.Module):

    def __init__(self, loss_type: str = 'l1'):
        """
        3D keypoint loss module.
        Args:
            loss_type (str): Choose between l1 and l2 losses.
        """
        super(Keypoint3DLossPCKT, self).__init__()
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError('Unsupported loss function')

    def forward(self, pred_keypoints_3d: torch.Tensor, gt_keypoints_3d: torch.Tensor, pelvis_id: int = 39):
        """
        Compute 3D keypoint loss.
        Args:
            pred_keypoints_3d (torch.Tensor): Tensor of shape [B, S, N, 3] containing the predicted 3D keypoints (B: batch_size, S: num_samples, N: num_keypoints)
            gt_keypoints_3d (torch.Tensor): Tensor of shape [B, S, N, 4] containing the ground truth 3D keypoints and confidence.
        Returns:
            torch.Tensor: 3D keypoint loss.
        """
        batch_size = pred_keypoints_3d.shape[0]
        gt_keypoints_3d = gt_keypoints_3d.clone()
        pred_keypoints_3d = pred_keypoints_3d - pred_keypoints_3d[:, pelvis_id, :].unsqueeze(dim=1)
        gt_keypoints_3d[:, :, :-1] = gt_keypoints_3d[:, :, :-1] - gt_keypoints_3d[:, pelvis_id, :-1].unsqueeze(dim=1)
        conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
        gt_keypoints_3d = gt_keypoints_3d[:, :, :-1]
        loss = (conf * self.loss_fn(pred_keypoints_3d, gt_keypoints_3d)).sum(dim=(1,2))
        return loss.sum()

class ParameterLoss(nn.Module):

    def __init__(self):
        """
        SMPL parameter loss module.
        """
        super(ParameterLoss, self).__init__()
        self.loss_fn = nn.MSELoss(reduction='none')

    def forward(self, pred_param: torch.Tensor, gt_param: torch.Tensor, has_param: torch.Tensor):
        """
        Compute SMPL parameter loss.
        Args:
            pred_param (torch.Tensor): Tensor of shape [B, S, ...] containing the predicted parameters (body pose / global orientation / betas)
            gt_param (torch.Tensor): Tensor of shape [B, S, ...] containing the ground truth SMPL parameters.
        Returns:
            torch.Tensor: L2 parameter loss loss.
        """
        batch_size = pred_param.shape[0]
        num_dims = len(pred_param.shape)
        mask_dimension = [batch_size] + [1] * (num_dims-1)
        has_param = has_param.type(pred_param.type()).view(*mask_dimension)
        loss_param = (has_param * self.loss_fn(pred_param, gt_param))
        return loss_param.sum()

class ParameterLossPCKT(nn.Module):

    def __init__(self):
        """
        SMPL parameter loss module.
        """
        super(ParameterLossPCKT, self).__init__()
        self.loss_fn = nn.MSELoss(reduction='none')

    def forward(self, pred_param: torch.Tensor, gt_param: torch.Tensor, has_param: torch.Tensor, valid_mask: torch.Tensor = None, weak_mask=None, LOOSE_WEIGHT=0.01):
        """
        Compute SMPL parameter loss.
        Args:
            pred_param (torch.Tensor): Tensor of shape [B, S, ...] containing the predicted parameters (body pose / global orientation / betas)
            gt_param (torch.Tensor): Tensor of shape [B, S, ...] containing the ground truth SMPL parameters.
        Returns:
            torch.Tensor: L2 parameter loss loss.
        """
        if valid_mask is not None:
            #print(valid_mask.shape, self.loss_fn(pred_param, gt_param).shape, self.loss_fn(pred_param, gt_param).sum((2,3)).shape)
            loss_param = (valid_mask * self.loss_fn(pred_param, gt_param).sum((2,3))).sum()
            if weak_mask is not None:
                batch_size, kp_num = pred_param.shape[:2]
                #print(self.loss_fn(pred_param, gt_param).sum((2,3)).shape, weak_mask.shape)
                weak_loose_loss = LOOSE_WEIGHT * (weak_mask * self.loss_fn(pred_param, gt_param).sum((2,3))).sum()
                
                loss_param = loss_param + weak_loose_loss
        else:
            pred_param, gt_param = pred_param.reshape(pred_param.shape[0], -1), gt_param.reshape(gt_param.shape[0], -1)
            batch_size = pred_param.shape[0]
            num_dims = len(pred_param.shape)
            mask_dimension = [batch_size] + [1] * (num_dims-1)
            has_param = has_param.type(pred_param.type()).view(*mask_dimension)
            loss_param = (has_param * self.loss_fn(pred_param, gt_param)).sum()
        return loss_param

class TokenLoss(nn.Module):

    def __init__(self):
        """
        SMPL parameter loss module.
        """
        super(TokenLoss, self).__init__()
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, pred_cls_logits_softmax: torch.Tensor, gt_tokens: torch.Tensor):
        """
        Compute SMPL Token classification loss.
        Args:
            pred_cls_logits_softmax (torch.Tensor): Tensor of shape [B, token_num, codebook_class_num] containing the predicted likelihood of each token class.
            gt_tokens (torch.Tensor): Tensor of shape [B, token_num] containing the ground truth discrete SMPL tokens.
        Returns:
            torch.Tensor: token classification loss.
        """
        batch_size, token_num, token_class_num = pred_cls_logits_softmax.shape
        # problem is that the order of predicted tokens mis-match with gt tokens. So we have to match them.
        # But it turns out the the order of SMPL tokens has meanings. 
        loss_param = self.loss_fn(pred_cls_logits_softmax.reshape((batch_size*token_num), token_class_num), gt_tokens.reshape(batch_size*token_num))
        return loss_param.sum()

class VerticesLoss(nn.Module):

    def __init__(self, loss_type: str = 'l1'):
        """
        3D vertices loss module.
        Args:
            loss_type (str): Choose between l1 and l2 losses.
        """
        super(VerticesLoss, self).__init__()
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='mean')
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss(reduction='mean')
        else:
            raise NotImplementedError('Unsupported loss function')

    def forward(self, pred_vertices: torch.Tensor, gt_vertices: torch.Tensor):
        """
        Compute 3D vertices loss.
        """
        batch_size = pred_vertices.shape[0]
        gt_vertices = gt_vertices.clone()
        loss = self.loss_fn(pred_vertices, gt_vertices)
        return loss

