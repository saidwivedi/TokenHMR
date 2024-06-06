import torch

from .geometry import perspective_projection
from .rotation_utils import axis_angle_to_matrix

JOINT_NAMES = [
    'OP Nose', 'OP Neck', 'OP RShoulder', 'OP RElbow', 'OP RWrist', 'OP LShoulder',
    'OP LElbow', 'OP LWrist', 'OP MidHip', 'OP RHip', 'OP RKnee', 'OP RAnkle',
    'OP LHip', 'OP LKnee', 'OP LAnkle', 'OP REye', 'OP LEye', 'OP REar', 'OP LEar',
    'OP LBigToe', 'OP LSmallToe','OP LHeel', 'OP RBigToe', 'OP RSmallToe', 'OP RHeel',
    # 19 Ground Truth joints (superset of joints from different datasets)
    'Right Ankle', 'Right Knee', 'Right Hip', 'Left Hip', 'Left Knee', 'Left Ankle',
    'Right Wrist', 'Right Elbow', 'Right Shoulder', 'Left Shoulder', 'Left Elbow', 'Left Wrist',
    'Neck (LSP)', 'Top of Head (LSP)', 'Pelvis (MPII)', 'Thorax (MPII)', 'Spine (H36M)', 'Jaw (H36M)', 'Head (H36M)']
# Dict containing the joints in numerical order
JOINT_IDS = {JOINT_NAMES[i]: i for i in range(len(JOINT_NAMES))} 

def camera_fitting_loss(model_joints, pred_cam_t, focal_length, joints_2d):
    """
    Loss function for camera optimization.
    """

    # Project model joints
    batch_size = model_joints.shape[0]
    projected_joints = perspective_projection(model_joints,
                                                   translation=pred_cam_t,
                                                   focal_length=focal_length / 256)
    
    reprojection_error = torch.sqrt(((joints_2d - projected_joints) ** 2).sum(-1)).sum(1)
    return reprojection_error.mean()

class SMPLifyInv():
    """Implementation of single-stage SMPLify.""" 
    def __init__(self, smpl_model,
                 step_size = 1e-3,
                 num_iters = 100,
                 margin = 20,
                 loss_thresh_f2d = 1,
                 loss_thresh_f3d = 0,
                 device=torch.device('cuda')):

        # Store options
        self.device = device
        self.step_size = step_size
        self.num_iters = num_iters
        self.smpl = smpl_model
        self.margin = margin
        self.loss_thresh_f2d = loss_thresh_f2d
        self.loss_thresh_f3d = loss_thresh_f3d

    def __call__(self, global_orient, body_pose, betas, pred_cam_t, focal_length, gt_keypoints_2d, gt_keypoints_3d):
        """Perform body fitting.
        Input:
            init_pose: SMPL pose estimate
            init_betas: SMPL betas estimate
            init_cam_t: Camera translation estimate
            camera_center: Camera center location
            gt_keypoints_2d: Keypoints used for the optimization
        Returns:
            vertices: Vertices of optimized shape
            joints: 3D joints of optimized shape
            pose: SMPL pose parameters of optimized shape
            betas: SMPL beta parameters of optimized shape
            camera_translation: Camera translation
            reprojection_loss: Final joint reprojection loss
        """
        # Get joint confidence
        joints_2d = gt_keypoints_2d[:, :, :2]
        joints_conf = gt_keypoints_2d[:, :, -1]

        # # Step 1: Optimize camera translation and body orientation
        # # Optimize only camera translation and body orientation
        # body_pose.requires_grad=False
        # betas.requires_grad=False
        # global_orient.requires_grad=False
        # pred_cam_t.requires_grad = True

        # camera_opt_params = [pred_cam_t]
        # camera_optimizer = torch.optim.Adam(camera_opt_params, lr=self.step_size, betas=(0.9, 0.999))

        # for i in range(self.num_iters):
        #     smpl_output = self.smpl(global_orient=axis_angle_to_matrix(global_orient).float(),
        #                             body_pose=axis_angle_to_matrix(body_pose.reshape(-1,23,3)).float(),
        #                             betas=betas.float())
        #     model_joints = smpl_output.joints
        #     loss = camera_fitting_loss(model_joints, pred_cam_t, focal_length, joints_2d)
        #     print('fitting cam', loss.item())
        #     if loss.item() < self.loss_thresh_f2d:
        #         break
        #     camera_optimizer.zero_grad()
        #     loss.backward()
        #     camera_optimizer.step()

        # smpl_output = self.smpl(global_orient=axis_angle_to_matrix(global_orient).float(),
        #                             body_pose=axis_angle_to_matrix(body_pose.reshape(-1,23,3)).float(),
        #                             betas=betas.float())
        # vertices_f2d = smpl_output.vertices.detach()
        # joints_f2d = smpl_output.joints.detach()  
        # batch_size = joints_f2d.shape[0]
        # pred_cam_t_f2d = pred_cam_t.clone()
        # pj2ds_f2d = perspective_projection(joints_f2d,
        #                                            translation=pred_cam_t_f2d,
        #                                            focal_length=focal_length / 256).reshape(batch_size, -1, 2)

        # Step 2: Optimize body joints to push away 3D and align 2D
        # Optimize only the body pose and global orientation of the body
        body_pose.requires_grad = True
        betas.requires_grad = False
        global_orient.requires_grad = True
        pred_cam_t.requires_grad = True
        body_opt_params = [body_pose, global_orient, pred_cam_t]
        
        body_optimizer = torch.optim.Adam(body_opt_params, lr=self.step_size, betas=(0.9, 0.999))
        for i in range(self.num_iters):
            # smpl_output = self.smpl(global_orient=axis_angle_to_matrix(global_orient).float(),
            #                         body_pose=axis_angle_to_matrix(body_pose.reshape(-1,23,3)).float(),
            #                         betas=betas.float())
            smpl_output = self.smpl(global_orient=global_orient,
                                    body_pose=body_pose,
                                    betas=betas)
            model_joints = smpl_output.joints
            fit2D = camera_fitting_loss(model_joints, pred_cam_t, focal_length, joints_2d)
            push3D = torch.sqrt(((model_joints - gt_keypoints_3d) ** 2).sum(2)).sum(1)
            loss = 4*fit2D - push3D.mean() /2 + self.margin
            if torch.isnan(loss).sum() is None:
                break
            #print('fitting 2D & push 3D {:.2f} | fitting 2D {:.2f} | push {:.2f}'.format(loss.item(), fit2D, push3D.mean()))
            if loss.item() < self.loss_thresh_f3d and fit2D.item() < self.loss_thresh_f2d:
                break
            body_optimizer.zero_grad()
            loss.backward()
            body_optimizer.step()
        #print('fitting 2D & push 3D {:.2f} | fitting 2D {:.2f} | push {:.2f}'.format(loss.item(), fit2D, push3D.mean()))
        # Get final loss value
        with torch.no_grad():
            # smpl_output = self.smpl(global_orient=axis_angle_to_matrix(global_orient).float(),
            #                         body_pose=axis_angle_to_matrix(body_pose.reshape(-1,23,3)).float(),
            #                         betas=betas.float())
            smpl_output = self.smpl(global_orient=global_orient,
                                    body_pose=body_pose,
                                    betas=betas)
            model_joints = smpl_output.joints
            reprojection_loss = camera_fitting_loss(model_joints, pred_cam_t, focal_length, joints_2d)

        vertices = smpl_output.vertices.detach()
        joints = smpl_output.joints.detach()
        pj2ds = perspective_projection(joints,
                                                   translation=pred_cam_t,
                                                   focal_length=focal_length / 256).reshape(joints.shape[0], -1, 2)

        global_orient = global_orient.detach()
        body_pose = body_pose.detach()
        betas = betas.detach()

        #vertices_f2d, joints_f2d, pj2ds_f2d, pred_cam_t_f2d, 
        return vertices, joints, pj2ds, global_orient, body_pose, betas, pred_cam_t, reprojection_loss