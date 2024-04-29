import numpy as np
import torch
import torch.nn as nn

class CameraPoses(nn.Module):
    def __init__(self, poses,pose_rep="9D"):
        super(CameraPoses, self).__init__()
        self.poses = poses
        self.pose_rep = pose_rep
        if pose_rep=="9D":
            poses = [pred_camera_pose.world_view_transform.transpose(1,0) for pred_camera_pose in poses]
            self.w2c = torch.stack(poses, dim=0)
            self.d9 = self.pose_to_d9(self.w2c)
            self.d9 = nn.Parameter(self.d9, requires_grad=True)
        
        elif pose_rep=='quaternion':
            self.R = np.array([pred_camera_pose.R.T for pred_camera_pose in poses])
            self.t = np.array([pred_camera_pose.T for pred_camera_pose in poses])

            self.R = torch.from_numpy(self.R).float().cuda()
            self.t = torch.from_numpy(self.t).float().cuda()
            
            w2c = torch.zeros((len(self.poses), 4, 4))
            w2c[:, :3, :3] = self.R
            w2c[:, :3, 3] = self.t
            w2c[:, 3, 3] = 1.0

            w2c_q, w2c_t = self.SE3_to_quaternion_and_translation_torch(w2c)



            self.q = nn.Parameter(w2c_q, requires_grad=True)
            self.t = nn.Parameter(w2c_t, requires_grad=True)
            
        else:
            raise ValueError("pose_rep should be either 9D' or 'quaternion'")
            
        
    
    def forward(self,i):
        if self.pose_rep=="9D":
            return self.d9[i]
        elif self.pose_rep=="quaternion":
            rot_matrix = self.quaternion_to_rotation_matrix_torch(self.q[i])
            return rot_matrix, self.t[i]
        else:
            return None

    def __len__(self):
        return len(self.poses)

    def get_all_poses(self):
        if self.pose_rep=="9D":
            return self.d9
        elif self.pose_rep=="quaternion":
            rot_matrix = self.quaternion_to_rotation_matrix_torch(self.q)
            return rot_matrix, self.t
        else:
            return None
    
    def pose_to_d9(self, pose: torch.Tensor) -> torch.Tensor:
        """Converts rotation matrix to 9D representation. 

        We take the two first ROWS of the rotation matrix, 
        along with the translation vector.
        ATTENTION: row or vector needs to be consistent from pose_to_d9 and r6d2mat
        """
        nbatch = pose.shape[0]
        R = pose[:, :3, :3]  # [N, 3, 3]
        t = pose[:, :3, -1]  # [N, 3]

        r6 = R[:, :2, :3].reshape(nbatch, -1)  # [N, 6]

        d9 = torch.cat((t, r6), -1)  # [N, 9]
        # first is the translation vector, then two first ROWS of rotation matrix

        return d9
    
    def normalize_quat(self):
        with torch.no_grad():
            self.q  /= torch.norm(
                self.q , dim=-1, keepdim=True)

    def SE3_to_quaternion_and_translation_torch(
        self,
        transform: torch.Tensor,  # (batch_size, 4, 4)
    ):
        R = transform[..., :3, :3]
        t = transform[..., :3, 3]
        q = self.rotation_matrix_to_quaternion_torch(R)
        return q, t
    
    def rotation_matrix_to_quaternion_torch(
        self,
        R: torch.Tensor  # (batch_size, 3, 3)
    ) -> torch.Tensor:
        q = torch.zeros(R.shape[0], 4, device=R.device,
                        dtype=R.dtype)  # (batch_size, 4) x, y, z, w
        trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
        q0_mask = trace > 0
        q1_mask = (R[..., 0, 0] > R[..., 1, 1]) & (
            R[..., 0, 0] > R[..., 2, 2]) & ~q0_mask
        q2_mask = (R[..., 1, 1] > R[..., 2, 2]) & ~q0_mask & ~q1_mask
        q3_mask = ~q0_mask & ~q1_mask & ~q2_mask
        if q0_mask.any():
            R_for_q0 = R[q0_mask]
            S_for_q0 = 0.5 / torch.sqrt(1 + trace[q0_mask])
            q[q0_mask, 3] = 0.25 / S_for_q0
            q[q0_mask, 0] = (R_for_q0[..., 2, 1] - R_for_q0[..., 1, 2]) * S_for_q0
            q[q0_mask, 1] = (R_for_q0[..., 0, 2] - R_for_q0[..., 2, 0]) * S_for_q0
            q[q0_mask, 2] = (R_for_q0[..., 1, 0] - R_for_q0[..., 0, 1]) * S_for_q0

        if q1_mask.any():
            R_for_q1 = R[q1_mask]
            S_for_q1 = 2.0 * \
                torch.sqrt(1 + R_for_q1[..., 0, 0] -
                        R_for_q1[..., 1, 1] - R_for_q1[..., 2, 2])
            q[q1_mask, 0] = 0.25 * S_for_q1
            q[q1_mask, 1] = (R_for_q1[..., 0, 1] + R_for_q1[..., 1, 0]) / S_for_q1
            q[q1_mask, 2] = (R_for_q1[..., 0, 2] + R_for_q1[..., 2, 0]) / S_for_q1
            q[q1_mask, 3] = (R_for_q1[..., 2, 1] - R_for_q1[..., 1, 2]) / S_for_q1

        if q2_mask.any():
            R_for_q2 = R[q2_mask]
            S_for_q2 = 2.0 * \
                torch.sqrt(1 + R_for_q2[..., 1, 1] -
                        R_for_q2[..., 0, 0] - R_for_q2[..., 2, 2])
            q[q2_mask, 0] = (R_for_q2[..., 0, 1] + R_for_q2[..., 1, 0]) / S_for_q2
            q[q2_mask, 1] = 0.25 * S_for_q2
            q[q2_mask, 2] = (R_for_q2[..., 1, 2] + R_for_q2[..., 2, 1]) / S_for_q2
            q[q2_mask, 3] = (R_for_q2[..., 0, 2] - R_for_q2[..., 2, 0]) / S_for_q2

        if q3_mask.any():
            R_for_q3 = R[q3_mask]
            S_for_q3 = 2.0 * \
                torch.sqrt(1 + R_for_q3[..., 2, 2] -
                        R_for_q3[..., 0, 0] - R_for_q3[..., 1, 1])
            q[q3_mask, 0] = (R_for_q3[..., 0, 2] + R_for_q3[..., 2, 0]) / S_for_q3
            q[q3_mask, 1] = (R_for_q3[..., 1, 2] + R_for_q3[..., 2, 1]) / S_for_q3
            q[q3_mask, 2] = 0.25 * S_for_q3
            q[q3_mask, 3] = (R_for_q3[..., 1, 0] - R_for_q3[..., 0, 1]) / S_for_q3
        return q
    
    def quaternion_to_rotation_matrix_torch(self,q):
        """
        Convert a quaternion into a full three-dimensional rotation matrix.

        Input:
        :param q: A tensor of size (B, 4), where B is batch size and quaternion is in format (x, y, z, w).

        Output:
        :return: A tensor of size (B, 3, 3), where B is batch size.
        """
        # Ensure quaternion has four components
        assert q.shape[-1] == 4, "Input quaternion should have 4 components!"

        x, y, z, w = q.unbind(-1)

        # Compute quaternion norms
        q_norm = torch.norm(q, dim=-1, keepdim=True)
        # Normalize input quaternions
        q = q / q_norm

        # Compute the quaternion outer product
        q_outer = torch.einsum('...i,...j->...ij', q, q)

        # Compute rotation matrix
        rot_matrix = torch.empty(
            (*q.shape[:-1], 3, 3), dtype=q.dtype, device=q.device)
        rot_matrix[..., 0, 0] = 1 - 2 * (y**2 + z**2)
        rot_matrix[..., 0, 1] = 2 * (x*y - z*w)
        rot_matrix[..., 0, 2] = 2 * (x*z + y*w)
        rot_matrix[..., 1, 0] = 2 * (x*y + z*w)
        rot_matrix[..., 1, 1] = 1 - 2 * (x**2 + z**2)
        rot_matrix[..., 1, 2] = 2 * (y*z - x*w)
        rot_matrix[..., 2, 0] = 2 * (x*z - y*w)
        rot_matrix[..., 2, 1] = 2 * (y*z + x*w)
        rot_matrix[..., 2, 2] = 1 - 2 * (x**2 + y**2)

        return rot_matrix