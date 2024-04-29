import open3d as o3d
import numpy as np

base_path = '/home/cvlab05/project/hg_nerf/gaussian-splatting_test/output/bicycle_rgbfix_lr_change'
input_pc_path = base_path + '/input.ply'
output_pc_path = base_path + '/point_cloud/iteration_30000/point_cloud.ply'
input_pcd = o3d.io.read_point_cloud(input_pc_path)
output_pcd = o3d.io.read_point_cloud(output_pc_path)
input_pcd = np.asarray(input_pcd.points)
output_pcd = np.asarray(output_pcd.points)

input_pcd_x = input_pcd[:, 0]
input_pcd_y = input_pcd[:, 1]
input_pcd_z = input_pcd[:, 2]

output_pcd_x = output_pcd[:, 0]
output_pcd_y = output_pcd[:, 1]
output_pcd_z = output_pcd[:, 2]

diff_x = np.mean(np.abs(input_pcd_x - output_pcd_x))
diff_y = np.mean(np.abs(input_pcd_y - output_pcd_y))
diff_z = np.mean(np.abs(input_pcd_z - output_pcd_z))

print('diff_x: ', diff_x)
print('diff_y: ', diff_y)
print('diff_z: ', diff_z)