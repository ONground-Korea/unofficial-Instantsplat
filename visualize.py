import open3d as o3d
# Monkey-patch torch.utils.tensorboard.SummaryWriter
from open3d.visualization.tensorboard_plugin import summary
# Utility function to convert Open3D geometry to a dictionary format
from open3d.visualization.tensorboard_plugin.util import to_dict_batch
from torch.utils.tensorboard import SummaryWriter
from plyfile import PlyData
import numpy as np

# cube = o3d.geometry.TriangleMesh.create_box(1, 2, 4)
# cube.compute_vertex_normals()
# cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=1.0,
#                                                      height=2.0,
#                                                      resolution=20,
#                                                      split=4)
# cylinder.compute_vertex_normals()
# colors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]

ply_path = '/home/cvlab05/project/hg_nerf/gaussian-splatting_test/output/bicycle_xyzfix_posrandominit_many/point_cloud/iteration_30000/point_cloud.ply'
pcd = o3d.io.read_point_cloud(ply_path)
## get color from pcd
plydata = PlyData.read(ply_path)
xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
try:
    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    ## draw color to pcd
    pcd.colors = o3d.utility.Vector3dVector(features_dc.squeeze())
except:
    print('no color')


logdir = "demo_logs"
writer = SummaryWriter(logdir)
for step in range(3):
    writer.add_3d('pcd', to_dict_batch([pcd]), step=step)