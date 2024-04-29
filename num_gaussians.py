import os
import numpy as np
import matplotlib.pyplot as plt


''''In one iteration, check the number of gaussians per iteration'''
base_path = '/home/cvlab05/project/hg_nerf/gaussian-splatting_test/output/lego_original/'
num_gaussians_path = os.path.join(base_path, 'num_gaussians_30000.npy')
if os.path.exists(num_gaussians_path) == False:
    num_gaussians_path = os.path.join(base_path, 'num_gaussians.npy')
num_gaussians = np.load(num_gaussians_path)
x_values = np.arange(1,num_gaussians.shape[0]+1)

plt.plot(x_values, num_gaussians)
plt.xlabel('iteration')
plt.ylabel('Number of Gaussians')
plt.title('Number of Gaussians per iteration')
plt.savefig(os.path.join(base_path,'num_gaussians_30000.png'), dpi=300)


'''In multiple iterations, check the number of gaussians after optimization'''
# base_path = '/home/cvlab05/project/hg_nerf/gaussian-splatting_test/output/lego_anal_sparsify_full_shot'
# densify_iteration = np.arange(1000,30001,1000)
# num_gaussians_after_optimization = []

# for i in densify_iteration:
#     num_gaussians_path = os.path.join(base_path, f'num_gaussians_{i}.npy')
#     num_gaussians = np.load(num_gaussians_path)
#     num_gaussians_after_optimization.append(num_gaussians[-1])

# plt.plot(densify_iteration, num_gaussians_after_optimization)
# plt.xlabel('densify end iteration')
# plt.ylabel('Number of Gaussians')
# plt.title('Number of Gaussians after optimization')
# plt.savefig(os.path.join(base_path,'num_gaussians_after_optimization.png'), dpi=300)
# plt.close()