import os
import numpy as np
import matplotlib.pyplot as plt
scene = 'lego'
x_values = np.arange(1000,30001,1000)
base_path = '/home/cvlab05/project/hg_nerf/gaussian-splatting_test/output/lego_anal_sparsify_full_shot/'
psnr_values = np.load(os.path.join(base_path,'psnr_values.npy'))
plt.plot(x_values, psnr_values)
plt.xlabel('densify end iteration')
plt.ylabel('test dataset psnr')
plt.title(f'Plot of psnr values of lego scene - full shot')
plt.savefig(os.path.join(base_path,f'{scene}_psnr_sparsify_full_shot.png'), dpi=300)
plt.close()