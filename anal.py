import os
import numpy as np
import matplotlib.pyplot as plt

base_path = '/home/cvlab05/project/hg_nerf/gaussian-splatting_test/output'
scenes = ['lego', 'mic', 'chair', 'drums', 'ficus', 'hotdog', 'materials', 'ship']
avg_psnr_values = []
for scene in scenes:
    psnr_values_path = os.path.join(base_path, f'{scene}_anal', 'psnr_values.npy')
    psnr_values = np.load(psnr_values_path)
    if scene == 'lego':
        avg_psnr_values = psnr_values
    else:
        avg_psnr_values += psnr_values
    x_values = np.arange(5, 101, 5)
    plt.plot(x_values, psnr_values)
    plt.xlabel('number of views')
    plt.ylabel('psnr')
    plt.title(f'Plot of psnr values of scene {scene}')
    plt.savefig(os.path.join(base_path, f'{scene}_anal',f'{scene}_psnr.png'), dpi=300)
    plt.close()

avg_psnr_values /= len(scenes)
plt.plot(x_values, avg_psnr_values)
plt.xlabel('number of views')
plt.ylabel('psnr')
plt.title('Plot of average psnr values')
plt.savefig(os.path.join(base_path, 'average_psnr.png'), dpi=300)
