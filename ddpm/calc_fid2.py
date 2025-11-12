import torch
import sys
import os
from pytorch_fid import fid_score

torch.cuda.set_device(0)

real_images_folder = sys.argv[1]
generated_images_folder = sys.argv[2]

real_subfolders = [f.path for f in os.scandir(real_images_folder) if f.is_dir()]
generated_subfolders = [f.path for f in os.scandir(generated_images_folder) if f.is_dir()]

fid_values = []

for real_subfolder, generated_subfolder in zip(real_subfolders, generated_subfolders):
    fid_value = fid_score.calculate_fid_given_paths([real_subfolder, generated_subfolder], batch_size=50, device='cuda', dims=2048)
    fid_values.append(fid_value)

for i, fid_value in enumerate(fid_values):
    print(f'FID value for subfolder {i + 1}: {fid_value}')

average_fid_value = sum(fid_values) / len(fid_values)
print('Average FID value:', average_fid_value)





