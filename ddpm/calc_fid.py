import torch
import sys
from pytorch_fid import fid_score

torch.cuda.set_device(0)

real_images_folder = sys.argv[1]
generated_images_folder = sys.argv[2]


fid_value = fid_score.calculate_fid_given_paths([real_images_folder, generated_images_folder],batch_size=50, device='cuda', dims=2048)

print('FID value:', fid_value)

