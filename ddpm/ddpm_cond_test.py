import os
import sys
import torch
import torch.nn as nn
import cv2
import time
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from utils2 import *
# from modules import UNet

from modules import UNet_mask
import logging

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

class Diffusion:

    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        self.img_size = img_size
        self.device = device

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n, masks, labels=None, cfg_scale=3):

        logging.info(f"Sampling {n} new images....")
        model.eval()

        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, masks, t, labels)    #

                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, masks, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)

                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]

                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)

                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)

        return x

'''
CUDA_VISIBLE_DEVICES=4 python ddpm_cond_test.py --dataset mri1 --num_classes 2 --image_size 256 --channels 1 --output /mnt/res/data/mri1/diffusion/ --image_path /mnt/mri1/data1/images/ --mask_path /mnt/mri1/data1/masks/ --batch_size 4

CUDA_VISIBLE_DEVICES=5 python ddpm_cond_test.py --dataset mri2 --num_classes 2 --image_size 256 --channels 2 --output /mnt/res/data/mri2/diffusion/ --image_path /mnt/mri2/data2/images/ --mask_path /mnt/mri2/data2/masks/ --batch_size 4
'''

def test():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('--image_path', type=str, default='./', help='dataset root dir')
    parser.add_argument('--mask_path', type=str, default='./', help='dataset root dir')
    parser.add_argument('--batch_size', type=int, default=16, help='batch_size root dir')
    parser.add_argument('--output', type=str, default='')
    parser.add_argument('--num_classes', type=int, default=14)
    parser.add_argument('--image_size', type=int, default=256, help='image_size')
    parser.add_argument('--channels', type=int, default=3, help='channels')

    args = parser.parse_args()

    device = "cuda"

    ch = args.channels

    dataloader = get_mask_data(args)

    model = UNet_mask(num_classes=args.num_classes).to(device)
    name1 = f"./models/{args.dataset}/ema_ckpt_latest.pt"
    name2 = f"./models/{args.dataset}/ema_ckpt_5035.pt"
    if os.path.exists(name2):
        modelname = name2
    else:
        modelname = name1
    ckpt = torch.load(modelname, weights_only=True)
    print('load', modelname)
    model.load_state_dict(ckpt, strict=False)

    diffusion = Diffusion(img_size=args.image_size, device=device)

    for k in range (0,args.num_classes):
        cls = k
        savePath = os.path.join(args.output, str(cls)+'_'+str(cls))
        if not os.path.exists(savePath):
            os.makedirs(savePath)

        print('generate class ', cls)

        pbar = tqdm(dataloader)
        for idx, (masks, labels, names) in enumerate(pbar):
            masks = masks.to(device)
        

        # for idx in range (1000, 1100):
            nums = args.batch_size
            y = torch.Tensor([cls] * nums).long().to(device)
            sampled_images = diffusion.sample(model, nums, masks).squeeze().to('cpu').numpy()
            sampled_masks = masks.squeeze().to('cpu').numpy()

            if (ch > 1): 
                sampled_images = np.transpose(sampled_images, [0, 2, 3, 1])
            for i in range (sampled_images.shape[0]):
                name = os.path.join(savePath, f"{idx}_{i}_img.png")
                name_m = os.path.join(savePath, f"{idx}_{i}_mask.png")
                if (ch > 1): 
                    img = cv2.cvtColor(sampled_images[i],cv2.COLOR_BGR2RGB)
                else:
                    img = cv2.cvtColor(sampled_images[i],cv2.COLOR_BGR2GRAY)
                msk = sampled_masks[i]*255*50
                cv2.imwrite(name, img)
                cv2.imwrite(name_m, msk)

if __name__ == '__main__':

    test()