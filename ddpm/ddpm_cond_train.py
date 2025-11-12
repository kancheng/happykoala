
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
import copy
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from utils import *
from modules import UNet_mask, EMA
import logging

import my_lr_schedul
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

    def sample(self, model, n, masks, labels, cfg_scale=3):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, masks, t)    #, labels
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

def train(args):
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)
    model = UNet_mask().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, my_lr_schedul.lr_lambda)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)
    ema = EMA(0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    
    dlen = len(dataloader)
    epochs = int(6.8e6 * args.num_classes / (dlen*args.batch_size))
    print('dlen, epochs:', dlen, epochs)
    glb_masks = []
    for epoch in range(epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (images, masks, labels) in enumerate(pbar):

            images = images.to(device) * 2 - 1
            masks = masks.to(device)
            labels = labels.to(device)

            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            if np.random.random() < 0.1:
                labels = None
            predicted_noise = model(x_t, masks, t)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_scheduler.step()
            ema.step_ema(ema_model, model)

            pbar.set_postfix(MSE=loss.item())

            glb_masks = masks[:args.num_classes]

        torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt_latest.pt"))
        torch.save(ema_model.state_dict(), os.path.join("models", args.run_name, f"ema_ckpt_latest.pt"))
        torch.save(optimizer.state_dict(), os.path.join("models", args.run_name, f"optim_latest.pt"))
        if epoch % 5 == 0:

            labels = torch.arange(args.num_classes).long().to(device)
            ema_sampled_images = diffusion.sample(ema_model, n=len(labels), masks=glb_masks, labels=labels)
            save_images(glb_masks, os.path.join("results", args.run_name, f"{epoch}_mask.png"))
            save_images(ema_sampled_images, os.path.join("results", args.run_name, f"{epoch}_ema.png"))
            torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt_{epoch}.pt"))
            torch.save(ema_model.state_dict(), os.path.join("models", args.run_name, f"ema_ckpt_{epoch}.pt"))
            torch.save(optimizer.state_dict(), os.path.join("models", args.run_name, f"optim_{epoch}.pt"))

def launch():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('--image_path', type=str, default='./', help='dataset root dir')
    parser.add_argument('--mask_path', type=str, default='./', help='dataset root dir')
    parser.add_argument('--batch_size', type=int, default=16, help='batch_size root dir')
    parser.add_argument('--num_classes', type=int, default=10, help='number of labeles')
    parser.add_argument('--image_size', type=int, default=64, help='image_size')
    parser.add_argument('--channels', type=int, default=3, help='channels')

    args = parser.parse_args()
    args.run_name = args.dataset
    args.epochs = 2000

    args.device = "cuda"
    args.lr = 3e-4
    train(args)

if __name__ == '__main__':
    launch()

