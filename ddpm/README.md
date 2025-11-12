# OTTER: Optimized Training with Trustworthy Enhanced Replication via Diffusion and Federated VMUNet for Privacy-Aware Medical Segmentation

[中文版本 (繁體中文) / Chinese Traditional Version](README_zhtw.md)

[![Paper](https://img.shields.io/badge/Paper-ICICS%202025-blue)](https://doi.org/10.1007/978-981-95-3543-9_18)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-orange.svg)](https://pytorch.org/)

This is the official implementation of the **OTTER** project, which proposes a privacy-aware medical image segmentation method based on Diffusion Models and Federated VMUNet. This paper has been accepted by **ICICS 2025**.

## 📋 Table of Contents

- [Introduction](#introduction)
- [Paper Information](#paper-information)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Experimental Results](#experimental-results)
- [Citation](#citation)
- [License](#license)

## Introduction

OTTER is a conditional diffusion model framework for medical image segmentation that combines:

- **Diffusion Models (DDPM)**: High-quality image generation using denoising diffusion probabilistic models
- **VMUNet Architecture**: A U-Net based variant that supports mask-conditioned generation
- **Privacy Protection**: Protects medical data privacy through federated learning framework
- **Conditional Generation**: Supports mask and label-based conditional image generation

## Paper Information

**Title**: OTTER: Optimized Training with Trustworthy Enhanced Replication via Diffusion and Federated VMUNet for Privacy-Aware Medical Segmentation

**Authors**: Haocheng Kan, Yuesheng Zhu, Guibo Luo, Hanwen Zhang

**Conference**: Information and Communications Security : 27th International Conference, ICICS 2025, Nanjing, China, October 29–31, 2025

**DOI**: [10.1007/978-981-95-3543-9_18](https://doi.org/10.1007/978-981-95-3543-9_18)

**Pages**: 331-346

## Features

- ✅ Conditional diffusion model training and inference
- ✅ Support for multi-class medical image segmentation
- ✅ EMA (Exponential Moving Average) for stable training
- ✅ Custom learning rate scheduler
- ✅ FID score calculation for evaluation
- ✅ Support for ISIC2018, SDSaliency900 and other datasets
- ✅ Configurable training parameters

## Requirements

- Python 3.8+
- PyTorch 1.8+
- CUDA (recommended for GPU acceleration)
- Other dependencies see Installation section

## Installation

Please refer to the Installation section for detailed dependency information.

## Data Preparation

Datasets should be organized in the following structure:

```
dataset_name/
├── images/
│   └── 0/          # Images for class 0
│       ├── img1.jpg
│       └── img2.jpg
└── masks/
    └── 0/            # Corresponding masks
        ├── img1.png
        └── img2.png
```

Supported dataset formats:
- **ISIC2018**: Skin lesion segmentation dataset
- **SDSaliency900**: Saliency detection dataset
- Other custom medical image segmentation datasets

## Usage

### Training Models

#### Basic Training

```bash
CUDA_VISIBLE_DEVICES=0 python ddpm_cond_train.py \
    --dataset kisic5000_256 \
    --image_path /path/to/images \
    --mask_path /path/to/masks \
    --num_classes 2 \
    --batch_size 2 \
    --channels 3 \
    --image_size 256
```

#### Training with Optimized Version

```bash
CUDA_VISIBLE_DEVICES=0 python ddpm_cond_train_opt.py \
    --dataset sdsaliency900_256 \
    --image_path /path/to/images \
    --mask_path /path/to/masks \
    --num_classes 2 \
    --batch_size 2 \
    --channels 3 \
    --image_size 256
```

### Testing/Image Generation

```bash
CUDA_VISIBLE_DEVICES=0 python ddpm_cond_test.py \
    --dataset your_dataset \
    --num_classes 2 \
    --image_size 256 \
    --channels 3 \
    --output /path/to/output/ \
    --image_path /path/to/images/ \
    --mask_path /path/to/masks/ \
    --batch_size 4
```

### Calculating FID Score

```bash
python calc_fid.py /path/to/real_images /path/to/generated_images
```

### Parameter Description

- `--dataset`: Dataset name (used for saving models and results)
- `--image_path`: Path to image data
- `--mask_path`: Path to mask data
- `--num_classes`: Number of classes
- `--batch_size`: Batch size
- `--image_size`: Image size (recommended 256)
- `--channels`: Number of image channels (RGB=3, Grayscale=1)
- `--output`: Output directory for testing

## Project Structure

```
ddpm/
├── ddpm_cond_train.py          # Main conditional diffusion model training file
├── ddpm_cond_train_opt.py      # Optimized training script
├── ddpm_cond_test.py           # Testing/generation script
├── ddpm.py                     # Basic DDPM implementation
├── modules.py                  # Network module definitions (UNet_mask, EMA, SelfAttention)
├── utils.py                    # Utility functions (data loading, image saving)
├── utils2.py                   # Additional utility functions
├── my_lr_schedul.py            # Custom learning rate scheduler
├── calc_fid.py                 # FID score calculation
├── calc_fid2.py                # Alternative FID calculation version
├── resizeImages.py             # Image resizing utility
├── noising_test.py             # Noise testing script
├── models/                     # Model save directory
│   └── {dataset_name}/
│       ├── ckpt_latest.pt      # Latest checkpoint
│       ├── ema_ckpt_latest.pt  # EMA model checkpoint
│       └── optim_latest.pt     # Optimizer state
└── results/                    # Results save directory
    └── {dataset_name}/
        ├── {epoch}_ema.png     # Generated images
        └── {epoch}_mask.png    # Corresponding masks
```

## Core Components

### UNet_mask

U-Net based conditional generation network that supports:
- Mask conditional input
- Time step embedding
- Self-attention mechanism
- Multi-scale feature extraction

### Diffusion Class

Implements the complete DDPM pipeline:
- Forward diffusion process (noising)
- Reverse denoising process (sampling)
- Conditional guidance generation (CFG)

### EMA (Exponential Moving Average)

Used to stabilize model training and improve generation quality.

## Experimental Results

The model has been tested on the following datasets:

- **ISIC2018**: Skin lesion segmentation
- **SDSaliency900**: Saliency detection

The following are automatically saved during training:
- Model checkpoints every 5 epochs
- Generated sample images
- EMA model weights

## Training Tips

1. **Batch Size**: Adjust according to GPU memory, recommended 2-4
2. **Learning Rate**: Default 3e-4, can be adjusted as needed
3. **Image Size**: Recommended 256x256, balancing quality and speed
4. **EMA Coefficient**: Default 0.995, higher values provide more stable models
5. **CFG Scale**: Use 3.0 during testing, can be adjusted to balance diversity and quality

## Citation

If you use this project in your research, please cite our paper:

```bibtex
@inproceedings{kan2025otter,
  title={OTTER: Optimized Training with Trustworthy Enhanced Replication via Diffusion and Federated VMUNet for Privacy-Aware Medical Segmentation},
  author={Kan, Haocheng and Zhu, Yuesheng and Luo, Guibo and Zhang, Hanwen},
  booktitle={Information and Communications Security: 27th International Conference, ICICS 2025, Nanjing, China, October 29--31, 2025, Proceedings, Part II},
  pages={331--346},
  year={2025},
  organization={Springer},
  doi={10.1007/978-981-95-3543-9_18}
}
```
