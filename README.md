# HappyKoala 🐨

A unified segmentation benchmark integrating advanced models on ISIC 2018 dataset.

<img src="./imgs/koala.png" alt="KOALA" style="display: block; margin: auto;" width="400">

## Description

HappyKoala is an experimental framework designed for **comprehensive evaluation of medical image segmentation models** on the ISIC 2018 skin lesion segmentation dataset.  
This project integrates multiple **state-of-the-art architectures** including classical UNet variants, attention-enhanced models, and modern Transformer-based designs.

By providing a unified training and evaluation pipeline, HappyKoala facilitates **fair comparison, reproducibility**, and **rapid prototyping** of segmentation models.

## Integrated Models

- **UNet**  
- **VMUNet**  
- **U2Net**  
- **UNet++**  
- **UNet+++**  
- **VMUNetV2**  
- **HVMUNet**  
- **TransUNet**  
- **ResUNet**  
- **ResUNet++**

## Features

- 📚 Unified training and evaluation pipeline
- 🏥 Focused on medical image segmentation (skin lesion)
- 🧩 Modular architecture: easily add new models
- 📊 Standard metrics (IoU, DSC, Accuracy, Sensitivity, Specificity)
- 📈 Visualization of segmentation results

## Installation

```bash
# Clone the repository
git clone https://github.com/kancheng/happykoala.git

# Navigate to the project directory
cd happykoala
```

## Environments

```bash
conda create -n vmunet python=3.8
conda activate vmunet
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
pip install packaging
pip install timm==0.4.12
pip install pytest chardet yacs termcolor
pip install submitit tensorboardX
pip install triton==2.0.0
pip install causal_conv1d==1.0.0  # causal_conv1d-1.0.0+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install mamba_ssm==1.0.1  # mmamba_ssm-1.0.1+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install scikit-learn matplotlib thop h5py SimpleITK scikit-image medpy yacs
```
The .whl files of causal_conv1d and mamba_ssm could be found here. {[Baidu](https://pan.baidu.com/s/1Tibn8Xh4FMwj0ths8Ufazw?pwd=uu5k) or [GoogleDrive](https://drive.google.com/drive/folders/1tZGs1YFHiDrMa-MjYY8ZoEnCyy7m7Gaj?usp=sharing)}

## Dataset

- **ISIC 2018**  
  - ISIC 2018: Skin Lesion Analysis Towards Melanoma Detection
  - [https://challenge.isic-archive.com/data](https://challenge.isic-archive.com/data)

### ISIC datasets
- The ISIC17 and ISIC18 datasets, divided into a 7:3 ratio, can be found here {[Baidu](https://pan.baidu.com/s/1Y0YupaH21yDN5uldl7IcZA?pwd=dybm) or [GoogleDrive](https://drive.google.com/file/d/1XM10fmAXndVLtXWOt5G0puYSQyI2veWy/view?usp=sharing)}. 

- After downloading the datasets, you are supposed to put them into './external/isic2017/' and './external/isic2018/', and the file format reference is as follows. (take the ISIC17 dataset as an example.)

- './data/isic17/'
  - train
    - images
      - .png
    - masks
      - .png
  - val
    - images
      - .png
    - masks
      - .png
