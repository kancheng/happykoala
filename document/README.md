# FedKoala(KoalaMamba)
This project is used for federated learning and image segmentation.



# VM-UNet
This is the official code repository for "VM-UNet: Vision Mamba UNet for Medical
Image Segmentation". {[Arxiv Paper](https://arxiv.org/abs/2402.02491)}

## Abstract
In the realm of medical image segmentation, both CNN-based and Transformer-based models have been extensively explored. However, CNNs exhibit limitations in long-range modeling capabilities, whereas Transformers are hampered by their quadratic computational complexity. Recently, State Space Models (SSMs), exemplified by Mamba, have emerged as a promising approach. They not only excel in modeling long-range interactions but also maintain a linear computational complexity. In this paper, leveraging state space models, we propose a U-shape architecture model for medical image segmentation, named Vision Mamba UNet (VM-UNet). Specifically, the Visual State Space (VSS) block is introduced as the foundation block to capture extensive contextual information, and an asymmetrical encoder-decoder structure is constructed. We conduct comprehensive experiments on the ISIC17, ISIC18, and Synapse datasets, and the results indicate that VM-UNet performs competitively in medical image segmentation tasks. To our best knowledge, this is the first medical image segmentation model constructed based on the pure SSM-based model. We aim to establish a baseline and provide valuable insights for the future development of more efficient and effective SSM-based segmentation systems.

## 0. Main Environments
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
The .whl files of causal_conv1d and mamba_ssm could be found here. {[Baidu](https://pan.baidu.com/s/1Tibn8Xh4FMwj0ths8Ufazw?pwd=uu5k)}

## 1. Prepare the dataset

### ISIC datasets
- The ISIC17 and ISIC18 datasets, divided into a 7:3 ratio, can be found here {[Baidu](https://pan.baidu.com/s/1Y0YupaH21yDN5uldl7IcZA?pwd=dybm) or [GoogleDrive](https://drive.google.com/file/d/1XM10fmAXndVLtXWOt5G0puYSQyI2veWy/view?usp=sharing)}. 

- After downloading the datasets, you are supposed to put them into './data/isic17/' and './data/isic18/', and the file format reference is as follows. (take the ISIC17 dataset as an example.)

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

### Synapse datasets

- For the Synapse dataset, you could follow [Swin-UNet](https://github.com/HuCaoFighting/Swin-Unet) to download the dataset, or you could download them from {[Baidu](https://pan.baidu.com/s/1JCXBfRL9y1cjfJUKtbEhiQ?pwd=9jti)}.

- After downloading the datasets, you are supposed to put them into './data/Synapse/', and the file format reference is as follows.

- './data/Synapse/'
  - lists
    - list_Synapse
      - all.lst
      - test_vol.txt
      - train.txt
  - test_vol_h5
    - casexxxx.npy.h5
  - train_npz
    - casexxxx_slicexxx.npz

## 2. Prepare the pre_trained weights

- The weights of the pre-trained VMamba could be downloaded [here](https://github.com/MzeroMiko/VMamba) or [Baidu](https://pan.baidu.com/s/1ci_YvPPEiUT2bIIK5x8Igw?pwd=wnyy). After that, the pre-trained weights should be stored in './pretrained_weights/'.



## 3. Train the VM-UNet
```bash
cd VM-UNet
python train.py  # Train and test VM-UNet on the ISIC17 or ISIC18 dataset.
python train_synapse.py  # Train and test VM-UNet on the Synapse dataset.
```

## 4. Obtain the outputs
- After trianing, you could obtain the results in './results/'

## 5. Acknowledgments

- We thank the authors of [VMamba](https://github.com/MzeroMiko/VMamba) and [Swin-UNet](https://github.com/HuCaoFighting/Swin-Unet) for their open-source codes.



# Federated Learning [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4321561.svg)](https://doi.org/10.5281/zenodo.4321561)

## 0. Reference

1. Communication-Efficient Learning of Deep Networks from Decentralized Data

This is partly the reproduction of the paper of [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629)   
Only experiments on MNIST and CIFAR10 (both IID and non-IID) is produced by far.

Note: The scripts will be slow without the implementation of parallel computing. 

2. Federated Learning on Non-IID Data with Local-drift Decoupling and Correction
Code for paper - **[Federated Learning on Non-IID Data with Local-drift Decoupling and Correction]**

We provide code to run FedDC, FedAvg, 
[FedDyn](https://openreview.net/pdf?id=B7v4QMR6Z9w), 
[Scaffold](https://openreview.net/pdf?id=B7v4QMR6Z9w), and [FedProx](https://arxiv.org/abs/1812.06127) methods.

3. FedUKD: Federated UNet Model with Knowledge Distillation for Land Use Classification from Satellite and Street Views

- https://arxiv.org/abs/2212.02196


## 1. Run

The MLP and CNN models are produced by:
> python [main_nn.py](main_nn.py)

Federated learning with MLP and CNN is produced by:
> python [main_fed.py](main_fed.py)

See the arguments in [options.py](utils/options.py). 

For example:

```
python main_fed.py --dataset mnist --iid --num_channels 1 --model cnn --epochs 50 --gpu 0  
```

`--all_clients` for averaging over all client models

NB: for CIFAR-10, `num_channels` must be 3.

## 2. Results

### MNIST
Results are shown in Table 1 and Table 2, with the parameters C=0.1, B=10, E=5.

Table 1. results of 10 epochs training with the learning rate of 0.01

| Model     | Acc. of IID | Acc. of Non-IID|
| -----     | -----       | ----           |
| FedAVG-MLP|  94.57%     | 70.44%         |
| FedAVG-CNN|  96.59%     | 77.72%         |

Table 2. results of 50 epochs training with the learning rate of 0.01

| Model     | Acc. of IID | Acc. of Non-IID|
| -----     | -----       | ----           |
| FedAVG-MLP| 97.21%      | 93.03%         |
| FedAVG-CNN| 98.60%      | 93.81%         |


## 3. Ackonwledgements
Acknowledgements give to [youkaichao](https://github.com/youkaichao).

## 4. Cite As
Shaoxiong Ji. (2018, March 30). A PyTorch Implementation of Federated Learning. Zenodo. http://doi.org/10.5281/zenodo.4321561

## 5. CMD

```
# MNIST CNN MLP
python3 main_nn.py --dataset mnist --iid --num_channels 1 --model cnn --epochs 10 --gpu 0 

python3 main_nn.py --dataset mnist --iid --num_channels 1 --model cnn --epochs 10 --gpu 0 --all_clients

python3 main_nn.py --dataset mnist --iid --num_channels 1 --model mlp --epochs 10 --gpu 0 

python3 main_nn.py --dataset mnist --iid --num_channels 1 --model mlp --epochs 10 --gpu 0 --all_clients

# MNIST NN
python3 main_nn.py --dataset mnist --iid --num_channels 1 --model 2nn --epochs 10 --gpu 0  

python3 main_nn.py --dataset mnist --iid --num_channels 1 --model 2nn --epochs 10 --gpu 0 --all_clients

# CIFAR10 CNN MLP
python3 main_nn.py --dataset cifar --iid --num_channels 3 --model cnn --epochs 10 --gpu 0

python3 main_nn.py --dataset cifar --iid --num_channels 3 --model cnn --epochs 10 --gpu 0 --all_clients

python3 main_nn.py --dataset cifar --iid --num_channels 3 --model mlp --epochs 10 --gpu 0

python3 main_nn.py --dataset cifar --iid --num_channels 3 --model mlp --epochs 10 --gpu 0 --all_clients

# CIFAR100 CNN MLP
python3 main_nn.py --dataset cifar100 --iid --num_channels 3 --model cnn --epochs 10 --gpu 0 

python3 main_nn.py --dataset cifar100 --iid --num_channels 3 --model cnn --epochs 10 --gpu 0 --all_clients

python3 main_nn.py --dataset cifar100 --iid --num_channels 3 --model mlp --epochs 10 --gpu 0

python3 main_nn.py --dataset cifar100 --iid --num_channels 3 --model mlp --epochs 10 --gpu 0 --all_clients

# EMNIST NN
python3 main_nn.py --dataset emnist --iid --num_channels 1 --model nn --epochs 10 --gpu 0

python3 main_nn.py --dataset emnist --iid --num_channels 1 --model nn --epochs 10 --gpu 0 --all_clients

# SALT UNET
python3 main_nn.py --dataset salt --iid --num_channels 1 --model unet --epochs 10 --gpu 0

python3 main_nn.py --dataset salt --iid --num_channels 1 --model unet --epochs 10 --gpu 0 --all_clients

```

## 6. Fed Test.

```
# MNIST CNN MLP FEDAVG
python3 main_fed.py --dataset mnist --iid --num_channels 1 --model cnn --epochs 10 --gpu 0 

python3 main_fed.py --dataset mnist --iid --num_channels 1 --model cnn --epochs 10 --gpu 0 --all_clients

python3 main_fed.py --dataset mnist --iid --num_channels 1 --model mlp --epochs 10 --gpu 0 

python3 main_fed.py --dataset mnist --iid --num_channels 1 --model mlp --epochs 10 --gpu 0 --all_clients

# MNIST NN FEDAVG
python3 main_fed.py --dataset mnist --iid --num_channels 1 --model 2nn --epochs 10 --gpu 0  

python3 main_fed.py --dataset mnist --iid --num_channels 1 --model 2nn --epochs 10 --gpu 0 --all_clients


# CIFAR10 CNN MLP FEDAVG
python3 main_fed.py --dataset cifar --iid --num_channels 3 --model cnn --epochs 10 --gpu 0

python3 main_fed.py --dataset cifar --iid --num_channels 3 --model cnn --epochs 10 --gpu 0 --all_clients

python3 main_fed.py --dataset cifar --iid --num_channels 3 --model mlp --epochs 10 --gpu 0

python3 main_fed.py --dataset cifar --iid --num_channels 3 --model mlp --epochs 10 --gpu 0 --all_clients


# CIFAR100 CNN MLP FEDAVG
python3 main_fed.py --dataset cifar100 --iid --num_channels 3 --model cnn --epochs 10 --gpu 0 --num_classes 100

python3 main_fed.py --dataset cifar100 --iid --num_channels 3 --model cnn --epochs 10 --gpu 0 --num_classes 100 --all_clients

python3 main_fed.py --dataset cifar100 --iid --num_channels 3 --model mlp --epochs 10 --gpu 0 --num_classes 100

python3 main_fed.py --dataset cifar100 --iid --num_channels 3 --model mlp --epochs 10 --gpu 0 --num_classes 100 --all_clients


# EMNIST NN FEDAVG
python3 main_fed.py --dataset emnist --iid --num_channels 1 --model nn --epochs 10 --gpu 0

python3 main_fed.py --dataset emnist --iid --num_channels 1 --model nn --epochs 10 --gpu 0 --all_clients


# SALT UNET FEDAVG
python3 main_fed.py --dataset salt --iid --num_channels 1 --model unet --epochs 10 --gpu 0

python3 main_fed.py --dataset salt --iid --num_channels 1 --model unet --epochs 10 --gpu 0 --all_clients


```
