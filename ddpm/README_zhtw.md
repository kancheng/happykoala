# OTTER: Optimized Training with Trustworthy Enhanced Replication via Diffusion and Federated VMUNet for Privacy-Aware Medical Segmentation

[![Paper](https://img.shields.io/badge/Paper-ICICS%202025-blue)](https://doi.org/10.1007/978-981-95-3543-9_18)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-orange.svg)](https://pytorch.org/)

這是 **OTTER** 項目的官方實現，該項目提出了一種基於擴散模型（Diffusion Model）和聯邦 VMUNet 的隱私感知醫學圖像分割方法。該論文已被 **ICICS 2025** 接受。

## 📋 目錄

- [簡介](#簡介)
- [論文信息](#論文信息)
- [功能特點](#功能特點)
- [環境要求](#環境要求)
- [安裝](#安裝)
- [數據準備](#數據準備)
- [使用方法](#使用方法)
- [項目結構](#項目結構)
- [實驗結果](#實驗結果)
- [引用](#引用)
- [許可證](#許可證)

## 簡介

OTTER 是一個用於醫學圖像分割的條件擴散模型框架，結合了：

- **擴散模型 (DDPM)**: 使用去噪擴散概率模型進行高質量圖像生成
- **VMUNet 架構**: 基於 U-Net 的變體，支持 mask 條件生成
- **隱私保護**: 通過聯邦學習框架保護醫療數據隱私
- **條件生成**: 支持基於 mask 和 label 的條件圖像生成

## 論文信息

**標題**: OTTER: Optimized Training with Trustworthy Enhanced Replication via Diffusion and Federated VMUNet for Privacy-Aware Medical Segmentation

**作者**: Haocheng Kan, Yuesheng Zhu, Guibo Luo, Hanwen Zhang

**會議**: Information and Communications Security : 27th International Conference, ICICS 2025, Nanjing, China, October 29–31, 2025

**DOI**: [10.1007/978-981-95-3543-9_18](https://doi.org/10.1007/978-981-95-3543-9_18)

**頁碼**: 331-346

## 功能特點

- ✅ 條件擴散模型訓練和推理
- ✅ 支持多類別醫學圖像分割
- ✅ EMA (Exponential Moving Average) 模型穩定訓練
- ✅ 自定義學習率調度器
- ✅ FID 分數計算評估
- ✅ 支持 ISIC2018、SDSaliency900 等數據集
- ✅ 可配置的訓練參數

## 環境要求

- Python 3.8+
- PyTorch 1.8+
- CUDA (推薦，用於 GPU 加速)
- 其他依賴見安裝部分

## 數據準備

數據集應按以下結構組織：

```
dataset_name/
├── images/
│   └── 0/          # 類別 0 的圖像
│       ├── img1.jpg
│       └── img2.jpg
└── masks/
    └── 0/            # 對應的 mask
        ├── img1.png
        └── img2.png
```

支持的數據集格式：
- **ISIC2018**: 皮膚病變分割數據集
- **SDSaliency900**: 顯著性檢測數據集
- 其他自定義醫學圖像分割數據集

## 使用方法

### 訓練模型

#### 基本訓練

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

#### 使用優化版本訓練

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

### 測試/生成圖像

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

### 計算 FID 分數

```bash
python calc_fid.py /path/to/real_images /path/to/generated_images
```

### 參數說明

- `--dataset`: 數據集名稱（用於保存模型和結果）
- `--image_path`: 圖像數據路徑
- `--mask_path`: Mask 數據路徑
- `--num_classes`: 類別數量
- `--batch_size`: 批次大小
- `--image_size`: 圖像尺寸（推薦 256）
- `--channels`: 圖像通道數（RGB=3, Grayscale=1）
- `--output`: 測試時輸出目錄

## 項目結構

```
ddpm/
├── ddpm_cond_train.py          # 條件擴散模型訓練主文件
├── ddpm_cond_train_opt.py      # 優化版訓練腳本
├── ddpm_cond_test.py           # 測試/生成腳本
├── ddpm.py                     # 基礎 DDPM 實現
├── modules.py                  # 網絡模塊定義（UNet_mask, EMA, SelfAttention）
├── utils.py                    # 工具函數（數據加載、圖像保存）
├── utils2.py                   # 額外工具函數
├── my_lr_schedul.py            # 自定義學習率調度器
├── calc_fid.py                 # FID 分數計算
├── calc_fid2.py                # FID 計算備用版本
├── resizeImages.py             # 圖像尺寸調整工具
├── noising_test.py             # 噪聲測試腳本
├── models/                     # 模型保存目錄
│   └── {dataset_name}/
│       ├── ckpt_latest.pt      # 最新檢查點
│       ├── ema_ckpt_latest.pt  # EMA 模型檢查點
│       └── optim_latest.pt     # 優化器狀態
└── results/                    # 結果保存目錄
    └── {dataset_name}/
        ├── {epoch}_ema.png     # 生成的圖像
        └── {epoch}_mask.png    # 對應的 mask
```

## 核心組件

### UNet_mask

基於 U-Net 的條件生成網絡，支持：
- Mask 條件輸入
- 時間步嵌入
- 自注意力機制
- 多尺度特徵提取

### Diffusion 類

實現了完整的 DDPM 流程：
- 前向擴散過程（加噪）
- 反向去噪過程（採樣）
- 條件引導生成（CFG）

### EMA (Exponential Moving Average)

用於穩定模型訓練，提高生成質量。

## 實驗結果

模型在以下數據集上進行了實驗：

- **ISIC2018**: 皮膚病變分割
- **SDSaliency900**: 顯著性檢測

訓練過程中會自動保存：
- 每 5 個 epoch 的模型檢查點
- 生成的樣本圖像
- EMA 模型權重

## 訓練技巧

1. **批次大小**: 根據 GPU 內存調整，推薦 2-4
2. **學習率**: 默認 3e-4，可根據需要調整
3. **圖像尺寸**: 推薦 256x256，平衡質量和速度
4. **EMA 係數**: 默認 0.995，較高值提供更穩定的模型
5. **CFG Scale**: 測試時使用 3.0，可調整以平衡多樣性和質量

## 引用

如果您在研究中使用了本項目，請引用我們的論文：

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
