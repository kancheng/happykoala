import os
import numpy as np
from PIL import Image  # 使用PIL讀取影像，如果使用OpenCV，請改為import cv2

def calculate_mean_std(dataset_path):
    # 用於累積所有像素值的總和和平方總和
    pixel_sum = 0
    pixel_squared_sum = 0
    total_pixels = 0

    # 遍歷資料夾中的所有圖像
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                # 使用PIL讀取圖像
                img_path = os.path.join(root, file)
                img = Image.open(img_path).convert('L')  # 轉換成灰階
                img_array = np.array(img, dtype=np.float32)

                # 更新累積的像素值
                pixel_sum += np.sum(img_array)
                pixel_squared_sum += np.sum(img_array ** 2)
                total_pixels += img_array.size

    # 計算mean和std
    mean = pixel_sum / total_pixels
    std = np.sqrt((pixel_squared_sum / total_pixels) - (mean ** 2))

    return mean, std

# 使用範例
# dataset_path = 'path/to/your/dataset'  # 替換成你的資料集路徑
# data_isic17, 18
dataset_path = './external/isic2018/train/images'

mean, std = calculate_mean_std(dataset_path)
print(f'Dataset Mean: {mean}')
print(f'Dataset Std: {std}')
