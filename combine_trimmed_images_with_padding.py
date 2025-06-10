from PIL import Image
import numpy as np
import os
import glob

# 去除白邊函數（精確判斷）
def precise_trim_white_borders(img, threshold=245, padding=10):
    gray = img.convert("L")
    np_gray = np.array(gray)
    mask = np_gray < threshold
    coords = np.argwhere(mask)
    if coords.size == 0:
        return img
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    # 加上 padding，同時避免超出圖像邊界
    x0 = max(x0 - padding, 0)
    y0 = max(y0 - padding, 0)
    x1 = min(x1 + padding, img.width)
    y1 = min(y1 + padding, img.height)

    return img.crop((x0, y0, x1, y1))

# 設定要處理的目錄路徑（可更改為任意路徑）
image_dir = "."
image_files = sorted(glob.glob(os.path.join(image_dir, "*.png")))

# 使用更精確的方式去除所有圖片的白邊
all_precise_trimmed_images = [precise_trim_white_borders(Image.open(path)) for path in image_files]

# 計算合併尺寸
total_width_all = sum(img.width for img in all_precise_trimmed_images)
max_height_all = max(img.height for img in all_precise_trimmed_images)

# 建立並排圖像
combined_all_horizontal = Image.new("RGB", (total_width_all, max_height_all))

x_offset = 0
for img in all_precise_trimmed_images:
    combined_all_horizontal.paste(img, (x_offset, 0))
    x_offset += img.width

# 儲存最終圖像
combined_all_output_path = "combined_all_precise_trimmed_horizontal.png"
combined_all_horizontal.save(combined_all_output_path)

