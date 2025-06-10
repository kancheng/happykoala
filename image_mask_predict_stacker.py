import os
import cv2
import numpy as np
from PIL import Image, ImageOps

def trim_white_border(img_np, threshold=250):
    gray = np.mean(img_np, axis=2)
    row_mean = np.mean(gray, axis=1)
    col_mean = np.mean(gray, axis=0)
    rows = np.where(row_mean < threshold)[0]
    cols = np.where(col_mean < threshold)[0]
    top, bottom = rows[0], rows[-1]
    left, right = cols[0], cols[-1]
    return img_np[top:bottom+1, left:right+1]

def find_vertical_segments(gray_img, threshold=250):
    v_proj = np.mean(gray_img, axis=1)
    gap_indices = np.where(v_proj > threshold)[0]
    from itertools import groupby
    from operator import itemgetter
    ranges = []
    for _, g in groupby(enumerate(gap_indices), lambda x: x[0]-x[1]):
        group = list(map(itemgetter(1), g))
        if len(group) > 10:
            ranges.append((group[0], group[-1]))
    segments = []
    start = 0
    H = gray_img.shape[0]
    for g0, g1 in ranges:
        if g0 > start:
            segments.append((start, g0))
        start = g1 + 1
    if start < H:
        segments.append((start, H))
    return segments

def add_white_border(pil_img, border=10):
    return ImageOps.expand(pil_img, border=border, fill='white')

# === 主程序 ===
input_dir = "./"   # 當前資料夾
output_dir = "./sub_output"
os.makedirs(output_dir, exist_ok=True)

input_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".png")])
stacked_images = []

for fname in input_files:
    try:
        print(f"▶ 處理中：{fname}")
        img_all = Image.open(os.path.join(input_dir, fname)).convert("RGB")
        img_np = np.array(img_all)
        img_trimmed = trim_white_border(img_np)
        gray = np.mean(img_trimmed, axis=2)
        segments = find_vertical_segments(gray)

        if len(segments) < 3:
            print(f"⚠️  {fname} 切割不足 3 段，略過。")
            continue

        parts = []
        for top, bot in segments:
            crop = img_trimmed[top:bot, :, :]
            parts.append(Image.fromarray(crop))

        img = np.array(parts[0].convert("RGB"))
        mask = np.array(parts[1].convert("L"))
        predict = np.array(parts[2].convert("L"))

        overlay_mask = img.copy()
        overlay_predict = img.copy()
        overlay_both = img.copy()

        # 畫藍線（mask）
        contours_mask, _ = cv2.findContours((mask > 128).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay_mask, contours_mask, -1, (0, 0, 255), 2)
        cv2.drawContours(overlay_both, contours_mask, -1, (0, 0, 255), 2)

        # 畫綠線（predict）
        contours_pred, _ = cv2.findContours((predict > 128).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay_predict, contours_pred, -1, (0, 255, 0), 2)
        cv2.drawContours(overlay_both, contours_pred, -1, (0, 255, 0), 2)

        images = [
            parts[0],
            parts[1],
            parts[2],
            Image.fromarray(overlay_mask),
            Image.fromarray(overlay_predict),
            Image.fromarray(overlay_both),
        ]

        images_with_border = [add_white_border(im, border=10) for im in images]
        total_height = sum(im.height for im in images_with_border)
        max_width = max(im.width for im in images_with_border)

        vertical_img = Image.new("RGB", (max_width, total_height), color="white")
        y = 0
        for im in images_with_border:
            vertical_img.paste(im, (0, y))
            y += im.height

        vertical_output_path = os.path.join(output_dir, f"stacked_{fname}")
        vertical_img.save(vertical_output_path)
        stacked_images.append(vertical_img)

    except Exception as e:
        print(f"❌ 錯誤處理 {fname}: {e}")

# === 合併所有直圖為橫圖 ===
if stacked_images:
    total_width = sum(im.width for im in stacked_images)
    max_height = max(im.height for im in stacked_images)
    final_img = Image.new("RGB", (total_width, max_height), color="white")
    x = 0
    for im in stacked_images:
        final_img.paste(im, (x, 0))
        x += im.width

    final_img.save(os.path.join(output_dir, "final_horizontal_result.png"))
    print("✅ 完成：已輸出 final_horizontal_result.png")
else:
    print("⚠️ 沒有任何圖片成功處理")
