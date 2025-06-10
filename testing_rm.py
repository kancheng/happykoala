import os
import shutil

def contains_images(directory, extensions=('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif', '.nii.gz')):
    outputs_path = os.path.join(directory, "outputs")
    if not os.path.exists(outputs_path):
        return False
    for file in os.listdir(outputs_path):
        if file.lower().endswith(extensions):
            return True
    return False

# 主目錄（替換成你的根資料夾路徑）
# root_dir = "/home/kan/proj/koala/results"
root_dir = "./results"
# 掃描子資料夾
for subdir in os.listdir(root_dir):
    full_path = os.path.join(root_dir, subdir)
    if os.path.isdir(full_path):
        if not contains_images(full_path):
            print(f"❌ 沒有圖片，刪除資料夾：{full_path}")
            shutil.rmtree(full_path)
        else:
            print(f"✅ 保留，有圖片：{full_path}")

