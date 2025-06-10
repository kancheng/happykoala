import os

def contains_images(directory, extensions=('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif', '.nii.gz')):
    outputs_path = os.path.join(directory, "outputs")
    if not os.path.exists(outputs_path):
        return False
    for file in os.listdir(outputs_path):
        if file.lower().endswith(extensions):
            return True
    return False

# 遍歷主資料夾
# root_dir = "/home/kan/proj/koala/results"  # ← 替換成你的路徑
root_dir = "./results"  # ← 替換成你的路徑
for subdir in os.listdir(root_dir):
    full_path = os.path.join(root_dir, subdir)
    if os.path.isdir(full_path):
        has_images = contains_images(full_path)
        print(f"{subdir} => {'✅ 有圖片' if has_images else '❌ 沒有圖片'}")
