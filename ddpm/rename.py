import os

train_dir = 'train/masks/'
test_dir = 'test/masks/'

def rename_files(directory, prefix):
    for count, filename in enumerate(os.listdir(directory), 1):
        if filename.endswith('.png'):
            new_name = f"{count}_{prefix}.png"
            old_file = os.path.join(directory, filename)
            new_file = os.path.join(directory, new_name)
            os.rename(old_file, new_file)
            print(f"Renamed: {old_file} -> {new_file}")

rename_files(train_dir, 'train')

rename_files(test_dir, 'test')