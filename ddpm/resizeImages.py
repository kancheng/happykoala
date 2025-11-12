from PIL import Image
import os
import sys

from PIL import Image
import os

def resize_images(input_folder, output_folder, target_resolution=(64, 64)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for root, dirs, files in os.walk(input_folder):
        for file in files:
            input_path = os.path.join(root, file)
            relative_path = os.path.relpath(input_path, input_folder)
            output_path = os.path.join(output_folder, relative_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            resize_and_save(input_path, output_path, target_resolution)

def resize_and_save(input_path, output_path, target_resolution):
    try:
        with Image.open(input_path) as img:
            rgb_img = img.convert("RGB")
            resized_img = rgb_img.resize(target_resolution, Image.ANTIALIAS)
            resized_img.save(output_path, format="JPEG")
            print(f"Resized and saved: {output_path}")
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")

input_folder = sys.argv[1]
output_folder = sys.argv[2]
resize_images(input_folder, output_folder)
