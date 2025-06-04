import os
import shutil
import random

dataset_dir = r'C:\PYTHON PY-CHARM\pythonProject1\dlproject\images'
train_dir = r'C:\PYTHON PY-CHARM\pythonProject1\dlproject\training'
val_dir = r'C:\PYTHON PY-CHARM\pythonProject1\dlproject\validation'

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

for fruit_class in os.listdir(dataset_dir):
    fruit_class_path = os.path.join(dataset_dir, fruit_class)

    if os.path.isdir(fruit_class_path):
        os.makedirs(os.path.join(train_dir, fruit_class), exist_ok=True)
        os.makedirs(os.path.join(val_dir, fruit_class), exist_ok=True)

        image_files = [f for f in os.listdir(fruit_class_path) if os.path.isfile(os.path.join(fruit_class_path, f))]
        random.shuffle(image_files)
        split_index = int(0.8 * len(image_files))

        for i, image in enumerate(image_files):
            src_path = os.path.join(fruit_class_path, image)
            dst_path = os.path.join(train_dir if i < split_index else val_dir, fruit_class, image)
            shutil.copy(src_path, dst_path)

        print(f"Finished splitting {fruit_class} images into train and validation.")
