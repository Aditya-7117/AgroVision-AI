import os
import shutil
import random

SOURCE_DIR = r"C:\Users\adity\Downloads\PlantVillage Dataset (Labeled)\color"
TARGET_DIR = r"C:\Users\adity\OneDrive\Desktop\GIT-PROJECTS\AgroVision-AI\dataset"

TRAIN_RATIO = 0.8
SEED = 42

random.seed(SEED)

os.makedirs(os.path.join(TARGET_DIR, "train"), exist_ok=True)
os.makedirs(os.path.join(TARGET_DIR, "val"), exist_ok=True)

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")

for class_name in os.listdir(SOURCE_DIR):
    class_path = os.path.join(SOURCE_DIR, class_name)
    if not os.path.isdir(class_path):
        continue

    images = [
        f for f in os.listdir(class_path)
        if f.lower().endswith(VALID_EXTENSIONS)
    ]

    if len(images) == 0:
        print(f"Skipping {class_name} (no images found)")
        continue

    random.shuffle(images)

    split_idx = int(len(images) * TRAIN_RATIO)
    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]

    train_class_dir = os.path.join(TARGET_DIR, "train", class_name)
    val_class_dir = os.path.join(TARGET_DIR, "val", class_name)

    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(val_class_dir, exist_ok=True)

    for img in train_imgs:
        shutil.copy2(
            os.path.join(class_path, img),
            os.path.join(train_class_dir, img)
        )

    for img in val_imgs:
        shutil.copy2(
            os.path.join(class_path, img),
            os.path.join(val_class_dir, img)
        )

    print(f"{class_name}: {len(train_imgs)} train | {len(val_imgs)} val")

print("Dataset split completed successfully.")