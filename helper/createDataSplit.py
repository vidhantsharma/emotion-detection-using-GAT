import os
import random
import shutil
from pathlib import Path

def create_split_dirs(emotion_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    # Create train, val, test directories inside each emotion folder
    for split in ['train', 'val', 'test']:
        split_dir = emotion_dir / split
        split_dir.mkdir(exist_ok=True)

def split_images(emotion_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    images = list(emotion_dir.glob("*.png"))  # Change extension as needed
    total_images = len(images)
    
    train_count = int(train_ratio * total_images)
    val_count = int(val_ratio * total_images)
    
    random.shuffle(images)
    
    # Split images
    train_images = images[:train_count]
    val_images = images[train_count:train_count + val_count]
    test_images = images[train_count + val_count:]
    
    # Move images to their respective directories
    for img in train_images:
        shutil.move(str(img), emotion_dir / 'train' / img.name)
    for img in val_images:
        shutil.move(str(img), emotion_dir / 'val' / img.name)
    for img in test_images:
        shutil.move(str(img), emotion_dir / 'test' / img.name)

def main(data_dir):
    data_dir = Path(data_dir)
    emotion_folders = [f for f in data_dir.iterdir() if f.is_dir()]
    
    for emotion_dir in emotion_folders:
        create_split_dirs(emotion_dir)
        split_images(emotion_dir)

# Run the main function with the path to your data folder
if __name__ == "__main__":
    main("data")
