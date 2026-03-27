import os
import random
import shutil
from pathlib import Path

# Config
SOURCE_DIR = Path("data/kaggle")
TARGET_DIR = Path("data/dataset_small")
TARGET_COUNTS = {
    "food": 5000,
    "background": 500
}

def create_subset():
    print(f"Creating small dataset subset in {TARGET_DIR}...")
    
    if TARGET_DIR.exists():
        shutil.rmtree(TARGET_DIR)
        
    for split in ["train", "val"]:
        for category, total_target in TARGET_COUNTS.items():
            is_train = split == "train"
            target = int(total_target * 0.8) if is_train else int(total_target * 0.2)
            
            out = TARGET_DIR / split / category
            out.mkdir(parents=True, exist_ok=True)
            
            images = []
            if category == "food":
                src = SOURCE_DIR / "food"
                images = list(src.rglob("*.jpg")) + list(src.rglob("*.png")) + list(src.rglob("*.jpeg"))
            else:
                # Get background images from demo_subset where they currently live
                demo_bg = Path("data/demo_subset") / split / "background"
                if demo_bg.exists(): images.extend(list(demo_bg.rglob("*.jpg")) + list(demo_bg.rglob("*.png")) + list(demo_bg.rglob("*.jpeg")))
                
            if not images:
                print(f"No images found for {category}")
                continue
                
            # Sample images
            sample_size = min(len(images), target)
            sampled = random.sample(images, sample_size)
            
            print(f"Copying {sample_size} images into {split}/{category}...")
            for img in sampled:
                shutil.copy2(img, out / f"{img.parent.name}_{img.name}")
                
    print("\n✅ Subset creation complete!")

if __name__ == "__main__":
    create_subset()
