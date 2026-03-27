"""
India-Focused Humanitarian Dataset Scraper & Kaggle Downloader

Builds a comprehensive training dataset for MobileNetV2 fine-tuning with
India-specific humanitarian donation images organized into 3 categories:
  - Food (rice, dal, atta, ration kits, canned food, grain sacks)
  - Clothing (sarees, blankets, textiles, winter wear donations)
  - Medicine (medical supplies, first aid, pharma packs)

Data Sources:
  1. Wikimedia Commons API (free, no auth needed)
  2. Kaggle Datasets (requires kaggle.json credentials)
  3. Open Images Dataset (Google, subset download)

Images are organized into PyTorch ImageFolder structure:
  data/dataset/
    ├── train/
    │   ├── food/
    │   ├── clothing/
    │   └── medicine/
    └── val/
        ├── food/
        ├── clothing/
        └── medicine/
"""

import os
import urllib.request
import urllib.parse
import json
import time
import ssl
import random
import shutil
import hashlib
from pathlib import Path

ssl._create_default_https_context = ssl._create_unverified_context

# ─── Configuration ────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "data", "dataset")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
TRAIN_SPLIT = 0.8  # 80% train, 20% val

# India-focused search queries per category
SEARCH_QUERIES = {
    "food": [
        # Indian staples
        "rice sack India",
        "rice bag donation",
        "dal lentils package",
        "atta flour bag India",
        "wheat flour sack",
        "Indian ration kit",
        "ration distribution India",
        "food packet distribution India",
        "relief food supply India",
        "grocery donation India",
        "canned food donation",
        "food grain storage bags",
        "rice distribution India",
        "midday meal India",
        "food relief package",
        "humanitarian food aid",
        "grain sacks warehouse",
        "PDS ration shop India",
        "food bank donation India",
        "dry ration kit distribution",
        # General food aid
        "relief supplies food",
        "emergency food package",
        "food donation box",
        "rice bags warehouse",
        "cooking oil donation",
        "sugar bags supply",
        "food aid truck",
        "disaster relief food India",
        "flood relief food packets",
        "NGO food distribution",
    ],
    "background": [
        # Generic non-food / noise backgrounds (to act as negative examples)
        "warehouse interior empty",
        "shipping boxes",
        "wooden pallets",
        "truck interior empty",
        "delivery truck",
        "grocery store aisle background",
        "storage facility",
        "logistics center",
        "street scene India",
        "office interior",
        "people standing background",
        "empty road India",
        "building exterior India",
        "cardboard boxes stack",
        "shipping container",
        "empty room",
        "packing materials",
        "crowd of people India"
    ],
}

# ─── Wikimedia Commons Scraper ────────────────────────────────────────────────


def fetch_wikimedia_images(query, count, output_dir, category, start_idx):
    """
    Fetch images from Wikimedia Commons API for a given search query.

    Args:
        query: Search string
        count: Max images to fetch per query
        output_dir: Directory to save images
        category: Category label (food/clothing/medicine)
        start_idx: Starting index for filename numbering

    Returns:
        Number of images successfully downloaded
    """
    endpoint = "https://commons.wikimedia.org/w/api.php"
    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": query,
        "gsrnamespace": 6,  # File namespace
        "gsrlimit": int(count),
        "prop": "imageinfo",
        "iiprop": "url|size",
        "format": "json"
    }

    query_string = urllib.parse.urlencode(params)
    url = f"{endpoint}?{query_string}"

    req = urllib.request.Request(url, headers={
        'User-Agent': 'HumanitarianLogisticsDatasetBot/2.0 (Research; India-focused)'
    })

    downloaded = 0
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            data = json.loads(response.read().decode())

        pages = data.get("query", {}).get("pages", {})
        for page_id, page_data in pages.items():
            imageinfo = page_data.get("imageinfo", [])
            if not imageinfo:
                continue

            img_url = imageinfo[0].get("url")
            img_size = imageinfo[0].get("size", 0)

            if not img_url:
                continue

            # Filter: only JPG/PNG, minimum 10KB (skip tiny thumbnails)
            if not img_url.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            if img_size < 10240:  # Skip images < 10KB
                continue

            ext = img_url.split('.')[-1].lower()
            # Use hash of URL for unique filename
            url_hash = hashlib.md5(img_url.encode()).hexdigest()[:8]  # type: ignore[index]
            filename = os.path.join(
                output_dir,
                f"{category}_{start_idx + downloaded:04d}_{url_hash}.{ext}"
            )

            try:
                img_req = urllib.request.Request(img_url, headers={
                    'User-Agent': 'Mozilla/5.0 (compatible; ResearchBot/2.0)'
                })
                with urllib.request.urlopen(img_req, timeout=15) as img_res:
                    img_data = img_res.read()
                    # Verify minimum image data
                    if len(img_data) < 5000:
                        continue
                    with open(filename, 'wb') as f:
                        f.write(img_data)
                downloaded += 1
            except Exception:
                pass

        return downloaded

    except Exception as e:
        print(f"    API error for '{query}': {e}")
        return 0


def scrape_wikimedia_dataset():
    """
    Scrape India-focused humanitarian images from Wikimedia Commons.
    Organized by category (food, clothing, medicine).
    """
    print("=" * 72)
    print("  PHASE 1: Wikimedia Commons Scraper (India-Focused)")
    print("=" * 72)

    total_downloaded: int = 0

    for category, queries in SEARCH_QUERIES.items():
        cat_dir = os.path.join(RAW_DIR, category)
        os.makedirs(cat_dir, exist_ok=True)

        cat_total: int = 0
        print(f"\n  [{category.upper()}] Scraping {len(queries)} search queries...")

        for i, query in enumerate(queries):
            count: int = int(fetch_wikimedia_images(
                query, count=20, output_dir=cat_dir,
                category=category, start_idx=cat_total
            ))
            cat_total += count  # type: ignore[operator]

            if (i + 1) % 5 == 0 or i == len(queries) - 1:
                print(f"    Progress: {i+1}/{len(queries)} queries | {cat_total} images")

            time.sleep(0.5)  # Respectful delay between API calls

        print(f"  [{category.upper()}] Total: {cat_total} images downloaded")
        total_downloaded += cat_total  # type: ignore[operator]

    print(f"\n  TOTAL WIKIMEDIA IMAGES: {total_downloaded}")
    return total_downloaded


# ─── Kaggle Dataset Downloader ────────────────────────────────────────────────

KAGGLE_DATASETS = [
    # ─── FOOD DATASETS ────────────────────────────────────────────────────
    {
        "slug": "iamsouravbanerjee/indian-food-images-dataset",
        "category": "food",
        "description": "Indian Food Images — 4000 images, 80 classes (dal, chapati, rice, biryani, etc.)"
    },
    {
        "slug": "muratkokludataset/rice-image-dataset",
        "category": "food",
        "description": "Rice Image Dataset — 75,000 images of rice varieties (Basmati, Jasmine, etc.)"
    },
    {
        "slug": "kritikseth/fruit-and-vegetable-image-recognition",
        "category": "food",
        "description": "Fruit & Vegetable Recognition — food items for donation classification"
    },
    {
        "slug": "trolukovich/food11-image-dataset",
        "category": "food",
        "description": "Food-11 — 16,643 images across 11 categories (bread, rice, meat, etc.)"
    },

    # ─── BACKGROUND / NOISE DATASETS ──────────────────────────────────────
    # To prevent class imbalance against 95k food images, we need negative examples.
    # We repurpose clothing/fashion datasets as "background/not-food" images.
    {
        "slug": "agrigorev/clothing-dataset-full",
        "category": "background",
        "description": "Clothing Dataset (Full) — 5000+ high-res non-food images"
    },
    {
        "slug": "validmodel/fashion-apparel-image-classification-dataset",
        "category": "background",
        "description": "Fashion Apparel — 5,413 non-food images for CNN classification"
    },
    {
        "slug": "paramaggarwal/fashion-product-images-small",
        "category": "background",
        "description": "Fashion Product Images — 40,000+ diverse non-food images"
    },

]


def download_kaggle_datasets():
    """
    Download datasets from Kaggle using the Kaggle API.

    Prerequisites:
        1. pip install kaggle
        2. Place kaggle.json in ~/.kaggle/kaggle.json
           Get it from: https://www.kaggle.com/settings -> API -> Create New Token
    """
    print("\n" + "=" * 72)
    print("  PHASE 2: Kaggle Dataset Downloader")
    print("=" * 72)

    try:
        import kaggle
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        print("  ✅ Kaggle API authenticated successfully\n")
    except ImportError:
        print("  ⚠️  Kaggle not installed. Run: pip install kaggle")
        print("     Then place your kaggle.json in ~/.kaggle/")
        print("     Get it from: https://www.kaggle.com/settings -> API -> Create New Token")
        return False
    except Exception as e:
        print(f"  ⚠️  Kaggle auth failed: {e}")
        print("     Place your kaggle.json in ~/.kaggle/kaggle.json")
        print("     Get it from: https://www.kaggle.com/settings -> API -> Create New Token")
        return False

    kaggle_dir = os.path.join(BASE_DIR, "data", "kaggle")

    for ds in KAGGLE_DATASETS:
        ds_dir = os.path.join(kaggle_dir, ds["category"], ds["slug"].split("/")[-1])
        os.makedirs(ds_dir, exist_ok=True)

        print(f"  Downloading: {ds['slug']}")
        print(f"    Category: {ds['category']} | {ds['description']}")

        try:
            api.dataset_download_files(
                ds["slug"],
                path=ds_dir,
                unzip=True,
                quiet=False
            )
            print(f"    ✅ Downloaded to {ds_dir}\n")
        except Exception as e:
            print(f"    ❌ Failed: {e}\n")

    return True


# ─── Dataset Organizer (Train/Val Split) ──────────────────────────────────────


def organize_into_splits():
    """
    Organize all scraped/downloaded images into PyTorch ImageFolder structure.

    Final structure:
        data/dataset/
        ├── train/
        │   ├── food/       (80% of food images)
        │   ├── clothing/   (80% of clothing images)
        │   └── medicine/   (80% of medicine images)
        └── val/
            ├── food/       (20% of food images)
            ├── clothing/   (20% of clothing images)
            └── medicine/   (20% of medicine images)
    """
    print("\n" + "=" * 72)
    print("  PHASE 3: Organizing Dataset (Train/Val Split)")
    print("=" * 72)

    categories = ["food", "background"]
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    # Create directory structure
    for split in ["train", "val"]:
        for cat in categories:
            os.makedirs(os.path.join(DATASET_DIR, split, cat), exist_ok=True)

    total_stats = {"train": {}, "val": {}}

    for cat in categories:
        # Collect all images for this category from multiple sources
        all_images = []

        # Source 1: Wikimedia raw scrapes
        wiki_dir = os.path.join(RAW_DIR, cat)
        if os.path.exists(wiki_dir):
            for f in os.listdir(wiki_dir):
                if Path(f).suffix.lower() in extensions:
                    all_images.append(os.path.join(wiki_dir, f))

        # Source 2: Kaggle downloads
        kaggle_dir = os.path.join(BASE_DIR, "data", "kaggle", cat)
        if os.path.exists(kaggle_dir):
            for root, dirs, files in os.walk(kaggle_dir):
                for f in files:
                    if Path(f).suffix.lower() in extensions:
                        all_images.append(os.path.join(root, f))

        # Shuffle and split
        random.shuffle(all_images)
        split_idx = int(len(all_images) * TRAIN_SPLIT)
        train_images = all_images[:split_idx]  # type: ignore[index]
        val_images = all_images[split_idx:]  # type: ignore[index]

        # Copy to dataset structure
        for i, img_path in enumerate(train_images):
            ext = Path(img_path).suffix
            dst = os.path.join(DATASET_DIR, "train", cat, f"{cat}_{i:05d}{ext}")
            shutil.copy2(img_path, dst)

        for i, img_path in enumerate(val_images):
            ext = Path(img_path).suffix
            dst = os.path.join(DATASET_DIR, "val", cat, f"{cat}_{i:05d}{ext}")
            shutil.copy2(img_path, dst)

        total_stats["train"][cat] = len(train_images)
        total_stats["val"][cat] = len(val_images)

        print(f"  [{cat.upper():>10s}] Train: {len(train_images):>5d} | Val: {len(val_images):>5d} | Total: {len(all_images)}")

    # Summary
    train_total = sum(total_stats["train"].values())
    val_total = sum(total_stats["val"].values())
    print(f"\n  {'TOTAL':>12s}  Train: {train_total:>5d} | Val: {val_total:>5d} | Total: {train_total + val_total}")
    print(f"\n  Dataset saved to: {DATASET_DIR}")

    return total_stats


# ─── Main Pipeline ────────────────────────────────────────────────────────────

def main():
    print("\n" + "▓" * 72)
    print("  INDIA-FOCUSED HUMANITARIAN DATASET BUILDER")
    print("  For MobileNetV2 Transfer Learning Fine-Tuning")
    print("▓" * 72)

    # Phase 1: Wikimedia (always works, no auth needed)
    wiki_count = scrape_wikimedia_dataset()

    # Phase 2: Kaggle (optional, needs credentials)
    print("\n  Attempting Kaggle download (optional)...")
    kaggle_ok = download_kaggle_datasets()

    # Phase 3: Organize into train/val splits
    stats = organize_into_splits()

    # Final report
    print("\n" + "▓" * 72)
    print("  DATASET READY FOR TRAINING")
    print("▓" * 72)
    print(f"\n  Structure: {DATASET_DIR}/")
    print(f"  ├── train/  (food, background)")
    print(f"  └── val/    (food, background)")
    print(f"\n  Next step: python src/train_mobilenet.py")
    if not kaggle_ok:
        print(f"\n  ℹ️  To add Kaggle data:")
        print(f"     1. pip install kaggle")
        print(f"     2. Go to https://www.kaggle.com/settings → API → Create New Token")
        print(f"     3. Place kaggle.json in ~/.kaggle/")
        print(f"     4. Re-run this script")
    print()


if __name__ == "__main__":
    main()
