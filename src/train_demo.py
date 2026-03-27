"""
Quick Demo Trainer for MobileNetV2 — Humanitarian Donation Classifier

This is a lightweight demo version that:
  - Picks 200 food images already on disk (no download needed)
  - Generates synthetic "background" noise images on the fly
  - Trains 3 rapid epochs (~2-4 minutes on Mac)
  - Prints accuracy and saves the trained model

Run with:
    venv/bin/python src/train_demo.py

Perfect for classroom/teacher demonstrations!
"""

import os
import sys
import random
import shutil
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import MobileNet_V2_Weights
from PIL import Image, ImageFilter
import numpy as np

# ─── Config ──────────────────────────────────────────────────────────────────

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR  = os.path.join(BASE_DIR, "data", "dataset", "train", "food")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
MODEL_OUT  = os.path.join(MODEL_DIR, "mobilenet_v2_humanitarian.pth")
DEMO_DIR   = os.path.join(BASE_DIR, "data", "demo_subset")

RICE_DIR    = os.path.join(BASE_DIR, "data", "kaggle", "food", "rice-image-dataset")
WIKI_BG_DIR = os.path.join(BASE_DIR, "data", "raw", "background")

NUM_FOOD_SAMPLES       = 500   # real Indian food images for demo
NUM_BACKGROUND_SAMPLES = 500   # real rice grain images as negative class
BATCH_SIZE             = 32
NUM_EPOCHS             = 8
LEARNING_RATE          = 5e-4

os.makedirs(MODEL_DIR, exist_ok=True)

# ─── Step 1: Build Demo Subset ───────────────────────────────────────────────

def build_demo_subset():
    food_train = os.path.join(DEMO_DIR, "train", "food")
    food_val   = os.path.join(DEMO_DIR, "val",   "food")
    bg_train   = os.path.join(DEMO_DIR, "train", "background")
    bg_val     = os.path.join(DEMO_DIR, "val",   "background")

    for d in [food_train, food_val, bg_train, bg_val]:
        os.makedirs(d, exist_ok=True)

    # ── Food: use real Indian food donation images ──
    if not os.path.exists(TRAIN_DIR):
        print(f"  ❌ No food images found at {TRAIN_DIR}")
        sys.exit(1)

    all_food = [f for f in os.listdir(TRAIN_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    random.shuffle(all_food)
    selected_food = all_food[:NUM_FOOD_SAMPLES]  # type: ignore[index]

    split = int(len(selected_food) * 0.8)
    for fname in selected_food[:split]:
        shutil.copy2(os.path.join(TRAIN_DIR, fname), os.path.join(food_train, fname))
    for fname in selected_food[split:]:
        shutil.copy2(os.path.join(TRAIN_DIR, fname), os.path.join(food_val, fname))
    print(f"  ✅ Food images:  {split} train | {len(selected_food)-split} val  (real Indian food photos)")

    # ── Background: use real RICE GRAIN images (white-bg lab photos — look nothing like food donation bags) ──
    all_bg = []

    # Source 1: rice grain dataset (white-background grain photos)
    if os.path.exists(RICE_DIR):
        for root, _, files in os.walk(RICE_DIR):
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    all_bg.append(os.path.join(root, f))

    # Source 2: Wikimedia background (warehouses, streets, etc.)
    if os.path.exists(WIKI_BG_DIR):
        for f in os.listdir(WIKI_BG_DIR):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                all_bg.append(os.path.join(WIKI_BG_DIR, f))

    if len(all_bg) < 10:
        print("  ⚠️  Warning: very few background images found.")
        print(f"     Looked in: {RICE_DIR}")
    else:
        print(f"  📦 Found {len(all_bg)} background images (rice grain + Wikimedia)")

    random.shuffle(all_bg)
    selected_bg = all_bg[:NUM_BACKGROUND_SAMPLES]  # type: ignore[index]
    bg_split = int(len(selected_bg) * 0.8)

    for i, src in enumerate(selected_bg[:bg_split]):
        ext = os.path.splitext(src)[1]
        shutil.copy2(src, os.path.join(bg_train, f"bg_{i:04d}{ext}"))
    for i, src in enumerate(selected_bg[bg_split:]):
        ext = os.path.splitext(src)[1]
        shutil.copy2(src, os.path.join(bg_val, f"bg_{i:04d}{ext}"))

    print(f"  ✅ Background:   {bg_split} train | {len(selected_bg)-bg_split} val  (real rice grain photos)")



# ─── Step 2: Dataset & DataLoader ────────────────────────────────────────────

def get_dataloaders():
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    from torchvision.datasets import ImageFolder
    train_ds = ImageFolder(os.path.join(DEMO_DIR, "train"), transform=transform)
    val_ds   = ImageFolder(os.path.join(DEMO_DIR, "val"),   transform=val_transform)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return train_dl, val_dl, train_ds.classes


# ─── Step 3: Model Setup ─────────────────────────────────────────────────────

def get_model(num_classes):
    model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

    # Freeze all layers first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last 2 inverted residual blocks so backbone adapts to our data
    for layer in model.features[-3:]:
        for param in layer.parameters():
            param.requires_grad = True

    # Replace final classifier head
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    # classifier head is always trainable (newly created)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable:,} (last 2 blocks + classifier head)")

    return model


# ─── Step 4: Training Loop ───────────────────────────────────────────────────

def train(model, train_dl, val_dl, device, classes):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)

    best_acc = 0.0
    print()

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss: float = 0.0
        correct: int = 0
        total: int = 0

        for batch_idx, (inputs, labels) in enumerate(train_dl):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += int((predicted == labels).sum().item())
            total += labels.size(0)

        train_acc = 100 * correct / total

        # Validation
        model.eval()
        val_correct: int = 0
        val_total: int = 0
        with torch.no_grad():
            for inputs, labels in val_dl:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                val_correct += int((predicted == labels).sum().item())
                val_total += labels.size(0)

        val_acc = 100 * val_correct / val_total

        print(f"  Epoch {epoch+1}/{NUM_EPOCHS}  |  "
              f"Train Acc: {train_acc:.1f}%  |  "
              f"Val Acc: {val_acc:.1f}%  |  "
              f"Loss: {running_loss/len(train_dl):.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            # Save checkpoint
            torch.save({
                "model_state_dict": model.state_dict(),
                "num_classes": len(classes),
                "classes": classes,
                "val_acc": val_acc,
            }, MODEL_OUT)
            print(f"    💾 Model saved (val acc: {val_acc:.1f}%)")

    return best_acc


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print()
    print("▓" * 64)
    print("  MobileNetV2 QUICK DEMO TRAINER")
    print("  2-Class Binary: Food vs Background Noise")
    print("▓" * 64)

    # Auto-detect best device: CUDA → MPS (Apple Silicon) → CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"\n  🖥  Device: {device}")

    print("\n  STEP 1: Building 500-image demo subset...")
    build_demo_subset()

    print("\n  STEP 2: Loading data...")
    train_dl, val_dl, classes = get_dataloaders()
    print(f"  Classes: {classes}")
    print(f"  Train batches: {len(train_dl)}  |  Val batches: {len(val_dl)}")

    print("\n  STEP 3: Loading pretrained MobileNetV2 (ImageNet weights)...")
    model = get_model(num_classes=len(classes)).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,}  |  Trainable: {trainable:,} (classifier head only)")

    print(f"\n  STEP 4: Training {NUM_EPOCHS} epochs on {device}...\n")
    t0 = time.time()
    best_acc = train(model, train_dl, val_dl, device, classes)
    elapsed = time.time() - t0

    print()
    print("▓" * 64)
    print(f"  ✅ DEMO TRAINING COMPLETE")
    print(f"  Best Validation Accuracy : {best_acc:.1f}%")
    print(f"  Training Time            : {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Model saved to           : {MODEL_OUT}")
    print()
    print("  Next: run the full server with:")
    print("    venv/bin/python -m uvicorn src.main:app --port 8000")
    print("▓" * 64)
    print()


if __name__ == "__main__":
    main()
