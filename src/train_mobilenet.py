"""
MobileNetV2 Transfer Learning Fine-Tuning for Humanitarian Donation Classification

Fine-tunes a pretrained MobileNetV2 (~3.5M parameters) on the India-focused
humanitarian dataset to classify images into 2 categories (Binary Classification):
  - Food (rice, dal, atta, ration kits, grain sacks)
  - Background (noise, non-food items, empty warehouses, trucks, clothing)

Architecture:
  MobileNetV2 uses depthwise separable convolutions and inverted residual blocks
  with linear bottlenecks. The final classifier layer (originally 1000 ImageNet
  classes) is replaced with a 2-class layer for Food vs Background.

  Only the final classifier layer is trainable initially (feature extraction),
  then the last few convolutional blocks are unfrozen for fine-tuning.

Usage:
  1. First run: python src/download_dataset.py  (to build training data)
  2. Then run:  python src/train_mobilenet.py    (to train the model)

The trained model is saved to: models/mobilenet_v2_humanitarian.pth
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets, models
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import os
import sys
import time
import csv
import copy
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# ─── Configuration ────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "data", "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "data", "results")

# Hyperparameters
NUM_CLASSES = 2  # Food, Background
BATCH_SIZE = 32
NUM_EPOCHS_FEATURE_EXTRACT = 10  # Phase 1: train only classifier head
NUM_EPOCHS_FINE_TUNE = 10        # Phase 2: unfreeze last blocks
LEARNING_RATE_FE = 0.001         # Feature extraction learning rate
LEARNING_RATE_FT = 0.0001        # Fine-tuning learning rate (lower)
IMAGE_SIZE = 224                 # MobileNetV2 input size

# Device selection
DEVICE = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")

# ─── Data Transforms ─────────────────────────────────────────────────────────

# Training transforms with augmentation for better generalization
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Validation transforms (no augmentation, just resize + normalize)
val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ─── Model Setup ─────────────────────────────────────────────────────────────

def create_model(num_classes, feature_extract=True):
    """
    Create MobileNetV2 with modified classifier for humanitarian categories.

    Phase 1 (feature_extract=True):
      Freeze all convolutional layers, train only the new classifier head.

    Phase 2 (feature_extract=False):
      Unfreeze the last few convolutional blocks for fine-tuning.
    """
    weights = MobileNet_V2_Weights.DEFAULT
    model = mobilenet_v2(weights=weights)

    # Replace the final classifier(1) → 3 classes instead of 1000
    # Original: classifier = Sequential(Dropout(0.2), Linear(1280, 1000))
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes)

    if feature_extract:
        # Freeze all feature extraction layers
        for param in model.features.parameters():
            param.requires_grad = False

    print(f"  Model: MobileNetV2")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"  Classes: {num_classes} (Food, Clothing, Medicine)")
    print(f"  Device: {DEVICE}")

    return model.to(DEVICE)


def unfreeze_last_blocks(model, num_blocks=4):
    """
    Unfreeze the last N convolutional blocks for fine-tuning.

    MobileNetV2 has 19 inverted residual blocks (features[0] to features[18]).
    Unfreezing the last 4 blocks allows the model to adapt its learned features
    to the specific characteristics of Indian humanitarian donation images.
    """
    total_blocks = len(model.features)
    unfreeze_from = total_blocks - num_blocks

    for i, block in enumerate(model.features):
        if i >= unfreeze_from:
            for param in block.parameters():
                param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Unfroze last {num_blocks} blocks (from block {unfreeze_from})")
    print(f"  Trainable parameters now: {trainable:,}")


# ─── Training Loop ───────────────────────────────────────────────────────────

def train_model(model, dataloaders, criterion, optimizer, num_epochs, phase_name):
    """
    Train the model for a given number of epochs.

    Returns:
        model: Best model weights
        history: Training history (loss, accuracy per epoch)
    """
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print(f"\n  {'─' * 60}")
    print(f"  {phase_name} ({num_epochs} epochs)")
    print(f"  {'─' * 60}")

    for epoch in range(num_epochs):
        epoch_start = time.time()

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            total_samples = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()
                total_samples += inputs.size(0)

            epoch_loss = running_loss / total_samples if total_samples > 0 else 0
            epoch_acc = running_corrects / total_samples if total_samples > 0 else 0

            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_acc"].append(epoch_acc)

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        epoch_time = time.time() - epoch_start
        train_acc = history["train_acc"][-1]
        val_acc = history["val_acc"][-1]
        train_loss = history["train_loss"][-1]
        val_loss = history["val_loss"][-1]

        print(f"  Epoch {epoch+1:>2d}/{num_epochs} | "
              f"Train: {train_acc*100:5.1f}% (loss: {train_loss:.4f}) | "
              f"Val: {val_acc*100:5.1f}% (loss: {val_loss:.4f}) | "
              f"{epoch_time:.1f}s")

    print(f"\n  Best Validation Accuracy: {best_acc*100:.2f}%")
    model.load_state_dict(best_model_wts)
    return model, history


# ─── Main Pipeline ────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("  MobileNetV2 TRANSFER LEARNING — HUMANITARIAN DONATION CLASSIFIER")
    print("  Architecture: Depthwise Separable Convolutions + Inverted Residuals")
    print("=" * 72)

    # Verify dataset exists
    train_dir = os.path.join(DATASET_DIR, "train")
    val_dir = os.path.join(DATASET_DIR, "val")

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        print(f"\n  ERROR: Dataset not found at {DATASET_DIR}")
        print(f"  Run 'python src/download_dataset.py' first to build the dataset.")
        sys.exit(1)

    # Load datasets using PyTorch ImageFolder
    print(f"\n  Loading dataset from: {DATASET_DIR}")

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)

    print(f"  Classes: {train_dataset.classes}")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")

    if len(train_dataset) == 0:
        print("\n  ERROR: No training images found. Run download_dataset.py first.")
        sys.exit(1)

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=2, pin_memory=True),
        'val': DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=True),
    }

    # ─── Phase 1: Feature Extraction (frozen backbone) ────────────────────

    print(f"\n{'═' * 72}")
    print("  PHASE 1: Feature Extraction (frozen backbone)")
    print(f"{'═' * 72}")

    model = create_model(NUM_CLASSES, feature_extract=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE_FE
    )

    model, history_fe = train_model(
        model, dataloaders, criterion, optimizer,
        NUM_EPOCHS_FEATURE_EXTRACT,
        "Phase 1: Feature Extraction"
    )

    # ─── Phase 2: Fine-Tuning (unfreeze last blocks) ──────────────────────

    print(f"\n{'═' * 72}")
    print("  PHASE 2: Fine-Tuning (unfreezing last 4 conv blocks)")
    print(f"{'═' * 72}")

    unfreeze_last_blocks(model, num_blocks=4)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE_FT  # Lower LR for fine-tuning
    )

    model, history_ft = train_model(
        model, dataloaders, criterion, optimizer,
        NUM_EPOCHS_FINE_TUNE,
        "Phase 2: Fine-Tuning"
    )

    # ─── Save Model ──────────────────────────────────────────────────────

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "mobilenet_v2_humanitarian.pth")

    torch.save({
        'model_state_dict': model.state_dict(),
        'classes': train_dataset.classes,
        'num_classes': NUM_CLASSES,
        'architecture': 'MobileNetV2',
        'input_size': IMAGE_SIZE,
    }, model_path)

    print(f"\n  ✅ Model saved to: {model_path}")

    # ─── Save Training History ─────────────────────────────────────────────

    os.makedirs(RESULTS_DIR, exist_ok=True)
    history_csv = os.path.join(RESULTS_DIR, "training_history.csv")

    with open(history_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Phase", "Epoch", "Train_Loss", "Train_Acc", "Val_Loss", "Val_Acc"])

        for i in range(len(history_fe["train_loss"])):
            writer.writerow([
                "Feature_Extraction", i + 1,
                f"{history_fe['train_loss'][i]:.4f}",
                f"{history_fe['train_acc'][i]:.4f}",
                f"{history_fe['val_loss'][i]:.4f}",
                f"{history_fe['val_acc'][i]:.4f}",
            ])

        for i in range(len(history_ft["train_loss"])):
            writer.writerow([
                "Fine_Tuning", i + 1,
                f"{history_ft['train_loss'][i]:.4f}",
                f"{history_ft['train_acc'][i]:.4f}",
                f"{history_ft['val_loss'][i]:.4f}",
                f"{history_ft['val_acc'][i]:.4f}",
            ])

    print(f"  Training history saved to: {history_csv}")

    # Final summary
    final_val_acc = history_ft["val_acc"][-1] if history_ft["val_acc"] else 0
    print(f"\n{'═' * 72}")
    print(f"  TRAINING COMPLETE")
    print(f"  Final Validation Accuracy: {final_val_acc*100:.2f}%")
    print(f"  Model: {model_path}")
    print(f"  Target: >= 88% (per tech doc specification)")
    if final_val_acc >= 0.88:
        print(f"  ✅ BENCHMARK MET!")
    else:
        print(f"  ℹ️  Add more training data or increase epochs to reach 88%")
    print(f"{'═' * 72}\n")


if __name__ == "__main__":
    main()
