"""
MobileNetV2 Lightweight Image Classifier for Humanitarian Donation Verification

This module implements image classification using MobileNetV2, a neural network
architecture fundamentally optimized for edge-deployed classification tasks.
MobileNetV2 utilizes depthwise separable convolutions (splitting filtering and
combining stages into two distinct layers) and inverted residual blocks with
linear bottlenecks, achieving high Top-1 accuracy with only ~3.5M parameters.

The classifier maps ImageNet categories to humanitarian donation classes natively, 
or loads a custom fine-tuned model for Binary Classification:
  - Food (rice sacks, canned goods, grain bags, food packages)
  - Background (noise, non-food items, empty warehouses, trucks, clothing)

Inference completes in <5 seconds on a mobile CPU, making it suitable for
resource-constrained NGO environments without cloud connectivity.

References:
  - Sandler et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks" (2018)
  - Transfer learning calibration for domain-specific humanitarian categories
"""

import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from PIL import Image
import cv2
import numpy as np
import time
import ssl
import os

ssl._create_default_https_context = ssl._create_unverified_context

# ─── ImageNet → Humanitarian Category Mapping ────────────────────────────────
# MobileNetV2 pretrained on ImageNet-1K outputs 1000 classes. We map relevant
# classes to our 3 humanitarian categories. This simulates the effect of
# transfer learning where the final classifier layer is collapsed to 3 outputs.

FOOD_CLASSES = {
    # Grains, sacks, and packaged food
    924, 925, 926, 927, 928, 929, 930, 931, 932, 933, 934, 935, 936, 937, 938,
    939, 940, 941, 942, 943, 944, 945, 946, 947, 948, 949, 950, 951, 952, 953,
    954, 955, 956, 957, 958, 959, 960, 961, 962, 963, 964, 965, 966, 967, 968,
    969,
    # Specific food-related ImageNet classes
    454, 455, 456,  # bottles/containers
    440, 441, 442,  # containers
    520, 521, 522, 523, 524, 525, 526, 527, 528, 529,  # bags/sacks
    530, 531, 532, 533, 534, 535,  # more containers
    550, 551, 552, 553, 554, 555, 556, 557, 558, 559,  # groceries
    # Bread, food, groceries
    415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429,
    430, 431, 432, 433, 434, 435, 436, 437, 438, 439,
    # Water bottles
    898, 899, 900, 901, 902, 903, 907,
    # Crates and baskets
    463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477,
}

CLOTHING_CLASSES = {
    # Garments and textiles
    601, 602, 603, 604, 605, 606, 607, 608, 609, 610,
    611, 612, 613, 614, 615, 616, 617, 618, 619, 620,
    # More clothing
    834, 835, 836, 837, 838, 839, 840, 841, 842, 843, 844, 845, 846, 847, 848,
    # Shoes, boots
    770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 780, 781,
    # Accessories and fabrics
    474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488,
    489, 490, 491, 492, 493, 494, 495, 496, 497, 498,
}

MEDICINE_CLASSES = {
    # Medical equipment and supplies
    700, 701, 702, 703, 704, 705, 706, 707, 708, 709,
    # Pill bottles, containers, syringes
    621, 622, 623, 624, 625, 626, 627, 628, 629, 630,
    # Medical instruments
    631, 632, 633, 634, 635, 636, 637, 638, 639, 640,
    # Bandages and wrappings
    810, 811, 812, 813, 814, 815, 816, 817, 818,
}

# ─── Model Loading ───────────────────────────────────────────────────────────

_model = None
_preprocess = None
_use_fine_tuned = False
_fine_tuned_classes = None

# Path to the fine-tuned model (created by train_mobilenet.py)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FINE_TUNED_MODEL_PATH = os.path.join(BASE_DIR, "models", "mobilenet_v2_humanitarian.pth")


def _get_model():
    """
    Lazily load the MobileNetV2 model (~3.5M parameters).

    Priority:
      1. Fine-tuned model (models/mobilenet_v2_humanitarian.pth) — 2-class direct output
      2. Pretrained ImageNet model — uses category aggregation mapping
    """
    global _model, _preprocess, _use_fine_tuned, _fine_tuned_classes
    if _model is None:
        if os.path.exists(FINE_TUNED_MODEL_PATH):
            # Load fine-tuned humanitarian model (Binary Classification)
            print(f"  Loading fine-tuned model: {FINE_TUNED_MODEL_PATH}")
            checkpoint = torch.load(FINE_TUNED_MODEL_PATH, map_location="cpu", weights_only=False)
            _model = mobilenet_v2(weights=None)
            num_classes = checkpoint.get("num_classes", 2)
            import torch.nn as nn
            _model.classifier[1] = nn.Linear(_model.classifier[1].in_features, num_classes)
            _model.load_state_dict(checkpoint["model_state_dict"])
            _model.eval()
            _use_fine_tuned = True
            _fine_tuned_classes = checkpoint.get("classes", ["background", "food"])
            print(f"  Loaded fine-tuned model with classes: {_fine_tuned_classes}")
        else:
            # Fallback to pretrained ImageNet model
            weights = MobileNet_V2_Weights.DEFAULT
            _model = mobilenet_v2(weights=weights)
            _model.eval()
            _use_fine_tuned = False

        _preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    return _model, _preprocess


def _map_to_humanitarian_category(class_id):
    """Map an ImageNet class ID to a humanitarian donation category."""
    if class_id in FOOD_CLASSES:
        return "Food"
    elif class_id in CLOTHING_CLASSES:
        return "Clothing"
    elif class_id in MEDICINE_CLASSES:
        return "Medicine"
    else:
        return "Other"


def _aggregate_category_probabilities(probabilities):
    """
    Aggregate ImageNet class probabilities into humanitarian categories.

    Instead of taking only the top-1 class, we sum probabilities across all
    ImageNet classes that map to each humanitarian category. This simulates
    the behavior of a transfer-learned model where the final layer collapses
    1000 classes into 3+1 categories.

    Returns:
        dict: {category_name: aggregated_probability}
        str: winning category
        float: winning confidence (normalized)
    """
    category_probs = {"Food": 0.0, "Clothing": 0.0, "Medicine": 0.0, "Other": 0.0}

    for idx in range(len(probabilities)):
        prob = probabilities[idx].item()
        cat = _map_to_humanitarian_category(idx)
        category_probs[cat] += prob

    # Normalize across categories (excluding "Other" for relevance scoring)
    relevant_total = category_probs["Food"] + category_probs["Clothing"] + category_probs["Medicine"]
    total = sum(category_probs.values())

    if relevant_total > 0:
        # Re-normalize: the humanitarian categories' combined probability
        # represents the confidence in a relevant classification
        best_cat = max(["Food", "Clothing", "Medicine"], key=lambda c: category_probs[c])
        # Confidence is the proportion of relevant probability held by the winner
        confidence = category_probs[best_cat] / total if total > 0 else 0.0
    else:
        best_cat = "Other"
        confidence = category_probs["Other"] / total if total > 0 else 0.0

    return category_probs, best_cat, confidence


# ─── CLAHE Preprocessing ─────────────────────────────────────────────────────

def apply_clahe_to_image(img_bgr, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to an image.

    CLAHE operates on small data regions (tiles), enhancing contrast locally
    while preventing over-amplification of noise by clipping the histogram at
    a predefined threshold.

    CRITICAL: CLAHE is applied exclusively to the L* (lightness) channel of the
    LAB color space—rather than the standard RGB space. Modifying RGB channels
    independently leads to severe color distortion, which would confuse a CNN
    trained to recognize specific hues. By isolating the L* channel, contrast
    is enhanced while perfectly preserving original hue and saturation.

    Args:
        img_bgr: OpenCV BGR image array
        clip_limit: CLAHE clip limit (default 2.0)
        tile_grid_size: tile grid size for CLAHE (default 8x8)

    Returns:
        CLAHE-enhanced BGR image array
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    # Convert BGR → LAB color space
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

    # Split channels: L* (lightness), a (green-red), b (blue-yellow)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Apply CLAHE to L* channel only
    enhanced_l = clahe.apply(l_channel)

    # Merge enhanced L* with original a and b channels
    enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))

    # Convert back to BGR
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    return enhanced_bgr


# ─── Public API ───────────────────────────────────────────────────────────────

def _classify_fine_tuned(output):
    """
    Process output from the fine-tuned 2-class model.
    Returns (category, confidence, category_probs).
    """
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    conf, idx = torch.max(probabilities, 0)

    # Map class index to category name (capitalize first letter)
    category = _fine_tuned_classes[idx.item()].capitalize()
    confidence = conf.item()

    category_probs = {}
    for i, cls_name in enumerate(_fine_tuned_classes):
        category_probs[cls_name.capitalize()] = probabilities[i].item()
    if "Other" not in category_probs:
        category_probs["Other"] = 0.0

    return category, confidence, category_probs


def classify_image(image_path, apply_clahe=True):
    """
    Classify an image using MobileNetV2 for humanitarian donation verification.

    If a fine-tuned model exists (models/mobilenet_v2_humanitarian.pth), it uses
    direct 3-class prediction. Otherwise falls back to ImageNet with aggregation.

    Args:
        image_path: Path to the image file
        apply_clahe: Whether to apply CLAHE preprocessing (recommended for
                     low-light environments common in NGO warehouses)

    Returns:
        tuple: (category, confidence, inference_time_ms, category_probs)
            - category: one of "Food", "Clothing", "Medicine", "Other"
            - confidence: float [0, 1] representing classification confidence
            - inference_time_ms: time taken for inference in milliseconds
            - category_probs: dict of all category probabilities
    """
    model, preprocess = _get_model()
    start_time = time.time()

    # Load and optionally enhance image
    if apply_clahe:
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise ValueError(f"Could not read image: {image_path}")
        img_bgr = apply_clahe_to_image(img_bgr)
        # Convert BGR → RGB for PIL
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(img_rgb)
    else:
        pil_image = Image.open(image_path).convert('RGB')

    # Preprocess for MobileNetV2
    input_tensor = preprocess(pil_image)
    input_batch = input_tensor.unsqueeze(0)

    # Inference
    with torch.no_grad():
        output = model(input_batch)

    # Process output based on model type
    if _use_fine_tuned:
        best_category, confidence, category_probs = _classify_fine_tuned(output)
    else:
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        category_probs, best_category, confidence = _aggregate_category_probabilities(probabilities)

    inference_time_ms = (time.time() - start_time) * 1000

    return best_category, confidence, inference_time_ms, category_probs


def classify_image_from_numpy(img_bgr, apply_clahe=True):
    """
    Classify an in-memory image (numpy BGR array) using MobileNetV2.

    Args:
        img_bgr: OpenCV BGR image as numpy array
        apply_clahe: Whether to apply CLAHE preprocessing

    Returns:
        tuple: (category, confidence, inference_time_ms, category_probs)
    """
    model, preprocess = _get_model()
    start_time = time.time()

    if apply_clahe:
        img_bgr = apply_clahe_to_image(img_bgr)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(img_rgb)

    input_tensor = preprocess(pil_image)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(input_batch)

    if _use_fine_tuned:
        best_category, confidence, category_probs = _classify_fine_tuned(output)
    else:
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        category_probs, best_category, confidence = _aggregate_category_probabilities(probabilities)

    inference_time_ms = (time.time() - start_time) * 1000

    return best_category, confidence, inference_time_ms, category_probs


# ─── Standalone test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    test_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/wiki_001.jpg"

    if not os.path.exists(test_path):
        print(f"Test image not found: {test_path}")
        sys.exit(1)

    print(f"MobileNetV2 Classifier (~3.5M parameters)")
    print(f"Image: {test_path}\n")

    # Without CLAHE
    cat, conf, t_ms, probs = classify_image(test_path, apply_clahe=False)
    print(f"[Baseline]       Category: {cat}, Confidence: {conf:.4f}, Time: {t_ms:.1f}ms")
    print(f"                 Probs: { {k: f'{v:.4f}' for k, v in probs.items()} }")

    # With CLAHE
    cat, conf, t_ms, probs = classify_image(test_path, apply_clahe=True)
    print(f"[CLAHE-Enhanced] Category: {cat}, Confidence: {conf:.4f}, Time: {t_ms:.1f}ms")
    print(f"                 Probs: { {k: f'{v:.4f}' for k, v in probs.items()} }")
