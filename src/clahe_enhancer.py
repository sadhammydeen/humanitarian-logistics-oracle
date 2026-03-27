"""
CLAHE (Contrast Limited Adaptive Histogram Equalization) Image Enhancer

This module addresses the significant operational hurdle of poor lighting conditions
in NGO warehouses and evening delivery scenarios. Standard CNNs experience severe
performance degradation when analyzing low-contrast images because critical edge
and texture features are obscured.

METHODOLOGY:
  Traditional Histogram Equalization globally distributes pixel intensities, which
  frequently results in noise amplification and the washing out of localized details.
  CLAHE, conversely, operates on small data regions (tiles), enhancing contrast
  locally while preventing over-amplification of noise by clipping the histogram
  at a predefined threshold.

LAB COLOR SPACE L* CHANNEL ISOLATION:
  Applying CLAHE exclusively to the L* (lightness) channel of the LAB color space—
  rather than the standard RGB space—is a crucial methodological decision. Modifying
  the RGB channels independently leads to severe color distortion, which would
  confuse a CNN trained to recognize specific hues of food packaging. By isolating
  the L* channel, the system enhances contrast while perfectly preserving the
  original hue and saturation.

EMPIRICAL EVIDENCE:
  Recent studies demonstrate that combining CLAHE with MobileNetV2 significantly
  improves feature visibility, accelerates model convergence, and boosts Mean
  Average Precision in low-light environments.

References:
  - Zuiderveld, K. "Contrast Limited Adaptive Histogram Equalization" (1994)
  - Pizer et al., "Adaptive Histogram Equalization and Its Variations" (1987)
"""

import cv2
import os
import glob
from pathlib import Path
import argparse
import numpy as np


def apply_clahe_single(image_path, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply CLAHE to a single image file and return the enhanced image.

    This function is designed for API pipeline use where the enhanced image
    needs to be passed directly to MobileNetV2 for classification.

    Args:
        image_path: Path to the input image file
        clip_limit: Contrast limiting threshold (default: 2.0)
        tile_grid_size: Size of grid for histogram equalization (default: 8x8)

    Returns:
        numpy.ndarray: CLAHE-enhanced BGR image, or None if image cannot be read
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
    return apply_clahe_to_numpy(img, clip_limit, tile_grid_size)


def apply_clahe_to_numpy(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply CLAHE to an in-memory BGR image (numpy array).

    The processing pipeline:
      1. Convert BGR → LAB color space
      2. Split into L* (lightness), a (green-red), b (blue-yellow) channels
      3. Apply CLAHE only to the L* channel
      4. Merge enhanced L* with original a and b channels
      5. Convert back to BGR

    This preserves hue and saturation while enhancing contrast—critical for
    CNN-based classification where color features inform category predictions.

    Args:
        img: OpenCV BGR image as numpy array
        clip_limit: Contrast limiting threshold
        tile_grid_size: Size of grid for histogram equalization

    Returns:
        numpy.ndarray: CLAHE-enhanced BGR image
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    # Convert to LAB color space — isolating lightness from color
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Split the channels
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Apply CLAHE exclusively to L* (lightness) channel
    enhanced_l = clahe.apply(l_channel)

    # Merge the CLAHE-enhanced L* with the original a and b channels
    enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))

    # Convert back from LAB to BGR color model
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    return enhanced_bgr


def apply_clahe_to_directory(input_dir, output_dir, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Batch-process all images in a directory with CLAHE enhancement.

    This function processes all JPG/JPEG/PNG images in the input directory,
    applies CLAHE to each on the LAB L* channel, and saves the enhanced
    images to the output directory with a 'clahe_' prefix.

    Args:
        input_dir: Directory containing raw images
        output_dir: Directory to save CLAHE-enhanced images
        clip_limit: CLAHE clip limit parameter
        tile_grid_size: CLAHE tile grid size parameter
    """
    os.makedirs(output_dir, exist_ok=True)

    # Supported image extensions
    extensions = ['*.jpg', '*.jpeg', '*.png']
    image_paths = []

    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(input_dir, ext.upper())))

    if not image_paths:
        print(f"No images found in {input_dir}. Please add some raw dataset images there first.")
        return 0

    processed_count = 0
    for img_path in image_paths:
        enhanced = apply_clahe_single(img_path, clip_limit, tile_grid_size)
        if enhanced is None:
            continue

        filename = os.path.basename(img_path)
        output_path = os.path.join(output_dir, f"clahe_{filename}")
        cv2.imwrite(output_path, enhanced)
        print(f"Processed: {filename} -> {output_path}")
        processed_count += 1

    return processed_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Apply CLAHE (LAB L* channel) to images for low-light enhancement.'
    )
    parser.add_argument('--input', type=str, default='../data/raw',
                        help='Input directory of raw images')
    parser.add_argument('--output', type=str, default='../data/processed',
                        help='Output directory for CLAHE enhanced images')
    parser.add_argument('--clip-limit', type=float, default=2.0,
                        help='CLAHE clip limit (default: 2.0)')
    parser.add_argument('--tile-size', type=int, default=8,
                        help='CLAHE tile grid size (default: 8)')
    args = parser.parse_args()

    count = apply_clahe_to_directory(
        args.input, args.output,
        clip_limit=args.clip_limit,
        tile_grid_size=(args.tile_size, args.tile_size)
    )
    print(f"\nCLAHE processing complete! {count} images enhanced.")
    print(f"Method: LAB color space, L* channel isolation, clip_limit={args.clip_limit}")
