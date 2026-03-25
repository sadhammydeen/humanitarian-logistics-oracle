import cv2
import os
import glob
from pathlib import Path
import argparse

def apply_clahe_to_directory(input_dir, output_dir, clip_limit=2.0, tile_grid_size=(8,8)):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize CLAHE object
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    # Supported image extensions
    extensions = ['*.jpg', '*.jpeg', '*.png']
    image_paths = []
    
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(input_dir, ext.upper())))
        
    if not image_paths:
        print(f"No images found in {input_dir}. Please add some raw dataset images there first.")
        return

    for img_path in image_paths:
        # Read the image
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        # Convert to LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        
        # Split the channels
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L-channel
        cl = clahe.apply(l)
        
        # Merge the CLAHE enhanced L-channel with the a and b channel
        limg = cv2.merge((cl,a,b))
        
        # Convert image from LAB Color model to BGR model
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        # Save the image
        filename = os.path.basename(img_path)
        output_path = os.path.join(output_dir, f"clahe_{filename}")
        cv2.imwrite(output_path, final)
        print(f"Processed: {filename} -> {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Apply CLAHE to images to improve visibility in low-light conditions.')
    parser.add_argument('--input', type=str, default='../data/raw', help='Input directory of raw images')
    parser.add_argument('--output', type=str, default='../data/processed', help='Output directory for CLAHE enhanced images')
    args = parser.parse_args()
    
    apply_clahe_to_directory(args.input, args.output)
    print("CLAHE processing complete!")
