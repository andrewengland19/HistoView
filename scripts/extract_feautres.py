# extract_features_downsample.py
from pathlib import Path
import numpy as np
import tifffile as tif
import cv2
import csv

# === CONFIG PATHS ===
HOME = Path.home()
INPUT = HOME / "Microscopy/Cohort1_TPH2/dataset"        # normalized TIFFs
PNG_OUTPUT = HOME / "Microscopy/Cohort1_TPH2/PNG"
PNG_OUTPUT.mkdir(parents=True, exist_ok=True)

FEATURE_OUTPUT = HOME / "Microscopy/Cohort1_TPH2/features"
FEATURE_OUTPUT.mkdir(parents=True, exist_ok=True)

# Downsample settings
PIXEL_SIZE = 128
EDGE_SIZE = 128

# Loop over all rats
for rat_dir in sorted(INPUT.iterdir()):
    if not rat_dir.is_dir():
        continue
    rat_name = rat_dir.name
    print(f"\nProcessing rat: {rat_name}")

    features_csv = FEATURE_OUTPUT / f"features_{rat_name}.csv"
    with open(features_csv, "w", newline="") as f_csv:
        writer = csv.writer(f_csv)

        ch1_dir = rat_dir / "CH1"
        if not ch1_dir.exists():
            print(f"  No CH1 folder for {rat_name}, skipping")
            continue

        images = sorted(ch1_dir.glob("*.tif"))
        for i, img_path in enumerate(images, 1):
            print(f"[{i}/{len(images)}] {img_path.name}")

            # Load TIFF
            img = tif.imread(str(img_path))

            # Multi-page TIFF? Take page 3 (DAPI)
            if img.ndim == 3:
                if img.shape[0] > 1:
                    img = img[2]
                elif img.shape[2] == 3:
                    img = img[:, :, 2]

            # Downsample
            img_small = cv2.resize(img, (PIXEL_SIZE, PIXEL_SIZE), interpolation=cv2.INTER_AREA)
            img_edge = cv2.resize(img, (EDGE_SIZE, EDGE_SIZE), interpolation=cv2.INTER_AREA)

            # Save PNG for QC
            png_dir = PNG_OUTPUT / rat_name / "CH1"
            png_dir.mkdir(parents=True, exist_ok=True)
            png_path = png_dir / (img_path.stem + ".png")
            cv2.imwrite(
                str(png_path),
                cv2.normalize(img_small, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            )

            # Normalize for features
            img_small = img_small.astype(np.float32)
            if img_small.max() > 1.0:
                img_small /= 65535.0
            edges = cv2.Canny((img_edge*255).astype(np.uint8), 50, 150)
            edges = edges.astype(np.float32)/255.0

            # Flatten and histogram
            pixels = img_small.flatten()
            edges_flat = edges.flatten()
            hist = cv2.calcHist([img_small], [0], None, [32], [0,1]).flatten()
            hist /= hist.sum()

            vec = np.concatenate([pixels, edges_flat, hist])

            # Write row: first column = image stem, rest = feature vector
            writer.writerow([img_path.stem] + vec.tolist())

    print(f"Features for {rat_name} saved to {features_csv}")