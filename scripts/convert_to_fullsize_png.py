from pathlib import Path
import tifffile as tif
import cv2
import numpy as np

# Config paths
HOME = Path.home()
INPUT = HOME / "Microscopy/Cohort1_TPH2/dataset"       # normalized TIFFs
PNG_OUTPUT = HOME / "Microscopy/Cohort1_TPH2/PNG_fullsize"
PNG_OUTPUT.mkdir(parents=True, exist_ok=True)

# Channels to process
CHANNELS = ["CH1", "CH2", "CH3", "CH4"] # all channels - change if needed

# Downsample height
TARGET_HEIGHT = 128

# Loop over rats
for rat_dir in sorted(INPUT.iterdir()):
    if not rat_dir.is_dir():
        continue
    rat_name = rat_dir.name
    print(f"\nProcessing rat: {rat_name}")

    for ch in CHANNELS:
        ch_dir = rat_dir / ch
        if not ch_dir.exists():
            print(f"  Warning: {ch_dir} does not exist, skipping")
            continue

        png_dir = PNG_OUTPUT / rat_name / ch
        png_dir.mkdir(parents=True, exist_ok=True)

        images = sorted(ch_dir.glob("*.tif"))
        for i, img_path in enumerate(images, 1):
            print(f"[{i}/{len(images)}] {img_path.name}")

            # Load TIFF
            img = tif.imread(str(img_path))

            # Multi-page TIFF? pick first page (assume channel-specific single page)
            if img.ndim == 3:
                if img.shape[0] > 1:
                    img = img[0]  # usually single page per channel
                elif img.shape[2] == 3:
                    img = img[:, :, 0]  # take first channel if RGB

            # # Downsample
            # h, w = img.shape[:2]
            # new_w = int(w * TARGET_HEIGHT / h)
            # img_small = cv2.resize(img, (new_w, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)

            # Save PNG
            png_path = png_dir / (img_path.stem + ".png")
            cv2.imwrite(
                str(png_path),
                cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            )