import cv2
import numpy as np
from pathlib import Path

HOME = Path.home()
WORK = HOME / "SectionSorter"

INPUT = WORK / "input_dapi"
OUTPUT = WORK / "processed"

OUTPUT.mkdir(parents=True, exist_ok=True)

TARGET_SIZE = 256

for rat_dir in INPUT.iterdir():

    if not rat_dir.is_dir():
        continue

    out_rat = OUTPUT / rat_dir.name
    out_rat.mkdir(exist_ok=True)

    files = sorted(rat_dir.glob("*.jpg"))
    total = len(files)

    print(f"\nProcessing {rat_dir.name} ({total} images)", flush=True)

    for i, img_path in enumerate(files):

        print(f"[{i+1}/{total}] {img_path.name}", flush=True)

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

        if img is None:
            print("warning: image not read", flush=True)
            continue

        # -----------------------------
        # Contrast normalization
        # -----------------------------

        p1, p99 = np.percentile(img, (1, 99))

        if p99 - p1 > 0:
            norm = np.clip((img - p1) / (p99 - p1), 0, 1)
        else:
            norm = img / 255.0

        norm = (norm * 255).astype(np.uint8)

        # -----------------------------
        # Resize only
        # -----------------------------

        resized = cv2.resize(norm, (TARGET_SIZE, TARGET_SIZE))

        out_path = out_rat / img_path.name.replace(".jpg", ".png")

        cv2.imwrite(str(out_path), resized)

print("\nPreprocessing complete.")