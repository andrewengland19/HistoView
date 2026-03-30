import cv2
import numpy as np
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------

HOME = Path.home()
WORK = HOME / "SectionSorter"

INPUT = WORK / "input_dapi"
OUTPUT = WORK / "processed"

OUTPUT.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Settings
# -----------------------------

TARGET_SIZE = 256
BORDER = 10
THRESH = 10


# -----------------------------
# Processing
# -----------------------------

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
            print("  warning: image not read", flush=True)
            continue

        # -----------------------------
        # Initial tissue mask
        # -----------------------------

        _, mask = cv2.threshold(img, THRESH, 255, cv2.THRESH_BINARY)

        coords = np.column_stack(np.where(mask > 0))

        if coords.size == 0:
            print("  warning: no tissue detected", flush=True)
            continue

        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)

        cropped = img[y0:y1, x0:x1]

        # -----------------------------
        # Orientation normalization
        # -----------------------------

        pts = np.column_stack(np.where(cropped > THRESH))

        if len(pts) < 10:
            print("  warning: insufficient tissue points", flush=True)
            continue

        mean, eigenvectors = cv2.PCACompute(pts.astype(np.float32), mean=None)

        angle = np.arctan2(eigenvectors[0,1], eigenvectors[0,0])
        angle = np.degrees(angle)

        h, w = cropped.shape
        center = (w//2, h//2)

        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)

        rotated = cv2.warpAffine(cropped, rot_mat, (w, h))

        # -----------------------------
        # Crop again after rotation
        # -----------------------------

        _, mask = cv2.threshold(rotated, THRESH, 255, cv2.THRESH_BINARY)

        coords = np.column_stack(np.where(mask > 0))

        if coords.size == 0:
            print("  warning: no tissue after rotation", flush=True)
            continue

        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)

        h, w = rotated.shape

        y0 = max(0, y0 - BORDER)
        x0 = max(0, x0 - BORDER)
        y1 = min(h, y1 + BORDER)
        x1 = min(w, x1 + BORDER)

        cropped2 = rotated[y0:y1, x0:x1]

        if cropped2.size == 0:
            print("  warning: empty crop", flush=True)
            continue

        # -----------------------------
        # Contrast normalization
        # -----------------------------

        p1, p99 = np.percentile(cropped2, (1, 99))

        if p99 - p1 > 0:
            norm = np.clip((cropped2 - p1) / (p99 - p1), 0, 1)
        else:
            norm = cropped2 / 255.0

        norm = (norm * 255).astype(np.uint8)

        # -----------------------------
        # Resize to standard size
        # -----------------------------

        resized = cv2.resize(norm, (TARGET_SIZE, TARGET_SIZE))

        out_path = out_rat / img_path.name.replace(".jpg", ".png")

        cv2.imwrite(str(out_path), resized)

print("\nPreprocessing complete.")