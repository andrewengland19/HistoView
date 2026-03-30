import cv2
import numpy as np
from pathlib import Path

HOME = Path.home()
WORK = HOME / "SectionSorter"

INPUT = WORK / "input_oriented"
OUTPUT = WORK / "processed"

OUTPUT.mkdir(exist_ok=True)

TARGET_SIZE = 256

for rat_dir in INPUT.iterdir():

    if not rat_dir.is_dir():
        continue

    out_rat = OUTPUT / rat_dir.name
    out_rat.mkdir(exist_ok=True)

    for img_path in rat_dir.glob("*.jpg"):

        img = cv2.imread(str(img_path))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # threshold to detect tissue
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

        coords = np.column_stack(np.where(mask > 0))

        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)

        cropped = gray[y0:y1, x0:x1]

        resized = cv2.resize(cropped, (TARGET_SIZE, TARGET_SIZE))

        out_path = out_rat / img_path.name.replace(".jpg", ".png")

        cv2.imwrite(str(out_path), resized)

        print(f"processed {img_path.name}", flush=True)