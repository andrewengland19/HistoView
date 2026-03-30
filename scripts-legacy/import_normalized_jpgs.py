import pandas as pd
import cv2
from pathlib import Path

DATA = Path("D:/Cohort1_5HT")
HOME = Path.home()
WORK = HOME / "SectionSorter"

INPUT = DATA / "03_overlays_QC"
OUTPUT = WORK / "input_oriented"
MANIFEST = DATA / "05_metadata/overlay_manifest.csv"

OUTPUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(MANIFEST)

total = len(df)

for i, r in df.iterrows():

    rat = r["rat"]
    sec = r["section_label"]
    rot = int(r["rotation_deg"])
    mirror = int(r["mirror_lr"])

    img_path = INPUT / rat / f"{rat}_{sec}_overlay.tif"

    print(f"[{i+1}/{total}] Processing {rat} {sec}", flush=True)

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  missing: {img_path}", flush=True)
        continue

    # rotation
    if rot == 90:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif rot == 180:
        img = cv2.rotate(img, cv2.ROTATE_180)
    elif rot == 270:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # mirror
    if mirror == 1:
        img = cv2.flip(img, 1)

    # downsample
    img = cv2.resize(img, None, fx=0.25, fy=0.25)

    out_dir = OUTPUT / rat
    out_dir.mkdir(exist_ok=True)

    out_path = out_dir / f"{rat}_{sec}.jpg"

    cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 90])

print("Orientation + compression complete.")