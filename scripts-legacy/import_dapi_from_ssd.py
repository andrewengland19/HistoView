import pandas as pd
import cv2
import numpy as np
from pathlib import Path

# --- paths ---
DATA = Path("D:/Cohort1_5HT")                 # external SSD dataset root
MANIFEST = DATA / "05_metadata/overlay_manifest.csv"

HOME = Path.home()
WORK = HOME / "SectionSorter"
OUT = WORK / "input_dapi"
OUT.mkdir(parents=True, exist_ok=True)

# --- settings ---
DOWNSAMPLE = 0.25          # 4x smaller (change if you want)
JPEG_QUALITY = 90

df = pd.read_csv(MANIFEST)
total = len(df)

for k, r in df.iterrows():
    rat = r["rat"]
    sec = r["section_label"]
    rot = int(r.get("rotation_deg", 0))
    mirror = int(r.get("mirror_lr", 0))

    src = DATA / "02_channels" / rat / "DAPI" / f"{rat}_{sec}_DAPI.tif"
    dst_dir = OUT / rat
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"{rat}_{sec}.jpg"

    print(f"[{k+1}/{total}] import DAPI {rat} {sec}", flush=True)

    img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"  missing: {src}", flush=True)
        continue

    # convert to 8-bit for sorter (safe for morphology)
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # rotation (clockwise degrees to canonical)
    if rot == 90:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif rot == 180:
        img = cv2.rotate(img, cv2.ROTATE_180)
    elif rot == 270:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # mirror L/R
    if mirror == 1:
        img = cv2.flip(img, 1)

    # downsample
    img = cv2.resize(img, None, fx=DOWNSAMPLE, fy=DOWNSAMPLE, interpolation=cv2.INTER_AREA)

    cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

print("DAPI import complete.")