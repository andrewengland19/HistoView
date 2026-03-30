from pathlib import Path
import pandas as pd
import cv2
import numpy as np

HOME = Path.home()
WORK = HOME / "SectionSorter"

ORDER = WORK / "sorting/predicted_order.csv"

CHANNEL_ROOT = Path("D:/Cohort1_5HT/02_channels")

OUT = WORK / "diagnostics/qc_grids"
OUT.mkdir(exist_ok=True)

RAT = "WT2"   # change if needed

CHANNELS = ["DAPI","5HT","mCherry","NeuN"]

THUMB_H = 120
GAP = 6


def parse_sec(name):

    s = name.split("_sec")[1]
    digits = "".join([c for c in s if c.isdigit()])

    return int(digits)


def load_img(path):

    img = cv2.imread(str(path),cv2.IMREAD_UNCHANGED)

    if img is None:
        return None

    if img.ndim == 3:
        img = img[:,:,np.argmax([img[:,:,i].mean() for i in range(img.shape[2])])]

    img = img.astype(np.float32)

    p1,p99 = np.percentile(img,(1,99))

    if p99 <= p1:
        p99 = p1 + 1

    img = np.clip((img-p1)/(p99-p1),0,1)

    img = (img*255).astype(np.uint8)

    h,w = img.shape

    scale = THUMB_H / h

    img = cv2.resize(img,(int(w*scale),THUMB_H))

    img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

    return img


order = pd.read_csv(ORDER)

rat_df = order[order["rat"]==RAT].sort_values("predicted_order")

rows = {ch:[] for ch in CHANNELS}

for row in rat_df.itertuples():

    sec = parse_sec(row.image)

    for ch in CHANNELS:

        src = CHANNEL_ROOT / RAT / ch / f"{RAT}_sec{sec:02d}_{ch}.tif"

        img = load_img(src)

        if img is None:
            continue

        rows[ch].append(img)


grid_rows = []

for ch in CHANNELS:

    imgs = rows[ch]

    if not imgs:
        continue

    row_img = imgs[0]

    for im in imgs[1:]:

        gap = np.zeros((row_img.shape[0],GAP,3),dtype=np.uint8)

        row_img = np.hstack([row_img,gap,im])

    grid_rows.append(row_img)


grid = grid_rows[0]

for r in grid_rows[1:]:

    gap = np.zeros((10,grid.shape[1],3),dtype=np.uint8)

    if r.shape[1] < grid.shape[1]:
        pad = np.zeros((r.shape[0],grid.shape[1]-r.shape[1],3),dtype=np.uint8)
        r = np.hstack([r,pad])

    grid = np.vstack([grid,gap,r])


out = OUT / f"qc_grid_{RAT}.png"

cv2.imwrite(str(out),grid)

print("QC grid written:",out)