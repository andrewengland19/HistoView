from pathlib import Path
import cv2
import numpy as np
import pandas as pd

HOME = Path.home()
WORK = HOME / "SectionSorter"

ORDER = WORK / "sorting/predicted_order.csv"
QC_GRIDS = WORK / "diagnostics/qc_grids"

DATA = Path("D:/Cohort1_5HT/02_channels")

OUT = WORK / "PI_review_package"
RAT_OUT = OUT / "Rats"
QC_OUT = OUT / "QC"
DIAG_OUT = OUT / "Diagnostics"

RAT_OUT.mkdir(parents=True, exist_ok=True)
QC_OUT.mkdir(parents=True, exist_ok=True)
DIAG_OUT.mkdir(parents=True, exist_ok=True)

DAPI_STRIP_OUT = OUT / "DAPI_strips"
DAPI_STRIP_OUT.mkdir(parents=True, exist_ok=True)


CHANNELS = {
    "DAPI": "blue",
    "5HT": "green",
    "mCherry": "red",
    "NeuN": "purple"
}

def generate_dapi_strip(rat, rat_df, out_dir):

    thumbs = []

    for row in rat_df.itertuples():

        sec = parse_section_number(row.image)

        src = raw_path(rat, sec, "DAPI")

        if not src.exists():
            continue

        img = read_grayscale(src)

        img = downsample(img)

        thumbs.append(img)

    if not thumbs:
        return

    strip = thumbs[0]

    for img in thumbs[1:]:

        gap = np.zeros((strip.shape[0], 5), dtype=np.uint8)

        strip = np.hstack([strip, gap, img])

    out = out_dir / f"{rat}_DAPI_strip.png"

    cv2.imwrite(str(out), strip)


def parse_section_number(image):

    sec = image.split("_sec")[1]
    digits = "".join([c for c in sec if c.isdigit()])

    return int(digits)


def raw_path(rat, section, ch):

    return DATA / rat / ch / f"{rat}_sec{section:02d}_{ch}.tif"


def read_grayscale(path, channel=None):

    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if img is None:
        return None

    # handle multi-channel TIFFs correctly
    if img.ndim == 3:

        means = [img[:,:,i].mean() for i in range(img.shape[2])]
        best = int(np.argmax(means))

        img = img[:,:,best]

    img = img.astype(np.float32)

    if channel == "5HT":

        vmin = 3000
        vmax = 36000

        if vmax <= vmin:
            vmax = vmin + 1

        img = np.clip((img - vmin) / (vmax - vmin), 0, 1)
        img = img ** 0.75

    elif channel == "mCherry":

        vmin = 0
        vmax = 60000

        if vmax <= vmin:
            vmax = vmin + 1

        img = np.clip((img - vmin) / (vmax - vmin), 0, 1)

    else:

        p1, p99 = np.percentile(img, (1, 99))

        if p99 <= p1:
            p99 = p1 + 1

        img = np.clip((img - p1) / (p99 - p1), 0, 1)

    img = (img * 255).astype(np.uint8)

    return img

def colorize(img, color):

    h,w = img.shape

    out = np.zeros((h,w,3), dtype=np.uint8)

    if color == "blue":
        out[:,:,0] = img

    elif color == "green":
        out[:,:,1] = img

    elif color == "red":
        out[:,:,2] = img

    elif color == "purple":
        out[:,:,0] = img
        out[:,:,2] = img

    return out


def downsample(img):

    h,w = img.shape[:2]

    target = 1200

    scale = target / max(h,w)

    if scale < 1:
        img = cv2.resize(img, (int(w*scale), int(h*scale)))

    return img


order = pd.read_csv(ORDER)

summary = []

for rat in order["rat"].unique():

    rat_dir = RAT_OUT / rat
    rat_dir.mkdir(exist_ok=True)

    rat_df = order[order["rat"] == rat].sort_values("predicted_order")

    for i,row in enumerate(rat_df.itertuples(), start=1):

        sec = parse_section_number(row.image)

        idx = f"{i:02d}"

        for ch,color in CHANNELS.items():

            src = raw_path(rat, sec, ch)

            if not src.exists():
                continue

            print(src)

            img = read_grayscale(src, ch)

            img = colorize(img, color)

            img = downsample(img)

            outname = f"{rat}-{idx}_{ch}.jpg"

            cv2.imwrite(str(rat_dir / outname), img,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        # overlay if exists
        overlay = DATA / rat / "overlay" / f"{rat}_sec{sec:02d}_overlay.tif"

        if overlay.exists():

            img = cv2.imread(str(overlay))

            img = downsample(img)

            outname = f"{rat}-{idx}_overlay.jpg"

            cv2.imwrite(str(rat_dir / outname), img,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    summary.append((rat, len(rat_df)))



# copy QC grids
for grid in QC_GRIDS.glob("qc_grid_*.png"):

    dest = QC_OUT / grid.name

    img = cv2.imread(str(grid))

    cv2.imwrite(str(dest), img)



# diagnostics
df = pd.DataFrame(summary, columns=["rat","section_count"])

df.to_csv(DIAG_OUT / "section_counts.csv", index=False)



# README
readme = OUT / "README.txt"

readme.write_text(
"""
PI Review Package

Folders:
Rats/     -> compressed channel images in anatomical order
QC/       -> QC grid strips for each rat
Diagnostics/ -> section counts

Channels:
DAPI = blue
5HT = green
mCherry = red
NeuN = purple

Sections numbered according to automated anatomical ordering.
"""
)

print("PI review package created")