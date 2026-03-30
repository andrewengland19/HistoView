from pathlib import Path
import pandas as pd
import numpy as np
import cv2
import shutil

HOME = Path.home()
WORK = HOME / "SectionSorter"

ORDER = WORK / "sorting/predicted_order.csv"
MANIFEST = WORK / "metadata/orientation_manifest.csv"

CHANNEL_ROOT = Path("D:/Cohort1_5HT/02_channels")
OVERLAY_ROOT = Path("D:/Cohort1_5HT/01_overlays")

QC_SRC = WORK / "diagnostics/qc_grids"
DIAG_SRC = WORK / "diagnostics"

OUT = WORK / "PI_review_package"

RAT_OUT = OUT / "Rats"
QC_OUT = OUT / "QC"
STRIP_OUT = OUT / "DAPI_strips"
DIAG_OUT = OUT / "Diagnostics"

for d in [RAT_OUT, QC_OUT, STRIP_OUT, DIAG_OUT]:
    d.mkdir(parents=True, exist_ok=True)


CHANNELS = ["DAPI","5HT","mCherry","NeuN"]


def parse_sec(name):
    s = name.split("_sec")[1]
    digits = "".join([c for c in s if c.isdigit()])
    return int(digits)


def apply_transform(img, rot, mirror):

    if rot == 90:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    elif rot == 180:
        img = cv2.rotate(img, cv2.ROTATE_180)

    elif rot == 270:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    if mirror == 1:
        img = cv2.flip(img,1)

    return img


def downsample(img):

    h,w = img.shape[:2]

    target = 1200

    scale = target / max(h,w)

    if scale < 1:
        img = cv2.resize(img,(int(w*scale),int(h*scale)))

    return img


def make_dapi_strip(images):

    target_h = 200   # height for strip thumbnails

    resized = []

    for img in images:

        h, w = img.shape[:2]

        scale = target_h / h

        new_w = int(w * scale)

        im = cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)

        resized.append(im)

    strip = resized[0]

    for im in resized[1:]:

        gap = np.zeros((target_h, 5), dtype=np.uint8)

        strip = np.hstack([strip, gap, im])

    return strip

order = pd.read_csv(ORDER)
manifest = pd.read_csv(MANIFEST)

for rat in order["rat"].unique():

    rat_df = order[order["rat"]==rat].sort_values("predicted_order")

    rat_dir = RAT_OUT / rat
    rat_dir.mkdir(exist_ok=True)

    dapi_strip_imgs = []

    for i,row in enumerate(rat_df.itertuples(),start=1):

        sec = parse_sec(row.image)

        idx = f"{i:02d}"

        mrow = manifest[(manifest["rat"]==rat) &
                        (manifest["section_label"]==row.image)]

        if len(mrow)>0:

            rot = int(mrow.iloc[0]["rotation_deg"])
            mir = int(mrow.iloc[0]["mirror_lr"])

        else:

            rot = 0
            mir = 0

        for ch in CHANNELS:

            src = CHANNEL_ROOT / rat / ch / f"{rat}_sec{sec:02d}_{ch}.tif"

            if not src.exists():
                continue

            print(src)

            img = cv2.imread(str(src),cv2.IMREAD_UNCHANGED)

            if img.ndim==3:
                img = img[:,:,np.argmax([img[:,:,i].mean() for i in range(img.shape[2])])]

            img = img.astype(np.float32)

            p1,p99 = np.percentile(img,(1,99))
            if p99<=p1:
                p99=p1+1

            img = np.clip((img-p1)/(p99-p1),0,1)
            img = (img*255).astype(np.uint8)

            img = apply_transform(img,rot,mir)
            img = downsample(img)

            if ch=="DAPI":
                dapi_strip_imgs.append(img)

            out = rat_dir / f"{rat}-{idx}_{ch}.jpg"

            cv2.imwrite(str(out),img,[int(cv2.IMWRITE_JPEG_QUALITY),90])

        overlay = OVERLAY_ROOT / rat / f"{rat}_sec{sec:02d}_overlay.tif"

        if overlay.exists():

            img = cv2.imread(str(overlay))
            img = apply_transform(img,rot,mir)
            img = downsample(img)

            out = rat_dir / f"{rat}-{idx}_overlay.jpg"

            cv2.imwrite(str(out),img,[int(cv2.IMWRITE_JPEG_QUALITY),90])


    if dapi_strip_imgs:

        strip = make_dapi_strip(dapi_strip_imgs)

        cv2.imwrite(str(STRIP_OUT / f"{rat}_DAPI_strip.png"),strip)


# copy QC grids
for f in QC_SRC.glob("*.png"):
    shutil.copy(f,QC_OUT / f.name)


# copy diagnostics
for f in DIAG_SRC.glob("*.png"):
    shutil.copy(f,DIAG_OUT / f.name)

for f in DIAG_SRC.glob("*.csv"):
    shutil.copy(f,DIAG_OUT / f.name)


shutil.copy(ORDER,OUT / "predicted_order.csv")

print("Review package rebuilt successfully.")