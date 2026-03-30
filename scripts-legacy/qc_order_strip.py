import pandas as pd
import cv2
import numpy as np
from pathlib import Path

HOME = Path.home()
WORK = HOME / "SectionSorter"

INPUT = WORK / "processed"
ORDER = WORK / "sorting/predicted_order.csv"
OUT = WORK / "diagnostics"

OUT.mkdir(exist_ok=True)

order = pd.read_csv(ORDER)

for rat in order["rat"].unique():

    print(f"QC strip: {rat}", flush=True)

    rat_df = order[order["rat"] == rat].sort_values("predicted_order")

    imgs = []

    for img_name in rat_df["image"]:

        path = INPUT / rat / f"{img_name}.png"
        img = cv2.imread(str(path))

        img = cv2.resize(img, (120,120))

        imgs.append(img)

    strip = np.hstack(imgs)

    cv2.imwrite(str(OUT / f"order_strip_{rat}.png"), strip)

print("QC strips generated.")