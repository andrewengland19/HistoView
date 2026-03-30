from pathlib import Path
import pandas as pd

HOME = Path.home()
WORK = HOME / "SectionSorter"

ORDER = WORK / "sorting/predicted_order.csv"

META = WORK / "metadata"
META.mkdir(exist_ok=True)

MANIFEST = META / "orientation_manifest.csv"

order = pd.read_csv(ORDER)

rows = []

for r in order.itertuples():

    rows.append({
        "rat": r.rat,
        "section_label": r.image,
        "rotation_deg": 0,
        "mirror_lr": 0
    })

manifest = pd.DataFrame(rows)

manifest.to_csv(MANIFEST,index=False)

print("Orientation manifest created:")
print(MANIFEST)