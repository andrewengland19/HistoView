import pandas as pd
import re
import shutil
from pathlib import Path

# -----------------------------
# USER SETTINGS
# -----------------------------

DATA_ROOT = Path(r"D:\Cohort1_TPH2\raw_exports")
OUTPUT_ROOT = Path(r"D:\Cohort1_TPH2\dataset")
MANIFEST_PATH = Path(r"D:\Cohort1_TPH2\dataset_manifest.csv")

# -----------------------------
# LOAD MANIFEST
# -----------------------------

manifest = pd.read_csv(MANIFEST_PATH)

# normalize set names
manifest["set"] = manifest["set"].str.upper()

# strip XY prefix and convert to integers
manifest["xy_start"] = (
    manifest["xy_start"]
    .astype(str)
    .str.replace("XY", "", regex=False)
    .astype(int)
)

manifest["xy_end"] = (
    manifest["xy_end"]
    .astype(str)
    .str.replace("XY", "", regex=False)
    .astype(int)
)

# rat_id stays a string (WT2, KO1, etc.)
manifest["rat_id"] = manifest["rat_id"].astype(str)

def find_mapping(set_name, xy_number):

    for _, row in manifest.iterrows():

        if row["set"] != set_name:
            continue

        if row["xy_start"] <= xy_number <= row["xy_end"]:
            return row["rat_id"], row["region"]

    return None, None


# -----------------------------
# TILE FILENAME PARSER
# -----------------------------

tile_regex = re.compile(
    r"Image_XY(\d+)_(\d+)_CH(\d+)\.tif",
    re.IGNORECASE
)

# -----------------------------
# PROCESS DATASET
# -----------------------------

for set_dir in DATA_ROOT.iterdir():

    if not set_dir.is_dir():
        continue

    set_name = set_dir.name

    print(f"\nProcessing {set_name}")

    for xy_dir in sorted(set_dir.glob("XY*")):

        xy_match = re.search(r"XY(\d+)", xy_dir.name)

        if not xy_match:
            continue

        xy_number = int(xy_match.group(1))
        section_name = f"XY{xy_number:02d}"

        rat, region = find_mapping(set_name, xy_number)

        if rat is None:
            print(f"WARNING: no mapping for {set_name} {section_name}")
            continue

        print(f"  {set_name}/{section_name} → rat{rat} {region}")

        for tif in xy_dir.glob("*.tif"):

            m = tile_regex.search(tif.name)

            if not m:
                continue

            tile_index = int(m.group(2))
            channel = int(m.group(3))

            dest_dir = (
                OUTPUT_ROOT
                / f"rat{rat}"
                / region
                / section_name
                / f"CH{channel}"
            )

            dest_dir.mkdir(parents=True, exist_ok=True)

            new_name = (
                f"rat{rat}_{region}_{section_name}_"
                f"CH{channel}_tile{tile_index:04d}.tif"
            )

            dest_path = dest_dir / new_name

            if dest_path.exists():
                continue

            shutil.move(str(tif), dest_path)

print("\nDataset organization complete.")