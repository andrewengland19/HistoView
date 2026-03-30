import os
import csv
import re
import xml.etree.ElementTree as ET
from pathlib import Path

print("\n====================================")
print("KEYENCE STITCH MANIFEST GENERATOR")
print("====================================\n")

root = input("Enter dataset root directory (example: D:/Cohort1_TPH2): ").strip()
root = Path(root)

if not root.exists():
    print("ERROR: directory not found")
    exit()

print("\nScanning dataset...\n")

# --------------------------------------------------
# STEP 1: find all XY folders + .gci files
# --------------------------------------------------

xy_entries = []

for set_dir in sorted(root.iterdir()):

    if not set_dir.is_dir():
        continue

    if not set_dir.name.upper().startswith("SET"):
        continue

    for xy_dir in sorted(set_dir.iterdir()):

        if not xy_dir.is_dir():
            continue

        if not xy_dir.name.upper().startswith("XY"):
            continue

        gci_files = list(xy_dir.glob("*.gci"))

        if len(gci_files) == 0:
            print(f"WARNING: {set_dir.name}/{xy_dir.name} has no .gci file")
            gci = None
        else:
            gci = gci_files[0]

        xy_entries.append({
            "set": set_dir.name,
            "xy": xy_dir.name,
            "path": xy_dir,
            "gci": gci
        })

print(f"\nFound {len(xy_entries)} XY folders\n")

print("Preview:\n")

for entry in xy_entries[:10]:
    print(
        f"{entry['set']}/{entry['xy']}  "
        f"GCI: {entry['gci'].name if entry['gci'] else 'NONE'}"
    )

print("\n------------------------------------")
input("Verify the above looks correct. Press ENTER to continue...")

# --------------------------------------------------
# STEP 2: parse XML metadata
# --------------------------------------------------

print("\nExtracting grid metadata...\n")

manifest_rows = []

for i, entry in enumerate(xy_entries):

    set_name = entry["set"]
    xy_name = entry["xy"]
    xy_path = entry["path"]

    print(f"[{i+1}/{len(xy_entries)}] {set_name}/{xy_name}")

    import zipfile

# --------------------------------------------------
# ensure .gci is extracted
# --------------------------------------------------

    extracted_dir = None

    for d in xy_path.iterdir():
        if d.is_dir() and "extracted" in d.name.lower():
            extracted_dir = d
            break

    if extracted_dir is None:

        if entry["gci"] is None:
            print("   ERROR: no .gci file to extract")
            continue

        gci_file = entry["gci"]

        extracted_dir = xy_path / f"{gci_file.stem}_extracted"

        print("   Extracting metadata from:", gci_file.name)

        try:

            with zipfile.ZipFile(gci_file, 'r') as z:
                z.extractall(extracted_dir)

        except Exception as e:

            print("   ERROR extracting .gci:", e)
            continue

    xml_path = (
        extracted_dir /
        "GroupFileProperty" /
        "ImageJoint" /
        "properties.xml"
    )

    if not xml_path.exists():
        print("   ERROR: properties.xml not found")
        continue

    try:

        tree = ET.parse(xml_path)
        root_xml = tree.getroot()

        rows = None
        cols = None

        for elem in root_xml.iter():

            if elem.tag == "Row":
                rows = int(elem.text)

            if elem.tag == "Column":
                cols = int(elem.text)

        if rows is None or cols is None:
            print("   ERROR: row/column metadata missing")
            continue

    except Exception as e:
        print("   ERROR parsing XML:", e)
        continue

    # --------------------------------------------------
    # count tiles
    # --------------------------------------------------

    tif_count = len(list(xy_path.glob("*.tif")))

    manifest_rows.append([
        set_name,
        xy_name,
        rows,
        cols,
        rows * cols,
        tif_count
    ])

    if rows * cols != tif_count:
        print(
            f"   WARNING: expected {rows*cols} tiles but found {tif_count}"
        )

# --------------------------------------------------
# STEP 3: save manifest
# --------------------------------------------------

out_path = root / "stitch_manifest.csv"

print("\nWriting manifest:", out_path)

with open(out_path, "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "set",
        "xy",
        "rows",
        "cols",
        "expected_tiles",
        "found_tiles"
    ])

    writer.writerows(manifest_rows)

print("\nDone.\n")

print("Manifest preview:\n")

for r in manifest_rows[:10]:
    print(r)

print("\n====================================")
print("Manifest generation complete")
print("====================================\n")