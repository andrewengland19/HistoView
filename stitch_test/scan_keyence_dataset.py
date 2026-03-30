import os
import re
import csv
from pathlib import Path
from collections import defaultdict

print("\n===================================")
print("Keyence Dataset Preflight Scanner")
print("===================================\n")

# -------------------------------------------------------
# Prompt helper
# -------------------------------------------------------

def ask(question):
    while True:
        ans = input(question + " (y/n): ").strip().lower()
        if ans in ["y", "yes"]:
            return True
        if ans in ["n", "no"]:
            return False

# -------------------------------------------------------
# Ask for dataset path
# -------------------------------------------------------

root = input("Enter dataset root directory (example: D:/Cohort1_TPH2): ").strip()

root_path = Path(root)

if not root_path.exists():
    print("\nERROR: Directory does not exist.")
    exit()

print(f"\nDataset root detected:\n{root_path}\n")

# -------------------------------------------------------
# Step prompts
# -------------------------------------------------------

scan_step = ask("Run dataset scan?")
validate_step = ask("Run tile validation?")
tilemap_step = ask("Generate tile map?")
summary_step = ask("Generate dataset summary?")

print("\nStarting pipeline...\n")

# -------------------------------------------------------
# Filename pattern for Keyence tiles
# -------------------------------------------------------

pattern = re.compile(r"XY(\d+).*_(\d+)_CH(\d+)\.tif$", re.IGNORECASE)

inventory = []
tile_map = []
validation_errors = []

total_files = 0
total_xy = 0

# -------------------------------------------------------
# Discover XY folders
# -------------------------------------------------------

xy_folders = []

if scan_step:

    print("Scanning SET folders...\n")

    set_dirs = [d for d in root_path.iterdir() if d.is_dir()]

    for s in set_dirs:

        xy_dirs = [
            d for d in s.iterdir()
            if d.is_dir() and "XY" in d.name.upper()
        ]

        for xy in xy_dirs:
            xy_folders.append((s.name, xy))

    total_xy = len(xy_folders)

    print(f"Detected {len(set_dirs)} SET folders")
    print(f"Detected {total_xy} XY scans\n")

# -------------------------------------------------------
# Main processing loop
# -------------------------------------------------------

for i, (set_name, xy_path) in enumerate(xy_folders, start=1):

    print(f"[{i}/{total_xy}] Processing {set_name}/{xy_path.name}")

    files = list(xy_path.glob("*.tif"))

    if not files:
        validation_errors.append(f"{set_name}/{xy_path.name} : no TIFF files")
        continue

    channel_counts = defaultdict(int)
    tile_sets = defaultdict(set)

    for f in files:

        total_files += 1

        m = pattern.search(f.name)

        if not m:
            continue

        xy = f"XY{m.group(1)}"
        tile = m.group(2)
        ch = f"CH{m.group(3)}"

        channel_counts[ch] += 1
        tile_sets[ch].add(tile)

        if tilemap_step:
            tile_map.append([
                set_name,
                xy,
                tile,
                ch,
                str(f)
            ])

    channels = len(channel_counts)
    tiles_per_channel = max(channel_counts.values())

    inventory.append([
        set_name,
        xy_path.name,
        channels,
        tiles_per_channel
    ])

    if validate_step:

        tile_lengths = [len(v) for v in tile_sets.values()]

        if len(set(tile_lengths)) != 1:
            validation_errors.append(
                f"{set_name}/{xy_path.name} inconsistent tiles: {tile_lengths}"
            )

# -------------------------------------------------------
# Write inventory
# -------------------------------------------------------

if scan_step:

    print("\nWriting dataset_inventory.csv")

    with open("dataset_inventory.csv", "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "set",
            "xy",
            "channels",
            "tiles_per_channel"
        ])

        writer.writerows(inventory)

# -------------------------------------------------------
# Write tile map
# -------------------------------------------------------

if tilemap_step:

    print("Writing tile_map.csv")

    with open("tile_map.csv", "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "set",
            "xy",
            "tile",
            "channel",
            "path"
        ])

        writer.writerows(tile_map)

# -------------------------------------------------------
# Write validation report
# -------------------------------------------------------

if validate_step:

    print("Writing validation_report.txt")

    with open("validation_report.txt", "w") as f:

        if not validation_errors:
            f.write("No validation errors detected.\n")

        else:

            f.write("Validation issues:\n\n")

            for err in validation_errors:
                f.write(err + "\n")

# -------------------------------------------------------
# Dataset summary
# -------------------------------------------------------

if summary_step:

    print("\nCalculating dataset size (this may take a minute)...\n")

    total_size = 0

    for p in root_path.rglob("*"):
        if p.is_file():
            total_size += p.stat().st_size

    gb = total_size / (1024 ** 3)

    summary = f"""
Dataset Summary
---------------

Root: {root_path}

Total XY scans: {total_xy}
Total TIFF files: {total_files}

Dataset size: {gb:.2f} GB
"""

    print(summary)

    with open("dataset_summary.txt", "w") as f:
        f.write(summary)

print("\nPreflight completed successfully.\n")