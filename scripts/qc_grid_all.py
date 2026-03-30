from pathlib import Path
import pandas as pd
from PIL import Image

# === CONFIG PATHS ===
HOME = Path.home()
PNG_ROOT = HOME / "Microscopy/Cohort1_TPH2/PNG"            # downsampled PNGs
SORT_ROOT = HOME / "Microscopy/Cohort1_TPH2/features/sorting"  # predicted_order CSVs
OUT = HOME / "Microscopy/Cohort1_TPH2/QC_grids"
OUT.mkdir(parents=True, exist_ok=True)

CHANNELS = ["CH1", "CH2", "CH3", "CH4"]  # DAPI, GFP, RFP, Cy5
TARGET_HEIGHT = 128

# Loop over each rat
for order_csv in SORT_ROOT.glob("predicted_order_*.csv"):
    rat_name = order_csv.stem.replace("predicted_order_", "")
    print(f"Generating multi-channel QC grid for {rat_name}")

    # Load predicted order CSV (CH1 only)
    df_order = pd.read_csv(order_csv)

    channel_rows = []

    for ch in CHANNELS:
        row_images = []

        # Build filenames for this channel
        for ch1_fname in df_order["image"].values:
            fname = ch1_fname.replace("_CH1", f"_{ch}")  # replace CH1 with current channel
            png_path = PNG_ROOT / rat_name / ch / f"{fname}.png"

            if not png_path.exists():
                print(f"  Warning: {png_path.name} not found, skipping")
                continue

            # Open PNG and resize to uniform height
            im = Image.open(png_path)
            w, h = im.size
            new_w = int(w * TARGET_HEIGHT / h)
            im_resized = im.resize((new_w, TARGET_HEIGHT))
            row_images.append(im_resized)

        if not row_images:
            print(f"  No images found for {ch}, skipping row")
            continue

        # Concatenate images horizontally for this channel row
        total_width = sum(im.width for im in row_images)
        row_img = Image.new("L", (total_width, TARGET_HEIGHT))
        x_offset = 0
        for im in row_images:
            row_img.paste(im, (x_offset, 0))
            x_offset += im.width

        channel_rows.append(row_img)

    # Combine all channel rows vertically
    if not channel_rows:
        print(f"No channel rows found for {rat_name}, skipping QC grid")
        continue

    total_width = max(r.width for r in channel_rows)
    total_height = sum(r.height for r in channel_rows)

    qc_grid = Image.new("L", (total_width, total_height))
    y_offset = 0
    for r in channel_rows:
        qc_grid.paste(r, (0, y_offset))
        y_offset += r.height

    # Save QC grid
    out_path = OUT / f"{rat_name}_QC_grid.png"
    qc_grid.save(out_path)
    print(f"  Saved multi-channel QC grid: {out_path}")