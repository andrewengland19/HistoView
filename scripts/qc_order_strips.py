from pathlib import Path
import pandas as pd
from PIL import Image

HOME = Path.home()
PNG_ROOT = HOME / "Microscopy/Cohort1_TPH2/PNG"  # folder with downsampled PNGs
ORDER_ROOT = HOME / "Microscopy/Cohort1_TPH2/features/sorting"  # predicted_order CSVs
OUT = HOME / "Microscopy/Cohort1_TPH2/QC_strips"
OUT.mkdir(parents=True, exist_ok=True)

# Loop over each rat's predicted order CSV
for order_csv in ORDER_ROOT.glob("predicted_order_*.csv"):
    rat_name = order_csv.stem.replace("predicted_order_", "")
    print(f"Generating QC strip for {rat_name}")

    df_order = pd.read_csv(order_csv)
    # Filter CH1 images
    df_order = df_order[df_order["image"].str.contains("_CH1", case=False)]

    png_dir = PNG_ROOT / rat_name / "CH1"
    if not png_dir.exists():
        print(f"  Warning: {png_dir} does not exist, skipping")
        continue

    # Build PNG file list in order
    png_files = []
    for fname in df_order["image"].values:
        png_path = png_dir / f"{Path(fname).stem}.png"
        if png_path.exists():
            png_files.append(png_path)
        else:
            print(f"  Warning: {png_path.name} not found")

    if not png_files:
        print(f"  No PNGs found for {rat_name}, skipping")
        continue

    # Open and resize all images to uniform height
    images = []
    target_height = 128
    for f in png_files:
        im = Image.open(f)
        w, h = im.size
        new_w = int(w * target_height / h)
        images.append(im.resize((new_w, target_height)))

    # Create horizontal QC strip
    total_width = sum(im.width for im in images)
    qc_strip = Image.new("L", (total_width, target_height))
    x_offset = 0
    for im in images:
        qc_strip.paste(im, (x_offset, 0))
        x_offset += im.width

    out_path = OUT / f"{rat_name}_DAPI_QC_strip.png"
    qc_strip.save(out_path)
    print(f"  Saved QC strip: {out_path}")