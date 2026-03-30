import os
import re
import math
import numpy as np
import tifffile

INPUT_DIR = os.getcwd()
OUTPUT_DIR = os.path.join(INPUT_DIR, "stitched_test")

file_regex = re.compile(r"Image_(XY\d+)_(\d{5})_(CH\d)")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\nScanning tiles...\n")

tiles = {}

for f in sorted(os.listdir(INPUT_DIR)):

    if not f.lower().endswith(".tif"):
        continue

    m = file_regex.search(f)
    if not m:
        continue

    xy, tile_id, ch = m.groups()

    if xy != "XY01":
        continue

    tile_id = int(tile_id)

    tiles.setdefault(ch, {})
    tiles[ch][tile_id] = os.path.join(INPUT_DIR, f)

print("Channels:", list(tiles.keys()))

def infer_grid(n):

    factors = []

    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            factors.append((i, n//i))

    # choose grid closest to square
    return min(factors, key=lambda x: abs(x[0]-x[1]))

for ch in sorted(tiles):

    print(f"\nStitching {ch}")

    tile_dict = tiles[ch]

    tile_ids = sorted(tile_dict.keys())
    n_tiles = len(tile_ids)

    rows, cols = infer_grid(n_tiles)

    print(f"Detected grid: {rows} x {cols}")

    images = []

    for i, tile_id in enumerate(tile_ids):

        print(f"Loading tile {i+1}/{n_tiles}")

        with tifffile.TiffFile(tile_dict[tile_id]) as tif:
            arr = tif.asarray()

            if arr.ndim == 3:
                arr = arr[0]

        images.append(arr)

    tile_h, tile_w = images[0].shape[:2]

    canvas = np.zeros((rows*tile_h, cols*tile_w), dtype=images[0].dtype)

    print("Placing tiles...")

    idx = 0

    for r in range(rows):

        row_tiles = images[idx:idx+cols]

        # snake scan correction
        if r % 2 == 1:
            row_tiles = row_tiles[::-1]

        for c, img in enumerate(row_tiles):

            y0 = r*tile_h
            y1 = y0+tile_h

            x0 = c*tile_w
            x1 = x0+tile_w

            canvas[y0:y1, x0:x1] = img

        idx += cols

    out = os.path.join(OUTPUT_DIR, f"XY01_{ch}_stitched.tif")

    tifffile.imwrite(out, canvas)

    print("Saved:", out)

print("\nDone.\n")