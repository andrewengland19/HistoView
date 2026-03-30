import os
import re
import numpy as np
import tifffile
import xml.etree.ElementTree as ET

def get_grid_size():

    xml_path = os.path.join(
        os.getcwd(),
        "Image_XY01_extracted",
        "GroupFileProperty",
        "ImageJoint",
        "properties.xml"
    )

    tree = ET.parse(xml_path)
    root = tree.getroot()

    rows = None
    cols = None

    for elem in root.iter():

        if elem.tag == "Row":
            rows = int(elem.text)

        if elem.tag == "Column":
            cols = int(elem.text)

    return rows, cols


def load_tiles():

    tile_regex = re.compile(r"Image_XY01_(\d{5})_CH1")

    tiles = {}

    for f in os.listdir():

        m = tile_regex.search(f)

        if m:

            tile_id = int(m.group(1))

            tiles[tile_id] = tifffile.imread(f)

    return tiles


def stitch_tiles(tiles, rows, cols):

    sample = next(iter(tiles.values()))

    if sample.ndim == 3:
        sample = sample[:,:,0]

    tile_h, tile_w = sample.shape

    canvas = np.zeros(
        (rows*tile_h, cols*tile_w),
        dtype=sample.dtype
    )

    for tile_id, img in tiles.items():

        if img.ndim == 3:
            img = img[:,:,0]

        idx = tile_id - 1

        row = idx // cols
        col = idx % cols

        if row % 2 == 1:
            col = cols - col - 1

        y = row * tile_h
        x = col * tile_w

        print(f"placing tile {tile_id} → row {row}, col {col}")

        canvas[y:y+tile_h, x:x+tile_w] = img

    return canvas


def main():

    print("Reading grid metadata...")

    rows, cols = get_grid_size()

    print(f"Grid: {rows} x {cols}")

    print("Loading tiles...")

    tiles = load_tiles()

    print(f"Found {len(tiles)} tiles")

    stitched = stitch_tiles(tiles, rows, cols)

    print("Saving output...")

    tifffile.imwrite("stitched_XY01_CH1.tif", stitched)

    print("Done.")


if __name__ == "__main__":
    main()