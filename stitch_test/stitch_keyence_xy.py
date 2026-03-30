import os
import xml.etree.ElementTree as ET
import imagej

XY = "XY01"
CHANNEL = "CH3"   # DAPI for your current export
OVERLAP = 10      # start here; repo uses 30 by default
FIJI_PATH = r"C:\Path\To\Fiji.app"   # change this

def read_grid():
    xml_path = os.path.join(
        os.getcwd(),
        f"Image_{XY}_extracted",
        "GroupFileProperty",
        "ImageJoint",
        "properties.xml"
    )
    tree = ET.parse(xml_path)
    root = tree.getroot()

    rows = cols = None
    for e in root.iter():
        if e.tag == "Row":
            rows = int(e.text)
        elif e.tag == "Column":
            cols = int(e.text)

    if rows is None or cols is None:
        raise RuntimeError("Could not read Row/Column from ImageJoint/properties.xml")

    return rows, cols

def main():
    rows, cols = read_grid()
    directory = os.getcwd().replace("\\", "/")

    print(f"Grid: {rows} x {cols}")
    print(f"Directory: {directory}")
    print(f"Channel: {CHANNEL}")

    ij = imagej.init(FIJI_PATH, mode='headless')

    # Important:
    # - pure macro, no #@ lines
    # - wrap directory in []
    # - use 5-digit index pattern
    # - use the exact macro-style labels
    macro = f'''
run("Grid/Collection stitching", 
"type=[Grid: snake by rows] "
+ "order=[Right & Down] "
+ "grid_size_x={cols} "
+ "grid_size_y={rows} "
+ "tile_overlap={OVERLAP} "
+ "first_file_index_i=1 "
+ "directory=[{directory}] "
+ "file_names=Image_{XY}_{{iiiii}}_{CHANNEL}.tif "
+ "output_textfile_name=TileConfiguration.txt "
+ "fusion_method=[Linear Blending] "
+ "regression_threshold=0.30 "
+ "max/avg_displacement_threshold=2.50 "
+ "absolute_displacement_threshold=3.50 "
+ "compute_overlap "
+ "subpixel_accuracy "
+ "computation_parameters=[Save computation time (but use more RAM)] "
+ "image_output=[Fuse and display]");
'''

    print("Running stitching macro...")
    ij.py.run_macro(macro)
    print("Finished.")

if __name__ == "__main__":
    main()