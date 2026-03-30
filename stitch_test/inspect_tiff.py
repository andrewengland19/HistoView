import tifffile
import pprint
import sys

fname = sys.argv[1]

with tifffile.TiffFile(fname) as tif:

    print("\n===== TIFF TAGS =====\n")

    for tag in tif.pages[0].tags.values():
        print(tag.name, ":", tag.value)

    print("\n===== IMAGE DESCRIPTION =====\n")

    desc = tif.pages[0].tags.get("ImageDescription")
    if desc:
        print(desc.value)

    print("\n===== FULL TIFF STRUCTURE =====\n")

    pprint.pprint(tif.series)

    print ("\n==== TAG NAMES ====\n")

    for tag in tif.pages[0].tags.values():
        print(tag.code, tag.name)