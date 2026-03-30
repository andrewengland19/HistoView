# inspect_makernote.py
import sys
import struct
import re
import tifffile

def printable_strings(data, min_len=4):
    pattern = rb"[\x20-\x7E]{" + str(min_len).encode() + rb",}"
    return [m.decode("ascii", errors="ignore") for m in re.findall(pattern, data)]

def dump_hex(data, start=0, length=512, width=16):
    end = min(len(data), start + length)
    for i in range(start, end, width):
        chunk = data[i:i+width]
        hexpart = " ".join(f"{b:02X}" for b in chunk)
        asciipart = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
        print(f"{i:08X}  {hexpart:<{width*3}}  {asciipart}")

def scan_float32(data, limit=200):
    hits = []
    for i in range(0, len(data) - 4, 4):
        try:
            val = struct.unpack("<f", data[i:i+4])[0]
            if abs(val) > 0 and abs(val) < 1e7:
                hits.append((i, val))
        except Exception:
            pass
    return hits[:limit]

fname = sys.argv[1]

with tifffile.TiffFile(fname) as tif:
    exif = tif.pages[0].tags["ExifTag"].value
    maker = exif.get("MakerNote", b"")

print(f"\nMakerNote length: {len(maker)} bytes\n")

print("=== First 512 bytes ===")
dump_hex(maker, 0, 512)

print("\n=== Printable strings ===")
for s in printable_strings(maker):
    print(s)

print("\n=== Candidate float32 values ===")
for off, val in scan_float32(maker):
    print(f"{off:08X}: {val}")