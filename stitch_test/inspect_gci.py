# inspect_gci.py
import sys
import struct
import re
from pathlib import Path

def printable_strings(data, min_len=4):
    pattern = rb"[\x20-\x7E]{" + str(min_len).encode() + rb",}"
    return [m.decode("ascii", errors="ignore") for m in re.findall(pattern, data)]

def dump_hex(data, start=0, length=256, width=16):
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

def scan_int32(data, limit=200):
    hits = []
    for i in range(0, len(data) - 4, 4):
        val = struct.unpack("<I", data[i:i+4])[0]
        if 0 < val < 1000000:
            hits.append((i, val))
    return hits[:limit]

def main(path_str):
    path = Path(path_str)
    data = path.read_bytes()

    print(f"\nFile: {path}")
    print(f"Size: {len(data):,} bytes\n")

    print("=== First 512 bytes (hex) ===")
    dump_hex(data, 0, 512)

    print("\n=== Printable ASCII strings ===")
    strings = printable_strings(data, min_len=4)
    for s in strings[:300]:
        print(s)

    print("\n=== Candidate float32 values ===")
    for off, val in scan_float32(data):
        print(f"{off:08X}: {val}")

    print("\n=== Candidate uint32 values ===")
    for off, val in scan_int32(data):
        print(f"{off:08X}: {val}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inspect_gci.py <file.gci>")
        sys.exit(1)
    main(sys.argv[1])