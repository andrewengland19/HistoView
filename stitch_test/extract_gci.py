import zipfile
import sys
from pathlib import Path

gci = Path(sys.argv[1])

outdir = gci.stem + "_extracted"

with zipfile.ZipFile(gci, 'r') as z:
    z.extractall(outdir)

print("Extracted to:", outdir)