from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# -----------------------------
# PATHS (your standard format)
# -----------------------------
HOME = Path.home()
WORK = HOME / "SectionSorter"

IN_DIR = WORK / "review_images"
OUT_DIR = WORK / "diagnostics"
OUT_DIR.mkdir(exist_ok=True)

OUT_PDF = OUT_DIR / "PI_review_packet.pdf"

# -----------------------------
# WHAT TO INCLUDE (EDIT THIS)
# -----------------------------
# Pick 4–8 key sections per rat (order index in filename, e.g., 001, 012, 025)
# Tip: include rostral-ish, mid, caudal-ish + strongest mCherry slice.
REPRESENTATIVE = {
    "761CB": [1, 5, 10, 15],
    # "763CB": [1, 6, 12, 18],
    # "766CB": [1, 6, 12, 18],
    # "768CB": [1, 6, 12, 18],
    # "WT2":   [1, 6, 12, 18],
}

# Channels to show on each page (2x2 grid)
GRID = [
    ("5ht", "5-HT"),
    ("mCherry", "mCherry"),
    ("neun", "NeuN"),
    ("dapi", "DAPI"),
]

# -----------------------------
# LAYOUT SETTINGS
# -----------------------------
PAGE_W, PAGE_H = letter
MARGIN = 36  # 0.5 inch
GAP = 12

TITLE_Y = PAGE_H - MARGIN - 10
CAPTION_Y = PAGE_H - MARGIN - 28

# 2x2 grid region
GRID_TOP = PAGE_H - MARGIN - 60
GRID_LEFT = MARGIN
GRID_W = PAGE_W - 2 * MARGIN
GRID_H = PAGE_H - (MARGIN + 80) - MARGIN

CELL_W = (GRID_W - GAP) / 2
CELL_H = (GRID_H - GAP) / 2


def img_path(rat: str, order_idx: int, key: str) -> Path:
    # review_images/<rat>/###_<key>.jpg
    return IN_DIR / rat / f"{order_idx:03d}_{key}.jpg"


def draw_image_fit(c: canvas.Canvas, path: Path, x: float, y: float, w: float, h: float):
    """Draw image fit into box while preserving aspect ratio."""
    if not path.exists():
        # placeholder box if missing
        c.rect(x, y, w, h, stroke=1, fill=0)
        c.setFont("Helvetica", 9)
        c.drawString(x + 4, y + h - 14, f"Missing: {path.name}")
        return

    img = ImageReader(str(path))
    iw, ih = img.getSize()

    # scale to fit
    s = min(w / iw, h / ih)
    dw, dh = iw * s, ih * s
    dx = x + (w - dw) / 2
    dy = y + (h - dh) / 2

    c.drawImage(img, dx, dy, dw, dh, preserveAspectRatio=True, mask="auto")


def main():
    c = canvas.Canvas(str(OUT_PDF), pagesize=letter)

    # Title page
    c.setFont("Helvetica-Bold", 16)
    c.drawString(MARGIN, PAGE_H - MARGIN, "Brainstem Validation — PI Review Packet")
    c.setFont("Helvetica", 11)
    c.drawString(MARGIN, PAGE_H - MARGIN - 22, "Generated from SectionSorter review_images (compressed, ordered).")
    c.drawString(MARGIN, PAGE_H - MARGIN - 38, "Each page shows: 5-HT, mCherry, NeuN, DAPI (2×2).")
    c.showPage()

    # Content pages
    for rat, sections in REPRESENTATIVE.items():
        for order_idx in sections:
            c.setFont("Helvetica-Bold", 14)
            c.drawString(MARGIN, TITLE_Y, f"{rat} — Ordered section {order_idx:03d}")

            # small caption
            c.setFont("Helvetica", 10)
            c.drawString(MARGIN, CAPTION_Y, "Top-left: 5-HT | Top-right: mCherry | Bottom-left: NeuN | Bottom-right: DAPI")

            # cell coordinates (top-left origin handled by y calculations)
            # Row 0 (top)
            x0 = GRID_LEFT
            x1 = GRID_LEFT + CELL_W + GAP
            y_top = GRID_TOP

            y0 = y_top - CELL_H  # top row y
            y1 = y0 - GAP - CELL_H  # bottom row y

            # Top-left: 5HT
            p = img_path(rat, order_idx, "5ht")
            draw_image_fit(c, p, x0, y0, CELL_W, CELL_H)
            c.setFont("Helvetica", 10)
            c.drawString(x0 + 4, y0 + 4, "5-HT")

            # Top-right: mCherry
            p = img_path(rat, order_idx, "mCherry")
            draw_image_fit(c, p, x1, y0, CELL_W, CELL_H)
            c.setFont("Helvetica", 10)
            c.drawString(x1 + 4, y0 + 4, "mCherry")

            # Bottom-left: NeuN
            p = img_path(rat, order_idx, "neun")
            draw_image_fit(c, p, x0, y1, CELL_W, CELL_H)
            c.setFont("Helvetica", 10)
            c.drawString(x0 + 4, y1 + 4, "NeuN")

            # Bottom-right: DAPI
            p = img_path(rat, order_idx, "dapi")
            draw_image_fit(c, p, x1, y1, CELL_W, CELL_H)
            c.setFont("Helvetica", 10)
            c.drawString(x1 + 4, y1 + 4, "DAPI")

            c.showPage()

    c.save()
    print(f"Wrote: {OUT_PDF}")


if __name__ == "__main__":
    main()