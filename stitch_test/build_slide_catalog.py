from __future__ import annotations

import re
import shutil
from pathlib import Path
import csv

# ---------- CONFIG ----------
CHANNEL_LABELS = {
    "ch1": "DAPI",
    "ch2": "GFP_5HT",
    "ch3": "RFP_mCherry",
    "ch4": "Cy5_NeuN",
}

# Change if your extensions differ
VALID_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

# Filename pattern like: XR2CB01_sec01-overlay.tif or XR2CB01_sec01-ch4.tif
PATTERN = re.compile(
    r"^(?P<slide>[^_]+)_(?P<section>sec\d+)-(?P<kind>overlay|ch\d+)\.(?P<ext>tif|tiff|png|jpg|jpeg)$",
    re.IGNORECASE
)

# ---------- HELPERS ----------
def parse_file(p: Path):
    m = PATTERN.match(p.name)
    if not m:
        return None
    d = m.groupdict()
    slide = d["slide"]
    section = d["section"].lower()
    kind = d["kind"].lower()
    ext = "." + d["ext"].lower()
    canonical = f"{slide}_{section}"
    return slide, section, canonical, kind, ext

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# ---------- STEP 1: ORGANIZE + RENAME ----------
def organize_exports(src_dir: Path, project_dir: Path, keep_inbox_copy: bool = True):
    inbox = project_dir / "00_inbox_exports"
    by_id = project_dir / "01_sections_by_id"

    safe_mkdir(project_dir)
    safe_mkdir(by_id)
    if keep_inbox_copy:
        safe_mkdir(inbox)

    for f in src_dir.iterdir():
        if not f.is_file():
            continue
        if f.suffix.lower() not in VALID_EXTS:
            continue

        parsed = parse_file(f)
        if not parsed:
            # Unmatched files go to a misc bucket
            misc_dir = project_dir / "01_sections_by_id" / "_UNMATCHED"
            safe_mkdir(misc_dir)
            shutil.copy2(f, misc_dir / f.name)
            continue

        slide, section, canonical, kind, ext = parsed

        # destination folder: 01_sections_by_id/<slide>/<section>/
        dest_dir = by_id / slide / section
        safe_mkdir(dest_dir)

        # Build new filenames
        if kind == "overlay":
            new_name = f"{canonical}_overlay{ext}"
        else:
            # kind is ch#
            label = CHANNEL_LABELS.get(kind, "UNKNOWN")
            new_name = f"{canonical}_{kind}_{label}{ext}"

        # Copy original to inbox (optional)
        if keep_inbox_copy:
            shutil.copy2(f, inbox / f.name)

        # Move into organized folder (copy2 if you prefer non-destructive)
        shutil.copy2(f, dest_dir / new_name)

# ---------- STEP 2: BUILD OVERLAY REVIEW INDEX ----------
def build_overlay_review(project_dir: Path):
    review_dir = project_dir / "02_review"
    by_id = project_dir / "01_sections_by_id"
    safe_mkdir(review_dir)

    overlays = sorted(by_id.rglob("*_overlay.*"))

    # If overlays are TIFF, browsers may not render. We’ll still index them,
    # but you may want to export PNGs for review separately.
    html = [
        "<html><head><meta charset='utf-8'><title>Overlay Review</title></head><body>",
        "<h1>Overlay Review</h1>",
        "<p>Tip: open in a browser and use Find (Ctrl/Cmd+F) to jump to a slide or section.</p>",
        "<ul>"
    ]

    for p in overlays:
        rel = p.relative_to(project_dir)
        html.append(f"<li><a href='../{rel.as_posix()}'>{p.stem}</a></li>")

    html += ["</ul></body></html>"]
    out = review_dir / "XR2_overlays_index.html"
    out.write_text("\n".join(html), encoding="utf-8")
    print(f"Wrote: {out}")

# ---------- STEP 3 (OPTIONAL): CREATE ROSTRAL-CAUDAL VIEW FROM CSV ----------
def build_rostral_caudal_view(project_dir: Path, map_csv: Path, mode: str = "copy"):
    """
    mode = 'copy' or 'symlink' (symlink is great if you’re on macOS/Linux and comfortable with it)
    """
    by_axis = project_dir / "04_by_axis"
    by_id = project_dir / "01_sections_by_id"
    safe_mkdir(by_axis)

    with map_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    required = {"slide_id", "sec_id", "rc_index"}
    if not required.issubset(reader.fieldnames or []):
        raise ValueError(f"CSV must include columns: {sorted(required)}")

    for r in rows:
        slide = r["slide_id"].strip()
        sec = r["sec_id"].strip().lower()
        rc = int(r["rc_index"])

        src_dir = by_id / slide / sec
        if not src_dir.exists():
            print(f"Missing source folder: {src_dir}")
            continue

        dest_dir = by_axis / slide / f"RC_{rc:03d}"
        safe_mkdir(dest_dir)

        for file in src_dir.iterdir():
            if not file.is_file():
                continue
            dest = dest_dir / file.name
            if dest.exists():
                continue

            if mode == "symlink":
                dest.symlink_to(file)
            else:
                shutil.copy2(file, dest)

    print(f"Built rostral-caudal view in: {by_axis}")

# ---------- RUN ----------
if __name__ == "__main__":
    # EDIT THESE PATHS
    SRC_EXPORT_DIR = Path(r"G:\COHORT 1 HISTO RAW\CB\2026-03-02\XR2-CB\reconstructed")
    PROJECT_DIR    = Path(r"C:\Users\tur94607\Desktop\5HT_NeuN_CaudalBrainstem\Master\WT2_5HT_NeuN")

    organize_exports(SRC_EXPORT_DIR, PROJECT_DIR, keep_inbox_copy=True)
    build_overlay_review(PROJECT_DIR)

    # Later, once you make the CSV map:
    # build_rostral_caudal_view(PROJECT_DIR, PROJECT_DIR / "03_ordering" / "rostral_caudal_map.csv", mode="copy")