"""
generate_manifest.py
====================
Companion script for histology_qc_viewer.py.

Scans the predicted_order_<ratID>.csv sorting files (the same source the
viewer uses) and writes a single manifest CSV listing every section for
every rat.  The manifest has an `include` column (default 1) that the user
edits to 0 for any section they want excluded from quantification.

Usage
-----
  # Generate / refresh the manifest
  python generate_manifest.py

  # Point at a different cohort directory
  python generate_manifest.py ~/Microscopy/Cohort2_5HT

Workflow
--------
1. Run this script once to produce  <BASE_DIR>/section_manifest.csv
2. Open the CSV in Excel / Numbers / any editor.
3. Set  include = 0  on any row you want skipped.
4. Save the CSV.
5. Launch the viewer with  --manifest-only  (or MANIFEST_ONLY = True in the
   viewer config) and it will load only the included sections.

Manifest columns
----------------
  rat         rat identifier (matches directory / CSV naming)
  section     CH1 stem string  (e.g. rat761_RBS_XY35_CH1)
  xy_label    XY tile label    (e.g. XY35)
  region      anatomical region parsed from stem, if present
  ch1_exists  1 / 0 — whether the CH1 PNG was found on disk
  include     1 = load in viewer, 0 = skip  (user-editable)
  notes       free-text field for the user
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Mirror viewer defaults — override via CLI arg
# ---------------------------------------------------------------------------

BASE_DIR   = Path.home() / "Microscopy" / "Cohort1_TPH2"
PNG_DIR    = BASE_DIR / "PNG_fullsize"
SORT_DIR   = BASE_DIR / "features" / "sorting"
MANIFEST   = BASE_DIR / "section_manifest.csv"

CHANNELS   = ["CH1", "CH2", "CH3", "CH4"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rat_from_filename(name: str) -> str:
    m = re.match(r"predicted_order_(.+)\.csv", name)
    return m.group(1) if m else name.replace(".csv", "")


def parse_xy(stem: str) -> str:
    m = re.search(r"(XY\d+)", stem)
    return m.group(1) if m else "XY??"


def parse_region(stem: str) -> str:
    """
    Best-effort extraction of the region token between ratID and XY##.
    e.g. rat761_RBS_XY35_CH1  →  RBS
         rat761_XY35_CH1       →  (none)
    """
    # Strip trailing _CH\d
    base = re.sub(r"_CH\d+$", "", stem)
    # Remove leading rat token (anything up to first underscore + digits / letters)
    parts = base.split("_")
    # Find the XY## index
    xy_idx = next(
        (i for i, p in enumerate(parts) if re.match(r"XY\d+", p, re.I)), None
    )
    if xy_idx is None or xy_idx < 2:
        return ""
    # Everything between parts[1] and XY## is the region
    return "_".join(parts[1:xy_idx])


def ch1_on_disk(rat: str, stem: str) -> int:
    path = PNG_DIR / rat / "CH1" / f"{stem}.png"
    return 1 if path.exists() else 0


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def build_manifest(sort_dir: Path, png_dir: Path) -> pd.DataFrame:
    pattern = "predicted_order_*.csv"
    csvs = sorted(sort_dir.glob(pattern))
    if not csvs:
        raise FileNotFoundError(
            f"No predicted_order_*.csv files found in {sort_dir}"
        )

    rows = []
    for csv_path in csvs:
        rat = rat_from_filename(csv_path.name)
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            print(f"[WARN] Could not read {csv_path}: {exc}")
            continue
        if "image" not in df.columns:
            print(f"[WARN] 'image' column missing in {csv_path}, skipping")
            continue

        for stem in df["image"].dropna():
            stem = str(stem).strip()
            rows.append(
                {
                    "rat":       rat,
                    "section":   stem,
                    "xy_label":  parse_xy(stem),
                    "region":    parse_region(stem),
                    "ch1_exists": ch1_on_disk(rat, stem),
                    "include":   1,
                    "notes":     "",
                }
            )

    if not rows:
        raise RuntimeError("No section records found across all CSVs.")

    manifest = pd.DataFrame(rows)
    print(
        f"[INFO] Built manifest: {len(manifest)} sections across "
        f"{manifest['rat'].nunique()} rat(s)."
    )
    return manifest


def merge_with_existing(new_df: pd.DataFrame, existing_path: Path) -> pd.DataFrame:
    """
    If a manifest already exists, preserve the user's  include  and  notes
    values for any section that is already present.  New sections get the
    default (include=1, notes='').
    """
    if not existing_path.exists():
        return new_df

    try:
        old = pd.read_csv(existing_path, dtype={"include": int, "notes": str})
    except Exception as exc:
        print(f"[WARN] Could not read existing manifest ({exc}); overwriting.")
        return new_df

    key = ["rat", "section"]
    preserved = old[key + ["include", "notes"]].copy()
    merged = new_df.drop(columns=["include", "notes"]).merge(
        preserved, on=key, how="left"
    )
    # Fill any new sections with defaults
    merged["include"] = merged["include"].fillna(1).astype(int)
    merged["notes"]   = merged["notes"].fillna("")
    n_new = len(merged) - len(old)
    if n_new > 0:
        print(f"[INFO] {n_new} new section(s) added to existing manifest.")
    print(f"[INFO] User include/notes values preserved for existing sections.")
    return merged


def write_manifest(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[INFO] Manifest written → {path}")
    # Print quick summary
    total    = len(df)
    included = int((df["include"] == 1).sum())
    excluded = total - included
    missing  = int((df["ch1_exists"] == 0).sum())
    print(
        f"\n  Total sections : {total}\n"
        f"  Included       : {included}\n"
        f"  Excluded       : {excluded}\n"
        f"  CH1 missing    : {missing}\n"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    global BASE_DIR, PNG_DIR, SORT_DIR, MANIFEST

    if len(sys.argv) > 1:
        BASE_DIR = Path(sys.argv[1]).expanduser().resolve()
        PNG_DIR  = BASE_DIR / "PNG_fullsize"
        SORT_DIR = BASE_DIR / "features" / "sorting"
        MANIFEST = BASE_DIR / "section_manifest.csv"
        print(f"[INFO] Using base directory: {BASE_DIR}")

    for d, label in [(PNG_DIR, "PNG"), (SORT_DIR, "sorting")]:
        if not d.exists():
            print(f"[WARN] Expected directory not found: {d}  ({label})")

    try:
        new_manifest = build_manifest(SORT_DIR, PNG_DIR)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    final = merge_with_existing(new_manifest, MANIFEST)
    write_manifest(final, MANIFEST)
    print(
        f"\nNext steps:\n"
        f"  1. Open {MANIFEST}\n"
        f"  2. Set include=0 on any section to skip.\n"
        f"  3. Launch the viewer:\n"
        f"       python histology_qc_viewer.py --manifest-only\n"
        f"     (or set MANIFEST_ONLY = True near the top of the viewer script)\n"
    )


if __name__ == "__main__":
    main()
