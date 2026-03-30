"""
histology_qc_viewer.py
======================
Napari-based keyboard-driven QC viewer for rat histology datasets.

Features
--------
  * Blinded cell-counting quantification mode with per-channel counts
  * Cells are counted per visible napari layer (DAPI, GFP, RFP, Cy5)
  * Right-click places a colour-coded marker on the active channel layer
  * Ctrl+S in quant mode saves and completes the current section
  * DeepLabCut-compatible coordinate export (flat CSV + DLC CollectedData)
  * Per-counter session resume and multi-counter support
  * Post-hoc section exclusion from per-rat exports
  * Photo-album-style subsets: named collections of sections you can share,
    load as a navigation source, and use for figures / student quant / etc.

Directory layout assumed
------------------------
~/Microscopy/Cohort1_TPH2/
    PNG_fullsize/<ratID>/CH1 ... CH4/
    features/sorting/predicted_order_<ratID>.csv  (column: "image")

Generated output files (all relative to BASE_DIR)
--------------------------------------------------
  qc_annotations.csv              QC flags / notes per section
  cell_counts_clicks.csv          Every right-click with channel + coords
  cell_counts_summary.csv         One row per completed section per counter
                                  (per-channel totals + excluded flag)
  cell_counts_per_rat.csv         Derived per-rat export (excluded omitted)
  dlc_coords.csv                  Flat DLC-ready coordinate table
  dlc_labeled_data/
      CollectedData_<scorer>.csv  DeepLabCut multi-index header format
  subsets/<name>.json             Photo-album subset files
  quant_progress_<id>.json        Resume checkpoint (deleted on clean exit)

Keyboard map — Normal mode
--------------------------
  <- / ->     previous / next section (within current navigation source)
  [ / ]       previous / next rat  (only in full-dataset mode)
  1-4         show CH1 / CH2 / CH3 / CH4 only
  Space       toggle overlay (all channels) <-> last single-channel
  m           toggle MIRROR flag
  r           rotate image 90 degrees clockwise
  q           enter QUANTIFICATION MODE
  x           toggle EXCLUDED flag on a counted section (post-hoc)
  a           add current section to a named subset / album (console)
  n           enter / edit NOTES (console)
  Ctrl+L      load a subset as the navigation source (console)
  Ctrl+A      list all subsets
  Ctrl+E      export per-rat counts CSV
  Ctrl+S      force-save QC annotations
  c           pin current section as comparison reference (Rat A)
  Tab         flip between the pinned reference and current section
  Ctrl+P      export the two paired image names to a .txt file

Keyboard map — Quantification mode
------------------------------------
  Right-click on a layer   place a marker and count for THAT channel
  Backspace                undo last marker (any channel)
  Ctrl+S                   save & complete the current section, advance
  Delete                   skip this section (discard clicks, re-queue)
  <- / ->                  navigate without completing (resume-safe)
  q                        exit quantification mode

CLI usage
---------
  python histology_qc_viewer.py [BASE_DIR] [--counter NAME]
  python histology_qc_viewer.py [BASE_DIR] --export
  python histology_qc_viewer.py [BASE_DIR] --export-dlc
  python histology_qc_viewer.py [BASE_DIR] --exclude IMAGE_STEM [--counter ID]
      [--exclude-reason TEXT]
  python histology_qc_viewer.py [BASE_DIR] --subset NAME
      Load subset NAME as navigation source on startup.

Author: generated for Cohort1_TPH2 pipeline
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import napari
import numpy as np
import pandas as pd
from imageio import v3 as iio
from napari.utils.notifications import show_warning, show_info

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR    = Path.home() / "Microscopy" / "Cohort1_TPH2"
PNG_DIR     = BASE_DIR / "PNG_fullsize"
SORT_DIR    = BASE_DIR / "features" / "sorting"
DATASET_DIR = BASE_DIR / "dataset"
QC_CSV      = BASE_DIR / "qc_annotations.csv"
COUNTS_CSV  = BASE_DIR / "cell_counts.csv"

CHANNELS     = ["CH1", "CH2", "CH3", "CH4"]
CH_LABELS    = {"CH1": "DAPI", "CH2": "TPH2/GFP", "CH3": "RFP", "CH4": "Cy5"}
CH_COLORMAPS = {"CH1": "blue",  "CH2": "green",    "CH3": "red",  "CH4": "magenta"}

# Marker colours per channel (RGBA 0-1) — match the channel colourmap roughly
CH_MARKER_COLORS: dict[str, np.ndarray] = {
    "CH1": np.array([[0.4, 0.6, 1.0, 0.95]]),   # pale blue   — DAPI
    "CH2": np.array([[0.2, 1.0, 0.2, 0.95]]),   # green       — GFP
    "CH3": np.array([[1.0, 0.3, 0.3, 0.95]]),   # red         — RFP
    "CH4": np.array([[0.9, 0.2, 0.9, 0.95]]),   # magenta     — Cy5
}
MARKER_COLOR_DONE = np.array([[1.0, 1.0, 1.0, 0.6]])  # white — section completed


# ---------------------------------------------------------------------------
# Data-loader abstraction
# ---------------------------------------------------------------------------

class SectionRecord:
    """One XY section for one rat, with derived paths for all channels."""

    __slots__ = ("rat", "stem_ch1", "stems", "png_paths", "xy_label")

    def __init__(self, rat: str, stem_ch1: str):
        self.rat       = rat
        self.stem_ch1  = stem_ch1
        self.xy_label  = self._parse_xy(stem_ch1)
        self.stems     = {ch: self._derive_stem(stem_ch1, ch) for ch in CHANNELS}
        self.png_paths = self._build_png_paths()

    @staticmethod
    def _parse_xy(stem: str) -> str:
        m = re.search(r"(XY\d+)", stem)
        return m.group(1) if m else "XY??"

    @staticmethod
    def _derive_stem(stem_ch1: str, ch: str) -> str:
        return re.sub(r"_CH1$", f"_{ch}", stem_ch1)

    def _build_png_paths(self) -> dict[str, Path]:
        return {ch: PNG_DIR / self.rat / ch / f"{self.stems[ch]}.png"
                for ch in CHANNELS}

    def __repr__(self) -> str:
        return f"<SectionRecord rat={self.rat} xy={self.xy_label}>"


class DatasetLoader:
    """Discovers all rats from predicted_order_<ratID>.csv files."""

    def __init__(self, sort_dir: Path = SORT_DIR):
        self.sort_dir  = sort_dir
        self.rats: list[str]                          = []
        self.sections: dict[str, list[SectionRecord]] = {}
        self._load()

    def _load(self) -> None:
        csvs = sorted(self.sort_dir.glob("predicted_order_*.csv"))
        if not csvs:
            raise FileNotFoundError(
                f"No predicted_order_*.csv files found in {self.sort_dir}")
        for csv_path in csvs:
            rat = self._rat_from_filename(csv_path.name)
            try:
                df = pd.read_csv(csv_path)
            except Exception as exc:
                print(f"[WARN] Could not read {csv_path}: {exc}")
                continue
            if "image" not in df.columns:
                print(f"[WARN] 'image' column missing in {csv_path}, skipping")
                continue
            stems   = df["image"].dropna().tolist()
            records = [SectionRecord(rat, str(s)) for s in stems]
            self.rats.append(rat)
            self.sections[rat] = records
        if not self.rats:
            raise RuntimeError("No valid rat datasets loaded.")
        print(f"[INFO] Loaded {len(self.rats)} rat(s): {self.rats}")

    @staticmethod
    def _rat_from_filename(name: str) -> str:
        m = re.match(r"predicted_order_(.+)\.csv", name)
        return m.group(1) if m else name.replace(".csv", "")

    def get_sections(self, rat: str) -> list[SectionRecord]:
        return self.sections.get(rat, [])

    def all_sections_flat(self) -> list[SectionRecord]:
        out = []
        for rat in self.rats:
            out.extend(self.sections[rat])
        return out

    def section_by_stem(self, stem_ch1: str) -> Optional[SectionRecord]:
        for rat in self.rats:
            for sec in self.sections[rat]:
                if sec.stem_ch1 == stem_ch1:
                    return sec
        return None


# ---------------------------------------------------------------------------
# QC annotation store
# ---------------------------------------------------------------------------

class QCStore:
    """Reads/writes per-section QC annotations to a CSV."""

    COLUMNS = ["rat", "image", "mirror", "rotate", "quantify", "notes"]
    _lock   = threading.Lock()

    def __init__(self, path: Path = QC_CSV):
        self.path = path
        if path.exists():
            try:
                self._df = pd.read_csv(path)
                for col in self.COLUMNS:
                    if col not in self._df.columns:
                        self._df[col] = "" if col == "notes" else False
            except Exception as exc:
                print(f"[WARN] Could not read QC CSV ({exc}); starting fresh.")
                self._df = pd.DataFrame(columns=self.COLUMNS)
        else:
            self._df = pd.DataFrame(columns=self.COLUMNS)

    def _key(self, rat: str, image: str) -> pd.Series:
        return (self._df["rat"] == rat) & (self._df["image"] == image)

    def get(self, rat: str, image: str) -> dict:
        mask = self._key(rat, image)
        if mask.any():
            row = self._df[mask].iloc[0]
            return {k: (str(row.get(k, "")) if k in ("notes",)
                        else bool(row.get(k, False)))
                    for k in ("mirror", "rotate", "quantify", "notes")}
        return {"mirror": False, "rotate": False, "quantify": False, "notes": ""}

    def set(self, rat: str, image: str, **kwargs) -> None:
        with self._lock:
            mask = self._key(rat, image)
            if mask.any():
                idx = self._df[mask].index[0]
                for k, v in kwargs.items():
                    if k in self.COLUMNS:
                        self._df.at[idx, k] = v
            else:
                row = {"rat": rat, "image": image,
                       "mirror": False, "rotate": False,
                       "quantify": False, "notes": ""}
                row.update({k: v for k, v in kwargs.items() if k in self.COLUMNS})
                self._df = pd.concat([self._df, pd.DataFrame([row])],
                                     ignore_index=True)

    def toggle(self, rat: str, image: str, flag: str) -> bool:
        new_val = not self.get(rat, image).get(flag, False)
        self.set(rat, image, **{flag: new_val})
        return new_val

    def save(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._df.to_csv(self.path, index=False)
            print(f"[INFO] QC annotations saved -> {self.path}")


# ---------------------------------------------------------------------------
# Cell-count store  (per-channel, manual completion, DLC export)
# ---------------------------------------------------------------------------

class CellCountStore:
    """
    Manages per-channel cell-count data.

    Click schema
    ------------
    Each right-click records:  session_id, counter_id, rat, image, xy_label,
        channel (CH1-CH4), channel_label, click_index, coord_y, coord_x,
        rotation_applied, timestamp

    On Ctrl+S the user manually completes the section.  Summary schema:
        session_id, counter_id, rat, image, xy_label,
        count_CH1, count_CH2, count_CH3, count_CH4, total_count,
        completed_at, excluded, exclude_reason, excluded_at

    DeepLabCut export (dlc_coords.csv)
    ------------------------------------
    Flat format: scorer, image_path, channel, bodypart, x, y, likelihood
    Plus DLC multi-index CollectedData_<scorer>.csv written to
    BASE_DIR/dlc_labeled_data/.

    The 'bodypart' label is the channel label (DAPI, TPH2/GFP, RFP, Cy5).
    'likelihood' is always 1.0 (manually labelled).
    """

    CLICKS_COLS  = ["session_id", "counter_id", "rat", "image", "xy_label",
                    "channel", "channel_label",
                    "click_index", "coord_y", "coord_x",
                    "rotation_applied", "timestamp"]

    SUMMARY_COLS = ["session_id", "counter_id", "rat", "image", "xy_label",
                    "count_CH1", "count_CH2", "count_CH3", "count_CH4",
                    "total_count", "completed_at",
                    "excluded", "exclude_reason", "excluded_at"]

    PER_RAT_COLS = ["rat", "xy_label", "image", "counter_id",
                    "count_CH1", "count_CH2", "count_CH3", "count_CH4",
                    "total_count", "completed_at"]

    def __init__(self, path: Path, counter_id: str):
        self.path        = path
        self.counter_id  = counter_id
        self.session_id  = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Pending clicks for current section only.
        # { (rat, stem_ch1): [ {row}, ... ] }
        self._pending: dict[tuple[str, str], list[dict]] = {}

        self._clicks_path   = path.parent / (path.stem + "_clicks.csv")
        self._summary_path  = path.parent / (path.stem + "_summary.csv")
        self._per_rat_path  = path.parent / (path.stem + "_per_rat.csv")
        self._dlc_flat_path = path.parent / "dlc_coords.csv"
        self._dlc_dir       = path.parent / "dlc_labeled_data"
        self._progress_path = path.parent / f"quant_progress_{counter_id}.json"

    # ------------------------------------------------------------------
    # Pending click management
    # ------------------------------------------------------------------

    def add_click(self, rat: str, image: str, xy_label: str,
                  channel: str,
                  coord_y: float, coord_x: float,
                  rotation_applied: int) -> dict[str, int]:
        """
        Add a pending click for the given channel.
        Returns per-channel pending counts for this section.
        """
        key = (rat, image)
        if key not in self._pending:
            self._pending[key] = []
        ch_count = sum(1 for r in self._pending[key] if r["channel"] == channel)
        self._pending[key].append({
            "session_id":       self.session_id,
            "counter_id":       self.counter_id,
            "rat":              rat,
            "image":            image,
            "xy_label":         xy_label,
            "channel":          channel,
            "channel_label":    CH_LABELS[channel],
            "click_index":      ch_count + 1,
            "coord_y":          round(coord_y, 2),
            "coord_x":          round(coord_x, 2),
            "rotation_applied": rotation_applied,
            "timestamp":        datetime.now().isoformat(timespec="seconds"),
        })
        return self._channel_counts(rat, image)

    def undo_last_click(self, rat: str, image: str) -> tuple[Optional[str], dict[str, int]]:
        """
        Remove the most recent click.
        Returns (removed_channel, remaining per-channel counts).
        """
        key  = (rat, image)
        rows = self._pending.get(key, [])
        removed_ch = None
        if rows:
            removed_ch = rows[-1]["channel"]
            rows.pop()
        return removed_ch, self._channel_counts(rat, image)

    def _channel_counts(self, rat: str, image: str) -> dict[str, int]:
        rows = self._pending.get((rat, image), [])
        return {ch: sum(1 for r in rows if r["channel"] == ch)
                for ch in CHANNELS}

    def pending_total(self, rat: str, image: str) -> int:
        return len(self._pending.get((rat, image), []))

    def pending_channel_counts(self, rat: str, image: str) -> dict[str, int]:
        return self._channel_counts(rat, image)

    def discard_pending(self, rat: str, image: str) -> None:
        self._pending.pop((rat, image), None)

    # ------------------------------------------------------------------
    # Manual completion  (Ctrl+S in quant mode)
    # ------------------------------------------------------------------

    def complete_section(self, rat: str, image: str, xy_label: str) -> dict[str, int]:
        """
        Finalise current section: write clicks + summary row.
        Returns final per-channel counts.
        """
        key   = (rat, image)
        rows  = self._pending.pop(key, [])
        ch_counts = {ch: sum(1 for r in rows if r["channel"] == ch)
                     for ch in CHANNELS}
        total = len(rows)

        if rows:
            self._append_clicks(rows)

        self._write_summary_row(rat, image, xy_label, ch_counts, total)
        return ch_counts

    def _write_summary_row(self, rat: str, image: str, xy_label: str,
                           ch_counts: dict[str, int], total: int,
                           excluded: bool = False,
                           exclude_reason: str = "",
                           excluded_at: str = "") -> None:
        self._summary_path.parent.mkdir(parents=True, exist_ok=True)
        new_row = pd.DataFrame([{
            "session_id":     self.session_id,
            "counter_id":     self.counter_id,
            "rat":            rat,
            "image":          image,
            "xy_label":       xy_label,
            "count_CH1":      ch_counts.get("CH1", 0),
            "count_CH2":      ch_counts.get("CH2", 0),
            "count_CH3":      ch_counts.get("CH3", 0),
            "count_CH4":      ch_counts.get("CH4", 0),
            "total_count":    total,
            "completed_at":   datetime.now().isoformat(timespec="seconds"),
            "excluded":       excluded,
            "exclude_reason": exclude_reason,
            "excluded_at":    excluded_at,
        }], columns=self.SUMMARY_COLS)

        if self._summary_path.exists():
            existing = pd.read_csv(self._summary_path)
            for col in self.SUMMARY_COLS:
                if col not in existing.columns:
                    existing[col] = 0 if col.startswith("count") else \
                                    ("" if col in ("exclude_reason", "excluded_at")
                                     else False)
            combo = pd.concat([existing, new_row], ignore_index=True)
            combo = combo.drop_duplicates(subset=["counter_id", "image"],
                                          keep="last")
            combo.to_csv(self._summary_path, index=False)
        else:
            new_row.to_csv(self._summary_path, index=False)
        print(f"[INFO] Summary updated -> {self._summary_path}")

    # ------------------------------------------------------------------
    # Completed-section query
    # ------------------------------------------------------------------

    def completed_images(self) -> set[str]:
        if not self._summary_path.exists():
            return set()
        try:
            df = pd.read_csv(self._summary_path)
            if "counter_id" not in df.columns or "image" not in df.columns:
                return set()
            return set(df.loc[df["counter_id"] == self.counter_id,
                               "image"].tolist())
        except Exception as exc:
            print(f"[WARN] Could not read summary CSV: {exc}")
            return set()

    # ------------------------------------------------------------------
    # Post-hoc exclusion
    # ------------------------------------------------------------------

    def toggle_exclusion(self, image: str, reason: str = "") -> Optional[bool]:
        if not self._summary_path.exists():
            print(f"[WARN] No summary CSV found; cannot exclude {image!r}")
            return None
        try:
            df = pd.read_csv(self._summary_path)
            for col in ("excluded", "exclude_reason", "excluded_at"):
                if col not in df.columns:
                    df[col] = False if col == "excluded" else ""
            mask = (df["counter_id"] == self.counter_id) & (df["image"] == image)
            if not mask.any():
                print(f"[WARN] No summary row for counter '{self.counter_id}', "
                      f"image '{image}'")
                return None
            idx      = df[mask].index[0]
            new_excl = not bool(df.at[idx, "excluded"])
            df.at[idx, "excluded"] = new_excl
            if new_excl:
                df.at[idx, "exclude_reason"] = reason
                df.at[idx, "excluded_at"]    = datetime.now().isoformat(
                    timespec="seconds")
            else:
                df.at[idx, "exclude_reason"] = ""
                df.at[idx, "excluded_at"]    = ""
            df.to_csv(self._summary_path, index=False)
            action = "EXCLUDED" if new_excl else "RE-INCLUDED"
            print(f"[INFO] '{image}' {action} for counter '{self.counter_id}'")
            return new_excl
        except Exception as exc:
            print(f"[WARN] Could not update exclusion flag: {exc}")
            return None

    def is_excluded(self, image: str) -> bool:
        if not self._summary_path.exists():
            return False
        try:
            df   = pd.read_csv(self._summary_path)
            mask = ((df.get("counter_id", pd.Series(dtype=str)) == self.counter_id) &
                    (df.get("image",      pd.Series(dtype=str)) == image))
            return bool(df[mask].iloc[0].get("excluded", False)) if mask.any() else False
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Per-rat export
    # ------------------------------------------------------------------

    def export_per_rat(self, path: Optional[Path] = None) -> Path:
        dest = path or self._per_rat_path
        dest.parent.mkdir(parents=True, exist_ok=True)

        if not self._summary_path.exists():
            print("[WARN] No summary CSV; per-rat export is empty.")
            pd.DataFrame(columns=self.PER_RAT_COLS).to_csv(dest, index=False)
            return dest

        df = pd.read_csv(self._summary_path)
        if "excluded" not in df.columns:
            df["excluded"] = False

        active = df[~df["excluded"].astype(bool)].copy()
        if active.empty:
            pd.DataFrame(columns=self.PER_RAT_COLS).to_csv(dest, index=False)
            return dest

        for col in ("count_CH1", "count_CH2", "count_CH3", "count_CH4"):
            if col not in active.columns:
                active[col] = 0

        def _xy_num(xy: str) -> int:
            m = re.search(r"(\d+)", str(xy))
            return int(m.group(1)) if m else 0

        active["_xy_n"] = active["xy_label"].apply(_xy_num)
        active = active.sort_values(["rat", "_xy_n"]).drop(columns=["_xy_n"])
        out    = active[[c for c in self.PER_RAT_COLS if c in active.columns]]
        out.to_csv(dest, index=False)
        print(f"\n[EXPORT] Per-rat counts -> {dest}\n"
              f"         {len(out)} sections, {out['rat'].nunique()} rat(s), "
              f"{out['counter_id'].nunique()} counter(s)")
        return dest

    # ------------------------------------------------------------------
    # DeepLabCut export
    # ------------------------------------------------------------------

    def export_dlc(self) -> tuple[Path, Path]:
        """
        Build two DLC-compatible files from *_clicks.csv:

        1. dlc_coords.csv  (flat, easy to parse)
           Columns: scorer, image_path, channel, bodypart, x, y, likelihood

        2. dlc_labeled_data/CollectedData_<scorer>.csv
           Standard DLC multi-index header format that DLC's GUI and
           training pipeline accept directly.

        Returns (flat_path, dlc_csv_path).
        """
        if not self._clicks_path.exists():
            print("[WARN] No clicks CSV found; DLC export is empty.")
            return self._dlc_flat_path, self._dlc_dir / f"CollectedData_{self.counter_id}.csv"

        df = pd.read_csv(self._clicks_path)
        # Keep only this counter's clicks
        df = df[df["counter_id"] == self.counter_id].copy()
        if df.empty:
            print(f"[WARN] No clicks for counter '{self.counter_id}'.")
            return self._dlc_flat_path, self._dlc_dir / f"CollectedData_{self.counter_id}.csv"

        # Build image_path column pointing to the CH1 PNG (full path)
        def _img_path(row: pd.Series) -> str:
            ch = row.get("channel", "CH1")
            rat = row.get("rat", "")
            image = row.get("image", "")
            # Derive stem for this channel
            stem = re.sub(r"_CH1$", f"_{ch}", image)
            p    = PNG_DIR / rat / ch / f"{stem}.png"
            return str(p)

        df["image_path"] = df.apply(_img_path, axis=1)
        df["bodypart"]   = df["channel"].map(CH_LABELS)
        df["likelihood"] = 1.0
        df["scorer"]     = self.counter_id

        # ── flat CSV ──────────────────────────────────────────────
        flat_cols = ["scorer", "image_path", "channel", "bodypart",
                     "x", "y", "likelihood"]
        flat = df.rename(columns={"coord_x": "x", "coord_y": "y"})[flat_cols]
        self._dlc_flat_path.parent.mkdir(parents=True, exist_ok=True)
        flat.to_csv(self._dlc_flat_path, index=False)
        print(f"[DLC] Flat coords -> {self._dlc_flat_path}")

        # ── DLC CollectedData format ──────────────────────────────
        # DLC expects a CSV with a 3-row header:
        #   row 0: scorer  (repeated for each bodypart × coord pair)
        #   row 1: bodypart name
        #   row 2: x or y
        # Index = image path (relative to labeled-images dir or absolute)
        self._dlc_dir.mkdir(parents=True, exist_ok=True)
        dlc_csv = self._dlc_dir / f"CollectedData_{self.counter_id}.csv"

        bodyparts = [CH_LABELS[ch] for ch in CHANNELS]
        scorer    = self.counter_id

        # Build column MultiIndex: (scorer, bodypart, coord)
        col_tuples = []
        for bp in bodyparts:
            col_tuples.append((scorer, bp, "x"))
            col_tuples.append((scorer, bp, "y"))
        col_mi = pd.MultiIndex.from_tuples(col_tuples)

        # Pivot: one row per image_path, columns = (scorer, bodypart, x/y)
        rows_out = {}
        for img_path, grp in df.groupby("image_path"):
            row_data = {}
            for bp in bodyparts:
                sub = grp[grp["bodypart"] == bp]
                if len(sub) == 0:
                    row_data[(scorer, bp, "x")] = np.nan
                    row_data[(scorer, bp, "y")] = np.nan
                else:
                    # If multiple clicks for same bodypart, take centroid
                    row_data[(scorer, bp, "x")] = float(sub["coord_x"].mean())
                    row_data[(scorer, bp, "y")] = float(sub["coord_y"].mean())
            rows_out[img_path] = row_data

        dlc_df = pd.DataFrame.from_dict(rows_out, orient="index")
        dlc_df.columns = col_mi
        dlc_df.index.name = "image"
        dlc_df.to_csv(dlc_csv)
        print(f"[DLC] CollectedData -> {dlc_csv}")

        return self._dlc_flat_path, dlc_csv

    # ------------------------------------------------------------------
    # Session progress checkpoint
    # ------------------------------------------------------------------

    def save_progress(self, shuffle_stems: list[str], current_idx: int) -> None:
        checkpoint = {
            "counter_id":    self.counter_id,
            "session_id":    self.session_id,
            "shuffle_stems": shuffle_stems,
            "current_idx":   current_idx,
            "saved_at":      datetime.now().isoformat(timespec="seconds"),
        }
        try:
            self._progress_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._progress_path, "w") as fh:
                json.dump(checkpoint, fh, indent=2)
        except Exception as exc:
            print(f"[WARN] Could not save progress checkpoint: {exc}")

    def load_progress(self) -> Optional[dict]:
        if not self._progress_path.exists():
            return None
        try:
            with open(self._progress_path) as fh:
                data = json.load(fh)
            return data if data.get("counter_id") == self.counter_id else None
        except Exception as exc:
            print(f"[WARN] Could not load progress checkpoint: {exc}")
            return None

    def delete_progress(self) -> None:
        try:
            if self._progress_path.exists():
                self._progress_path.unlink()
                print(f"[INFO] Checkpoint removed -> {self._progress_path}")
        except Exception as exc:
            print(f"[WARN] Could not delete checkpoint: {exc}")

    # ------------------------------------------------------------------
    # Low-level CSV writer
    # ------------------------------------------------------------------

    def _append_clicks(self, rows: list[dict]) -> None:
        self._clicks_path.parent.mkdir(parents=True, exist_ok=True)
        new_df = pd.DataFrame(rows, columns=self.CLICKS_COLS)
        if self._clicks_path.exists():
            existing = pd.read_csv(self._clicks_path)
            # Ensure schema compatibility
            for col in self.CLICKS_COLS:
                if col not in existing.columns:
                    existing[col] = ""
            combo = pd.concat([existing, new_df], ignore_index=True)
            combo = combo.drop_duplicates(
                subset=["session_id", "counter_id", "image",
                        "channel", "click_index"],
                keep="first")
            combo.to_csv(self._clicks_path, index=False)
        else:
            new_df.to_csv(self._clicks_path, index=False)
        print(f"[INFO] Clicks saved -> {self._clicks_path}")

    def emergency_save_pending(self) -> None:
        """On unclean exit, write pending clicks to the raw CSV only."""
        all_rows = [r for rows in self._pending.values() for r in rows]
        if all_rows:
            self._append_clicks(all_rows)
            print("[INFO] Pending (incomplete) clicks saved for reference.")
        self._pending.clear()


# ---------------------------------------------------------------------------
# Subset / album store
# ---------------------------------------------------------------------------

class SubsetStore:
    """
    Manages named photo-album-style subsets of sections.

    Each subset is a JSON file: BASE_DIR/subsets/<name>.json
    Schema:
        {
          "name":        "for_figures",
          "description": "Best sections for publication figures",
          "created_by":  "Alice",
          "created_at":  "2025-01-15T14:30:00",
          "updated_at":  "2025-01-16T09:00:00",
          "sections":    ["rat761_RBS_XY12_CH1", "rat762_RBS_XY05_CH1", ...]
        }

    The JSON is loadable directly into napari (or any other tool) as a
    simple list of CH1 stems.  Sharing is just copying the JSON file.
    """

    def __init__(self, base_dir: Path):
        self._dir = base_dir / "subsets"
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        safe = re.sub(r"[^\w\-]", "_", name)[:64]
        return self._dir / f"{safe}.json"

    def list_subsets(self) -> list[dict]:
        """Return summary dicts for all subsets, sorted by name."""
        out = []
        for p in sorted(self._dir.glob("*.json")):
            try:
                with open(p) as fh:
                    d = json.load(fh)
                out.append({
                    "name":        d.get("name", p.stem),
                    "description": d.get("description", ""),
                    "n_sections":  len(d.get("sections", [])),
                    "created_by":  d.get("created_by", ""),
                    "created_at":  d.get("created_at", ""),
                    "updated_at":  d.get("updated_at", ""),
                    "path":        str(p),
                })
            except Exception:
                pass
        return out

    def load(self, name: str) -> Optional[dict]:
        p = self._path(name)
        if not p.exists():
            # Also try exact match in case name already is safe
            candidates = list(self._dir.glob(f"{name}.json"))
            if not candidates:
                return None
            p = candidates[0]
        try:
            with open(p) as fh:
                return json.load(fh)
        except Exception as exc:
            print(f"[WARN] Could not load subset '{name}': {exc}")
            return None

    def save(self, name: str, sections: list[str],
             description: str = "", created_by: str = "") -> Path:
        p    = self._path(name)
        now  = datetime.now().isoformat(timespec="seconds")
        # Preserve created_at if this subset already exists
        existing_created = now
        if p.exists():
            try:
                with open(p) as fh:
                    old = json.load(fh)
                existing_created = old.get("created_at", now)
            except Exception:
                pass
        data = {
            "name":        name,
            "description": description,
            "created_by":  created_by,
            "created_at":  existing_created,
            "updated_at":  now,
            "sections":    sections,
        }
        with open(p, "w") as fh:
            json.dump(data, fh, indent=2)
        print(f"[SUBSET] '{name}' saved ({len(sections)} sections) -> {p}")
        return p

    def add_section(self, name: str, stem_ch1: str,
                    description: str = "", created_by: str = "") -> int:
        """Add a section to a subset (creating it if needed). Returns new length."""
        existing = self.load(name) or {}
        secs = existing.get("sections", [])
        if stem_ch1 not in secs:
            secs.append(stem_ch1)
        self.save(name, secs,
                  description=existing.get("description", description),
                  created_by=existing.get("created_by", created_by))
        return len(secs)

    def remove_section(self, name: str, stem_ch1: str) -> int:
        existing = self.load(name)
        if not existing:
            return 0
        secs = [s for s in existing.get("sections", []) if s != stem_ch1]
        self.save(name, secs,
                  description=existing.get("description", ""),
                  created_by=existing.get("created_by", ""))
        return len(secs)

    def delete(self, name: str) -> bool:
        p = self._path(name)
        if p.exists():
            p.unlink()
            print(f"[SUBSET] '{name}' deleted.")
            return True
        return False


# ---------------------------------------------------------------------------
# Per-rat export (standalone CLI mode, all counters)
# ---------------------------------------------------------------------------

def export_per_rat_standalone(summary_path: Path, dest: Path) -> None:
    PER_RAT_COLS = ["rat", "xy_label", "image", "counter_id",
                    "count_CH1", "count_CH2", "count_CH3", "count_CH4",
                    "total_count", "completed_at"]
    if not summary_path.exists():
        print(f"[WARN] Summary not found: {summary_path}")
        pd.DataFrame(columns=PER_RAT_COLS).to_csv(dest, index=False)
        return
    df = pd.read_csv(summary_path)
    if "excluded" not in df.columns:
        df["excluded"] = False
    for col in ("count_CH1", "count_CH2", "count_CH3", "count_CH4"):
        if col not in df.columns:
            df[col] = 0
    active = df[~df["excluded"].astype(bool)].copy()

    def _xy_num(xy: str) -> int:
        m = re.search(r"(\d+)", str(xy))
        return int(m.group(1)) if m else 0

    active["_n"] = active["xy_label"].apply(_xy_num)
    active = active.sort_values(["rat", "_n"]).drop(columns=["_n"])
    out    = active[[c for c in PER_RAT_COLS if c in active.columns]]
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)
    print(f"\n[EXPORT] Per-rat counts -> {dest}\n"
          f"         {len(out)} sections, {out['rat'].nunique()} rat(s), "
          f"{out['counter_id'].nunique()} counter(s)")


# ---------------------------------------------------------------------------
# Image loading helpers
# ---------------------------------------------------------------------------

def load_png_as_array(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        show_warning(f"Missing: {path.name}")
        return None
    try:
        img = iio.imread(path)
        if img.ndim == 3 and img.shape[-1] in (3, 4):
            img = img[..., :3].mean(axis=-1).astype(np.float32)
        return img.astype(np.float32)
    except Exception as exc:
        show_warning(f"Failed to read {path.name}: {exc}")
        return None


def placeholder_array(shape: tuple = (512, 512)) -> np.ndarray:
    return np.zeros(shape, dtype=np.float32)


def apply_rotation(arr: np.ndarray, k: int) -> np.ndarray:
    return np.rot90(arr, k=-k)


# ---------------------------------------------------------------------------
# Napari viewer controller
# ---------------------------------------------------------------------------

class HistologyViewer:
    """Manages a napari viewer instance and all keyboard / mouse callbacks."""

    def __init__(self, loader: DatasetLoader, qc: QCStore,
                 counts: CellCountStore, subsets: SubsetStore,
                 counter_id: str,
                 initial_subset: Optional[str] = None):
        self.loader     = loader
        self.qc         = qc
        self.counts     = counts
        self.subsets    = subsets
        self.counter_id = counter_id

        # Normal-mode navigation
        self.rat_idx = 0
        self.sec_idx = 0

        # Subset navigation — when a subset is active, navigation uses it
        # instead of the full per-rat section lists.
        self._subset_name: Optional[str]               = None
        self._subset_seq:  list[SectionRecord]         = []
        self._subset_idx:  int                         = 0

        # Channel display
        self._overlay = False
        self._last_ch = "CH1"
        self._layers: dict[str, napari.layers.Image] = {}

        # Rotation
        self._rotation_k = 0

        # Quantification mode
        self._quant_mode              = False
        self._quant_sequence: list[SectionRecord] = []
        self._quant_stems: list[str]  = []
        self._quant_idx               = 0
        self._current_sec_completed   = False

        # One Points layer per channel (None until quant mode is entered)
        self._marker_layers: dict[str, Optional[napari.layers.Points]] = {
            ch: None for ch in CHANNELS}

        # ── Cross-rat comparison ──────────────────────────────────────
        # Pinned reference section (set with 'c')
        self._cmp_ref: Optional[SectionRecord] = None
        # True = currently showing the pinned reference; False = current section
        self._cmp_showing_ref: bool = False

        self.viewer = napari.Viewer(title="Histology QC Viewer")
        self._init_layers()
        self._bind_keys()
        self._bind_mouse()

        # Load initial subset if requested
        if initial_subset:
            self._load_subset_by_name(initial_subset, quiet=False)

        self._load_section()

    # ------------------------------------------------------------------
    # Navigation source helpers
    # ------------------------------------------------------------------

    @property
    def _in_subset(self) -> bool:
        return self._subset_name is not None and bool(self._subset_seq)

    @property
    def sections(self) -> list[SectionRecord]:
        if self._quant_mode:
            return self._quant_sequence
        if self._in_subset:
            return self._subset_seq
        return self.loader.get_sections(self.loader.rats[self.rat_idx])

    @property
    def current_section(self) -> SectionRecord:
        if self._quant_mode:
            return self._quant_sequence[self._quant_idx]
        if self._in_subset:
            return self._subset_seq[self._subset_idx]
        return self.loader.get_sections(self.loader.rats[self.rat_idx])[self.sec_idx]

    @property
    def _current_sec_idx(self) -> int:
        if self._quant_mode:
            return self._quant_idx
        if self._in_subset:
            return self._subset_idx
        return self.sec_idx

    # ------------------------------------------------------------------
    # Layer initialisation
    # ------------------------------------------------------------------

    def _init_layers(self) -> None:
        blank = placeholder_array()
        for ch in CHANNELS:
            layer = self.viewer.add_image(
                blank,
                name=f"{ch} - {CH_LABELS[ch]}",
                colormap=CH_COLORMAPS[ch],
                blending="additive",
                visible=(ch == "CH1"),
            )
            self._layers[ch] = layer

    def _active_channel(self) -> str:
        """
        Return the channel key (CH1-CH4) of the currently selected layer,
        or CH1 as fallback.
        """
        sel = self.viewer.layers.selection.active
        if sel is not None:
            for ch, layer in self._layers.items():
                if layer is sel:
                    return ch
        # Fallback: use the currently visible single-channel layer
        if not self._overlay:
            return self._last_ch
        return "CH1"

    # ------------------------------------------------------------------
    # Marker layer helpers (one per channel)
    # ------------------------------------------------------------------

    def _ensure_marker_layer(self, ch: str) -> napari.layers.Points:
        ml = self._marker_layers.get(ch)
        if ml is None or ml not in self.viewer.layers:
            color = CH_MARKER_COLORS[ch]
            ml = self.viewer.add_points(
                np.empty((0, 2), dtype=float),
                name=f"Markers {ch} - {CH_LABELS[ch]}",
                face_color=color,
                edge_color="white",
                edge_width=0.5,
                size=18,
                opacity=0.9,
                symbol="disc",
            )
            self._marker_layers[ch] = ml
        return ml

    def _clear_all_marker_layers(self) -> None:
        for ch in CHANNELS:
            ml = self._marker_layers.get(ch)
            if ml is not None and ml in self.viewer.layers:
                ml.data = np.empty((0, 2), dtype=float)

    def _remove_all_marker_layers(self) -> None:
        for ch in CHANNELS:
            ml = self._marker_layers.get(ch)
            if ml is not None and ml in self.viewer.layers:
                self.viewer.layers.remove(ml)
            self._marker_layers[ch] = None

    def _add_point_to_layer(self, ch: str, coord_y: float, coord_x: float) -> None:
        ml       = self._ensure_marker_layer(ch)
        new_pt   = np.array([[coord_y, coord_x]])
        existing = ml.data
        ml.data  = new_pt if existing.shape[0] == 0 else np.vstack([existing, new_pt])

    def _remove_last_point_from_layer(self, ch: str) -> None:
        ml = self._marker_layers.get(ch)
        if ml is not None and ml in self.viewer.layers and ml.data.shape[0] > 0:
            ml.data = ml.data[:-1]

    def _tint_all_markers_done(self) -> None:
        """Turn all marker layers to white/faded when section is completed."""
        for ch in CHANNELS:
            ml = self._marker_layers.get(ch)
            if ml is not None and ml in self.viewer.layers:
                n = ml.data.shape[0]
                if n > 0:
                    ml.face_color = np.repeat(MARKER_COLOR_DONE, n, axis=0)

    # ------------------------------------------------------------------
    # Section loading
    # ------------------------------------------------------------------

    def _load_section(self) -> None:
        # If we were previewing the pinned reference, snap back to current section.
        if self._cmp_showing_ref:
            self._cmp_showing_ref = False

        sec       = self.current_section
        anns      = self.qc.get(sec.rat, sec.stem_ch1)
        ref_shape = None

        for ch in CHANNELS:
            arr = load_png_as_array(sec.png_paths[ch])
            if arr is None:
                arr = (placeholder_array() if ref_shape is None
                       else placeholder_array(ref_shape))
            else:
                ref_shape = arr.shape
            arr   = apply_rotation(arr, self._rotation_k)
            layer = self._layers[ch]
            layer.data = arr
            if arr.max() > 0:
                layer.contrast_limits = (0, float(arr.max()))

        if self._quant_mode:
            self._clear_all_marker_layers()
            for ch in CHANNELS:
                self._ensure_marker_layer(ch)
            self._current_sec_completed = False

        self._update_title()
        self._update_visibility()
        self._print_section_info(anns)

    def _print_section_info(self, anns: dict) -> None:
        sec   = self.current_section
        idx   = self._current_sec_idx
        total = len(self.sections)
        if self._quant_mode:
            ch_counts = self.counts.pending_channel_counts(sec.rat, sec.stem_ch1)
            total_n   = self.counts.pending_total(sec.rat, sec.stem_ch1)
            done      = self._current_sec_completed
            ch_str    = "  ".join(
                f"{CH_LABELS[ch]}={ch_counts[ch]}" for ch in CHANNELS)
            print(
                f"\n{'─'*62}\n"
                f"  [QUANTIFICATION]  counter: {self.counter_id}\n"
                f"  Image  : {idx+1} of {total}  (identity hidden)\n"
                f"  Counts : {ch_str}  |  total={total_n}\n"
                f"  Status : {'COMPLETE - Ctrl+S to advance' if done else 'counting...'}\n"
                f"  Rot    : {self._rotation_k * 90} deg\n"
                f"  Click layer to select channel for counting\n"
                f"  Ctrl+S=save+complete  Delete=skip  Backspace=undo\n"
                f"{'─'*62}"
            )
        else:
            src  = f"SUBSET:{self._subset_name}" if self._in_subset else sec.rat
            excl = self.counts.is_excluded(sec.stem_ch1)
            flags = (f"mirror={anns['mirror']}  rotate={anns['rotate']}  "
                     f"quantify={anns['quantify']}"
                     + ("  [EXCLUDED]" if excl else ""))
            print(
                f"\n{'─'*62}\n"
                f"  Rat    : {sec.rat}  ({src})\n"
                f"  XY     : {sec.xy_label}  [{idx+1}/{total}]\n"
                f"  Rot    : {self._rotation_k * 90} deg\n"
                f"  Flags  : {flags}\n"
                f"  Notes  : {anns['notes'] or '(no notes)'}\n"
                f"{'─'*62}"
            )

    # ------------------------------------------------------------------
    # Visibility
    # ------------------------------------------------------------------

    def _update_visibility(self) -> None:
        if self._overlay:
            for layer in self._layers.values():
                layer.visible = True
        else:
            for ch, layer in self._layers.items():
                layer.visible = (ch == self._last_ch)

    def _show_channel(self, ch: str) -> None:
        self._overlay = False
        self._last_ch = ch
        self._update_visibility()
        show_info(f"Showing {ch} - {CH_LABELS[ch]}")

    def _toggle_overlay(self) -> None:
        self._overlay = not self._overlay
        self._update_visibility()
        show_info("OVERLAY" if self._overlay else f"Single: {self._last_ch}")

    # ------------------------------------------------------------------
    # Window title
    # ------------------------------------------------------------------

    def _update_title(self) -> None:
        sec   = self.current_section
        idx   = self._current_sec_idx
        total = len(self.sections)
        if self._quant_mode:
            ch_counts = self.counts.pending_channel_counts(sec.rat, sec.stem_ch1)
            total_n   = self.counts.pending_total(sec.rat, sec.stem_ch1)
            done      = self._current_sec_completed
            ch_str    = " ".join(f"{CH_LABELS[ch][:3]}={ch_counts[ch]}"
                                  for ch in CHANNELS)
            state     = "DONE" if done else ch_str
            self.viewer.title = (
                f"HistQC | QUANT [{self.counter_id}] | "
                f"{idx+1}/{total} | {state} | total={total_n}"
            )
        else:
            anns  = self.qc.get(sec.rat, sec.stem_ch1)
            excl  = self.counts.is_excluded(sec.stem_ch1)
            flags = " ".join(f"[{f.upper()}]"
                             for f in ("mirror", "rotate", "quantify") if anns[f])
            if excl:
                flags = "[EXCLUDED] " + flags
            src   = f"[SUBSET:{self._subset_name}] " if self._in_subset else ""
            title = (f"HistQC | {src}{sec.rat} | {sec.xy_label} | "
                     f"Section {idx+1}/{total}")
            if flags:
                title += f"  {flags}"
            self.viewer.title = title

    # ------------------------------------------------------------------
    # Rotation
    # ------------------------------------------------------------------

    def _rotate_image(self) -> None:
        self._rotation_k = (self._rotation_k + 1) % 4
        self._load_section()
        show_info(f"Rotated {self._rotation_k * 90} deg CW")

    # ------------------------------------------------------------------
    # Quantification mode -- enter / exit
    # ------------------------------------------------------------------

    def _build_fresh_sequence(self) -> tuple[list[SectionRecord], list[str]]:
        done    = self.counts.completed_images()
        pending = [s for s in self.loader.all_sections_flat()
                   if s.stem_ch1 not in done]
        random.shuffle(pending)
        return pending, [s.stem_ch1 for s in pending]

    def _enter_quantification_mode(self) -> None:
        if self._quant_mode:
            return
        checkpoint = self.counts.load_progress()
        resumed    = False

        if checkpoint:
            stems_saved = checkpoint["shuffle_stems"]
            saved_idx   = checkpoint["current_idx"]
            done        = self.counts.completed_images()
            remaining   = [st for st in stems_saved if st not in done]
            sequence    = [s for s in
                           (self.loader.section_by_stem(st) for st in remaining)
                           if s is not None]
            if sequence:
                done_before = sum(1 for st in stems_saved[:saved_idx] if st in done)
                resume_pos  = min(max(0, saved_idx - done_before), len(sequence) - 1)
                self._quant_sequence = sequence
                self._quant_stems    = [s.stem_ch1 for s in sequence]
                self._quant_idx      = resume_pos
                resumed = True
                print(
                    f"\n{'='*60}\n"
                    f"  QUANTIFICATION MODE -- RESUMED\n"
                    f"  Counter    : {self.counter_id}\n"
                    f"  Checkpoint : {checkpoint['saved_at']}\n"
                    f"  Remaining  : {len(sequence)} images\n"
                    f"  Resuming at image {resume_pos + 1}\n"
                    f"{'='*60}"
                )
            else:
                print("[INFO] Checkpoint found but all images complete; fresh session.")

        if not resumed:
            sequence, stems = self._build_fresh_sequence()
            if not sequence:
                show_warning(f"No remaining images for counter '{self.counter_id}'.")
                return
            self._quant_sequence = sequence
            self._quant_stems    = stems
            self._quant_idx      = 0
            print(
                f"\n{'='*60}\n"
                f"  QUANTIFICATION MODE -- NEW SESSION\n"
                f"  Counter  : {self.counter_id}\n"
                f"  Images   : {len(sequence)} (shuffled, identity hidden)\n"
                f"  Rats     : {len(self.loader.rats)}\n"
                f"  Right-click a layer to count cells for that channel\n"
                f"  Ctrl+S = save+complete  |  Delete = skip  |  q = exit\n"
                f"{'='*60}"
            )

        self._quant_mode            = True
        self._current_sec_completed = False
        self._rotation_k            = 0
        self.counts.save_progress(self._quant_stems, self._quant_idx)
        show_info(f"QUANT ON [{self.counter_id}] — click a layer, right-click to count")
        for ch in CHANNELS:
            self._ensure_marker_layer(ch)
        self._load_section()

    def _exit_quantification_mode(self) -> None:
        if not self._quant_mode:
            return
        sec = self.current_section
        n   = self.counts.pending_total(sec.rat, sec.stem_ch1)
        if not self._current_sec_completed and n > 0:
            show_info(f"Exiting — {n} marker(s) on current image NOT saved "
                      "(section not completed, will re-appear next session).")
        self._quant_mode = False
        self._remove_all_marker_layers()
        self.counts.delete_progress()
        self.counts.emergency_save_pending()
        show_info("Quantification mode OFF")
        print(f"\n{'='*60}\n"
              f"  QUANTIFICATION ENDED — Counter: {self.counter_id}\n"
              f"{'='*60}")
        self._rotation_k = 0
        self._load_section()

    def _toggle_quant_mode(self) -> None:
        if self._quant_mode:
            self._exit_quantification_mode()
        else:
            self._enter_quantification_mode()

    # ------------------------------------------------------------------
    # Quant actions
    # ------------------------------------------------------------------

    def _quant_save_complete(self) -> None:
        """Ctrl+S in quant mode: save current section and advance."""
        if not self._quant_mode:
            # In normal mode Ctrl+S saves QC annotations
            self.qc.save()
            return

        sec = self.current_section

        if self._current_sec_completed:
            # Already done — just advance
            self._go_section(+1)
            return

        ch_counts = self.counts.complete_section(sec.rat, sec.stem_ch1, sec.xy_label)
        total     = sum(ch_counts.values())
        self._current_sec_completed = True
        self._tint_all_markers_done()

        ch_str = "  ".join(f"{CH_LABELS[ch]}={ch_counts[ch]}" for ch in CHANNELS)
        show_info(f"SAVED — {ch_str}  |  total={total}")
        print(f"[QUANT] Saved: {ch_str}  total={total}  "
              f"rat={sec.rat}  xy={sec.xy_label}")
        self._update_title()
        # Auto-advance after short pause so user sees the "DONE" state
        self._go_section(+1)

    def _quant_skip_section(self) -> None:
        if not self._quant_mode:
            return
        if self._current_sec_completed:
            show_info("Section already saved; use -> to navigate.")
            return
        sec = self.current_section
        n   = self.counts.pending_total(sec.rat, sec.stem_ch1)
        self.counts.discard_pending(sec.rat, sec.stem_ch1)
        self._clear_all_marker_layers()
        show_info(f"Skipped ({n} marker(s) discarded) — image re-queued.")
        print(f"[QUANT] Skipped — {n} click(s) discarded.")
        self._go_section(+1)

    def _quant_undo_last(self) -> None:
        if not self._quant_mode:
            return
        if self._current_sec_completed:
            show_info("Section already saved; cannot undo.")
            return
        sec = self.current_section
        removed_ch, ch_counts = self.counts.undo_last_click(sec.rat, sec.stem_ch1)
        if removed_ch:
            self._remove_last_point_from_layer(removed_ch)
            total = sum(ch_counts.values())
            show_info(f"Undo ({CH_LABELS[removed_ch]}) — total remaining: {total}")
            print(f"[QUANT] Undo {CH_LABELS[removed_ch]}. "
                  + "  ".join(f"{CH_LABELS[ch]}={ch_counts[ch]}" for ch in CHANNELS))
        else:
            show_info("Nothing to undo.")
        self._update_title()

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _go_section(self, delta: int) -> None:
        if self._quant_mode:
            total = len(self._quant_sequence)
            if not total:
                return
            self._quant_idx = (self._quant_idx + delta) % total
            self._current_sec_completed = False
            self.counts.save_progress(self._quant_stems, self._quant_idx)
        elif self._in_subset:
            total = len(self._subset_seq)
            if not total:
                return
            self._subset_idx = (self._subset_idx + delta) % total
        else:
            secs = self.loader.get_sections(self.loader.rats[self.rat_idx])
            if not secs:
                return
            self.sec_idx = (self.sec_idx + delta) % len(secs)
        self._rotation_k = 0
        self._load_section()

    def _go_rat(self, delta: int) -> None:
        if self._quant_mode:
            show_info("Rat navigation disabled in quantification mode.")
            return
        if self._in_subset:
            show_info(f"Navigating within subset '{self._subset_name}'. "
                      "Press Ctrl+L and enter blank name to return to full dataset.")
            return
        self.qc.save()
        self.rat_idx = (self.rat_idx + delta) % len(self.loader.rats)
        self.sec_idx = 0
        show_info(f"Switched to {self.loader.rats[self.rat_idx]}")
        self._rotation_k = 0
        self._load_section()

    # ------------------------------------------------------------------
    # Subset management
    # ------------------------------------------------------------------

    def _add_to_subset(self) -> None:
        """Prompt for subset name and add current section to it."""
        sec = self.current_section
        print(f"\n[SUBSET] Current: {sec.rat} / {sec.xy_label}")
        print("[SUBSET] Enter subset name to add to (e.g. for_figures, "
              "for_student_quant, comparison_XYZ):")
        try:
            name = input("  >> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("[SUBSET] Cancelled.")
            return
        if not name:
            show_info("No subset name entered; cancelled.")
            return

        # Optional description on first creation
        existing = self.subsets.load(name)
        desc     = existing.get("description", "") if existing else ""
        if not existing:
            print(f"[SUBSET] Creating new subset '{name}'. "
                  "Enter a description (blank = none):")
            try:
                desc = input("  >> ").strip()
            except (EOFError, KeyboardInterrupt):
                desc = ""

        n = self.subsets.add_section(name, sec.stem_ch1,
                                     description=desc,
                                     created_by=self.counter_id)
        show_info(f"Added to subset '{name}' ({n} sections total)")
        print(f"[SUBSET] '{name}' now has {n} section(s).")

    def _load_subset_by_name(self, name: str, quiet: bool = False) -> bool:
        """Activate a subset as the navigation source. Empty name = deactivate."""
        if not name:
            self._subset_name = None
            self._subset_seq  = []
            self._subset_idx  = 0
            show_info("Subset deactivated — navigating full dataset.")
            return True

        data = self.subsets.load(name)
        if data is None:
            show_warning(f"Subset '{name}' not found.")
            return False

        stems = data.get("sections", [])
        seq   = [s for s in (self.loader.section_by_stem(st) for st in stems)
                 if s is not None]
        if not seq:
            show_warning(f"Subset '{name}' has no loadable sections.")
            return False

        self._subset_name = name
        self._subset_seq  = seq
        self._subset_idx  = 0
        if not quiet:
            show_info(f"Loaded subset '{name}' — {len(seq)} sections")
            print(
                f"\n[SUBSET] '{name}' loaded\n"
                f"         {len(seq)} section(s)  |  "
                f"desc: {data.get('description', '(none)')}\n"
                f"         created by: {data.get('created_by', '?')}  "
                f"at: {data.get('created_at', '?')}"
            )
        return True

    def _prompt_load_subset(self) -> None:
        """Ctrl+L: prompt for a subset name and activate it."""
        subs = self.subsets.list_subsets()
        if subs:
            print("\n[SUBSET] Available subsets:")
            for s in subs:
                print(f"   {s['name']:30s}  {s['n_sections']:3d} sections  "
                      f"by {s['created_by'] or '?'}  |  {s['description'] or ''}")
        print("\n[SUBSET] Enter subset name to load (blank = return to full dataset):")
        try:
            name = input("  >> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("[SUBSET] Cancelled.")
            return
        self._load_subset_by_name(name)
        if not self._quant_mode:
            self._load_section()

    def _list_subsets(self) -> None:
        """Ctrl+A: print all subsets to console."""
        subs = self.subsets.list_subsets()
        if not subs:
            print("\n[SUBSET] No subsets defined yet. Press 'a' to add sections.")
            return
        print(f"\n{'─'*62}")
        print(f"  {'NAME':<30} {'N':>4}  {'BY':<12}  DESCRIPTION")
        print(f"{'─'*62}")
        for s in subs:
            print(f"  {s['name']:<30} {s['n_sections']:>4}  "
                  f"{(s['created_by'] or '?'):<12}  {s['description'] or ''}")
        print(f"{'─'*62}")

    # ------------------------------------------------------------------
    # Post-hoc exclusion
    # ------------------------------------------------------------------

    def _toggle_exclusion(self) -> None:
        if self._quant_mode:
            show_info("Use normal mode to exclude sections.")
            return
        sec  = self.current_section
        excl = self.counts.is_excluded(sec.stem_ch1)
        if excl:
            new_state = self.counts.toggle_exclusion(sec.stem_ch1)
            show_info("Section RE-INCLUDED in exports.")
        else:
            print(f"\n[EXCLUDE] Reason for excluding {sec.xy_label} "
                  "(blank = no reason, Ctrl+C = cancel):")
            try:
                reason = input("  >> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("[EXCLUDE] Cancelled.")
                return
            new_state = self.counts.toggle_exclusion(sec.stem_ch1, reason=reason)
            show_info("Section EXCLUDED from exports.")
        if new_state is not None:
            self._update_title()
            self._print_section_info(self.qc.get(sec.rat, sec.stem_ch1))

    # ------------------------------------------------------------------
    # Per-rat export
    # ------------------------------------------------------------------

    def _export_per_rat(self) -> None:
        dest = self.counts.export_per_rat()
        show_info(f"Per-rat export -> {dest.name}")

    # ------------------------------------------------------------------
    # QC annotation helpers
    # ------------------------------------------------------------------

    def _toggle_flag(self, flag: str) -> None:
        if self._quant_mode:
            show_info("QC flags disabled in quantification mode.")
            return
        sec     = self.current_section
        new_val = self.qc.toggle(sec.rat, sec.stem_ch1, flag)
        show_info(f"{flag.upper()} -> {'ON' if new_val else 'OFF'}")
        self._update_title()

    def _enter_notes(self) -> None:
        if self._quant_mode:
            show_info("Notes disabled in quantification mode.")
            return
        sec  = self.current_section
        anns = self.qc.get(sec.rat, sec.stem_ch1)
        print(f"\n[NOTES] Current: {anns['notes'] or '(empty)'}")
        print("[NOTES] New notes (blank = keep, 'clear' = delete):")
        try:
            user_input = input("  >> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("[NOTES] Cancelled.")
            return
        if user_input.lower() == "clear":
            self.qc.set(sec.rat, sec.stem_ch1, notes="")
            show_info("Notes cleared.")
        elif user_input:
            self.qc.set(sec.rat, sec.stem_ch1, notes=user_input)
            show_info(f"Notes saved: {user_input[:40]}")
        else:
            show_info("Notes unchanged.")

    # ------------------------------------------------------------------
    # Cross-rat comparison
    # ------------------------------------------------------------------

    def _pin_comparison_ref(self) -> None:
        """'c' — pin the current section as the comparison reference (Rat A)."""
        if self._quant_mode:
            show_info("Comparison pinning disabled in quantification mode.")
            return
        sec = self.current_section
        self._cmp_ref = sec
        self._cmp_showing_ref = False
        show_info(f"[CMP] Pinned: {sec.rat}  {sec.xy_label}")
        print(
            f"\n{'─'*62}\n"
            f"  [COMPARE] Reference pinned\n"
            f"  Rat  : {sec.rat}\n"
            f"  XY   : {sec.xy_label}\n"
            f"  Stem : {sec.stem_ch1}\n"
            f"  Navigate to another rat/section, then press Tab to flip.\n"
            f"{'─'*62}"
        )

    def _toggle_comparison_view(self) -> None:
        """Tab — flip between the pinned reference and the current section."""
        if self._quant_mode:
            show_info("Comparison toggle disabled in quantification mode.")
            return
        if self._cmp_ref is None:
            show_info("[CMP] No reference pinned. Press 'c' on a section first.")
            return
        sec = self.current_section
        if self._cmp_ref.rat == sec.rat and self._cmp_ref.stem_ch1 == sec.stem_ch1:
            show_info("[CMP] Current section IS the pinned reference — navigate away first.")
            return

        self._cmp_showing_ref = not self._cmp_showing_ref
        target = self._cmp_ref if self._cmp_showing_ref else sec
        self._load_images_into_layers(target)
        label = f"[REF] {target.rat}  {target.xy_label}" if self._cmp_showing_ref \
                else f"[CUR] {target.rat}  {target.xy_label}"
        show_info(f"[CMP] {label}  |  Tab to flip  |  Ctrl+P to save pair")

    def _load_images_into_layers(self, sec: "SectionRecord") -> None:
        """Load a section's images into the existing napari layers (no new layers)."""
        anns      = self.qc.get(sec.rat, sec.stem_ch1)
        rot_k     = 1 if anns.get("rotate") else 0
        ref_shape = None
        for ch in CHANNELS:
            arr = load_png_as_array(sec.png_paths[ch])
            if arr is None:
                arr = (placeholder_array() if ref_shape is None
                       else placeholder_array(ref_shape))
            else:
                ref_shape = arr.shape
            arr = apply_rotation(arr, rot_k)
            layer = self._layers[ch]
            layer.data = arr
            if arr.max() > 0:
                layer.contrast_limits = (0, float(arr.max()))

    def _export_comparison_pair(self) -> None:
        """Ctrl+P — write a .txt with the two paired image stems."""
        if self._quant_mode:
            show_info("Comparison export disabled in quantification mode.")
            return
        if self._cmp_ref is None:
            show_info("[CMP] No reference pinned. Press 'c' on a section first.")
            return
        sec = self.current_section
        if self._cmp_ref.rat == sec.rat and self._cmp_ref.stem_ch1 == sec.stem_ch1:
            show_info("[CMP] Reference and current section are the same — "
                      "navigate to a different section first.")
            return

        now   = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"comparison_pair_{now}.txt"
        dest  = BASE_DIR / fname

        lines = [
            f"# Histology comparison pair",
            f"# Saved: {datetime.now().isoformat(timespec='seconds')}",
            f"# Counter: {self.counter_id}",
            f"",
            f"[Reference]",
            f"rat        = {self._cmp_ref.rat}",
            f"xy_label   = {self._cmp_ref.xy_label}",
            f"stem_ch1   = {self._cmp_ref.stem_ch1}",
            f"png_CH1    = {self._cmp_ref.png_paths['CH1']}",
            f"png_CH2    = {self._cmp_ref.png_paths['CH2']}",
            f"png_CH3    = {self._cmp_ref.png_paths['CH3']}",
            f"png_CH4    = {self._cmp_ref.png_paths['CH4']}",
            f"",
            f"[Current]",
            f"rat        = {sec.rat}",
            f"xy_label   = {sec.xy_label}",
            f"stem_ch1   = {sec.stem_ch1}",
            f"png_CH1    = {sec.png_paths['CH1']}",
            f"png_CH2    = {sec.png_paths['CH2']}",
            f"png_CH3    = {sec.png_paths['CH3']}",
            f"png_CH4    = {sec.png_paths['CH4']}",
        ]

        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text("\n".join(lines) + "\n")
            show_info(f"[CMP] Pair saved -> {fname}")
            print(
                f"\n{'─'*62}\n"
                f"  [COMPARE] Pair exported\n"
                f"  File      : {dest}\n"
                f"  Reference : {self._cmp_ref.rat}  {self._cmp_ref.xy_label}\n"
                f"  Current   : {sec.rat}  {sec.xy_label}\n"
                f"{'─'*62}"
            )
        except Exception as exc:
            show_warning(f"[CMP] Could not save pair: {exc}")
            print(f"[ERROR] Comparison export failed: {exc}")

    # ------------------------------------------------------------------
    # Key bindings
    # ------------------------------------------------------------------

    def _bind_keys(self) -> None:
        v = self.viewer

        # Navigation
        v.bind_key("Right",      lambda _: self._go_section(+1))
        v.bind_key("Left",       lambda _: self._go_section(-1))
        v.bind_key("]",          lambda _: self._go_rat(+1))
        v.bind_key("[",          lambda _: self._go_rat(-1))

        # Channel / display
        v.bind_key("1",          lambda _: self._show_channel("CH1"))
        v.bind_key("2",          lambda _: self._show_channel("CH2"))
        v.bind_key("3",          lambda _: self._show_channel("CH3"))
        v.bind_key("4",          lambda _: self._show_channel("CH4"))
        v.bind_key("Space",      lambda _: self._toggle_overlay())

        # QC flags
        v.bind_key("m",          lambda _: self._toggle_flag("mirror"))
        v.bind_key("r",          lambda _: self._rotate_image())

        # q = toggle quant mode (enter if off, exit if on)
        v.bind_key("q",          lambda _: self._toggle_quant_mode())

        # In-quant controls (overwrite=True for keys napari may reserve)
        v.bind_key("Control-s",  lambda _: self._quant_save_complete(),
                   overwrite=True)
        v.bind_key("Delete",     lambda _: self._quant_skip_section(),
                   overwrite=True)
        v.bind_key("Backspace",  lambda _: self._quant_undo_last(),
                   overwrite=True)

        # Post-hoc exclusion
        v.bind_key("x",          lambda _: self._toggle_exclusion())

        # Subsets
        v.bind_key("a",          lambda _: self._add_to_subset())
        v.bind_key("Control-l",  lambda _: self._prompt_load_subset())
        v.bind_key("Control-a",  lambda _: self._list_subsets())

        # Export
        v.bind_key("Control-e",  lambda _: self._export_per_rat())

        # Notes  (Ctrl+S in normal mode = QC save, handled inside _quant_save_complete)
        v.bind_key("n",          lambda _: self._enter_notes())

        # Cross-rat comparison
        v.bind_key("c",          lambda _: self._pin_comparison_ref())
        v.bind_key("Tab",        lambda _: self._toggle_comparison_view(),
                   overwrite=True)
        v.bind_key("Control-p",  lambda _: self._export_comparison_pair(),
                   overwrite=True)

    # ------------------------------------------------------------------
    # Mouse callback
    # ------------------------------------------------------------------

    def _bind_mouse(self) -> None:
        viewer = self.viewer

        @viewer.mouse_drag_callbacks.append
        def on_mouse_drag(v, event):
            if event.type != "mouse_press" or event.button != 2:
                return

            if not self._quant_mode:
                # Normal mode: informational log
                try:
                    coords = tuple(round(c, 2) for c in event.position)
                    sec    = self.current_section
                    anns   = self.qc.get(sec.rat, sec.stem_ch1)
                    print(f"\n[CLICK] Coords={coords}  rat={sec.rat}  "
                          f"xy={sec.xy_label}  anns={anns}")
                except Exception as exc:
                    print(f"[WARN] Mouse callback error: {exc}")
                return

            if self._current_sec_completed:
                show_info("Section already saved. Press Ctrl+S or -> to advance.")
                return

            try:
                pos     = event.position
                coord_y = float(pos[-2])
                coord_x = float(pos[-1])
                sec     = self.current_section
                ch      = self._active_channel()

                ch_counts = self.counts.add_click(
                    rat              = sec.rat,
                    image            = sec.stem_ch1,
                    xy_label         = sec.xy_label,
                    channel          = ch,
                    coord_y          = coord_y,
                    coord_x          = coord_x,
                    rotation_applied = self._rotation_k * 90,
                )
                self._add_point_to_layer(ch, coord_y, coord_x)
                self._update_title()
                total = sum(ch_counts.values())
                print(f"[QUANT] {CH_LABELS[ch]} marker at "
                      f"({coord_y:.1f}, {coord_x:.1f})  "
                      + "  ".join(f"{CH_LABELS[c]}={ch_counts[c]}" for c in CHANNELS)
                      + f"  total={total}")

            except Exception as exc:
                print(f"[WARN] Quantification click error: {exc}")

        @viewer.mouse_double_click_callbacks.append
        def on_double_click(v, event):
            if self._quant_mode:
                return
            try:
                coords = tuple(round(c, 2) for c in event.position)
                sec    = self.current_section
                anns   = self.qc.get(sec.rat, sec.stem_ch1)
                print(f"\n[DBLCLICK] Coords={coords}  rat={sec.rat}  "
                      f"xy={sec.xy_label}  stem={sec.stem_ch1}  anns={anns}")
            except Exception as exc:
                print(f"[WARN] Mouse callback error: {exc}")

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self) -> None:
        src = (f"SUBSET: {self._subset_name}" if self._in_subset
               else "full dataset")
        print(
            "\n"
            "╔═══════════════════════════════════════════════════════════╗\n"
            "║         Histology QC Viewer  --  Key Map                  ║\n"
            "╠═══════════════════════════════════════════════════════════╣\n"
            "║  NORMAL MODE                                               ║\n"
            "║  <- / ->     Previous / Next section                      ║\n"
            "║  [ / ]       Previous / Next rat  (full dataset only)     ║\n"
            "║  1-4         Show CH1 / CH2 / CH3 / CH4 only              ║\n"
            "║  Space       Toggle overlay <-> single channel            ║\n"
            "║  m           Toggle MIRROR flag                           ║\n"
            "║  r           Rotate image 90 deg CW                      ║\n"
            "║  q           Enter / Exit QUANTIFICATION MODE             ║\n"
            "║  x           Toggle EXCLUDED flag (post-hoc)              ║\n"
            "║  a           Add current section to a subset / album      ║\n"
            "║  n           Edit NOTES (console)                         ║\n"
            "║  Ctrl+L      Load a subset as navigation source           ║\n"
            "║  Ctrl+A      List all subsets                             ║\n"
            "║  Ctrl+E      Export per-rat counts CSV                    ║\n"
            "║  Ctrl+S      Force-save QC annotations                    ║\n"
            "╠═══════════════════════════════════════════════════════════╣\n"
            "║  CROSS-RAT COMPARISON                                      ║\n"
            "║  c           Pin current section as reference (Rat A)     ║\n"
            "║  Tab         Flip between reference and current section    ║\n"
            "║  Ctrl+P      Save comparison pair to .txt                 ║\n"
            "╠═══════════════════════════════════════════════════════════╣\n"
            "║  QUANTIFICATION MODE                                       ║\n"
            "║  Click a layer first to select the channel to count in    ║\n"
            "║  Right-click        Place a marker on the active channel  ║\n"
            "║  Backspace          Undo last marker (any channel)        ║\n"
            "║  Ctrl+S             Save & complete section, advance      ║\n"
            "║  Delete             Skip section (discard, re-queue)      ║\n"
            "║  <- / ->            Navigate without completing           ║\n"
            "║  q                  Exit quantification mode              ║\n"
            "╚═══════════════════════════════════════════════════════════╝\n"
           f"  Counter : {self.counter_id}\n"
           f"  Source  : {src}\n"
        )
        try:
            napari.run()
        finally:
            self.qc.save()
            if self._quant_mode:
                self.counts.emergency_save_pending()
                self.counts.save_progress(self._quant_stems, self._quant_idx)


# ---------------------------------------------------------------------------
# Counter ID prompt
# ---------------------------------------------------------------------------

def prompt_counter_id(provided: Optional[str]) -> str:
    if provided:
        cid = re.sub(r"[^\w\-]", "_", provided.strip())[:32]
        if cid:
            print(f"[INFO] Counter ID: {cid}")
            return cid
    print("\n" + "-" * 50)
    print("  Enter your name or initials (counter ID).")
    print("  This is used for session resume and multi-counter support.")
    print("-" * 50)
    while True:
        try:
            raw = input("  Name / initials: ").strip()
        except (EOFError, KeyboardInterrupt):
            return "anonymous"
        cid = re.sub(r"[^\w\-]", "_", raw)[:32]
        if cid:
            print(f"  Counter ID: {cid}\n")
            return cid
        print("  Please enter a non-empty identifier.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    global BASE_DIR, PNG_DIR, SORT_DIR, DATASET_DIR, QC_CSV, COUNTS_CSV

    parser = argparse.ArgumentParser(
        description="Napari histology QC viewer with blinded per-channel cell counting.")
    parser.add_argument("base_dir", nargs="?",
                        help="Override BASE_DIR")
    parser.add_argument("--counter", metavar="NAME",
                        help="Counter ID (skips prompt)")
    parser.add_argument("--export", action="store_true",
                        help="Export per-rat counts CSV and exit")
    parser.add_argument("--export-dlc", action="store_true",
                        help="Export DLC coordinate files and exit")
    parser.add_argument("--exclude", metavar="IMAGE_STEM",
                        help="Toggle excluded flag for a section and exit")
    parser.add_argument("--exclude-reason", metavar="REASON", default="",
                        help="Reason string for --exclude")
    parser.add_argument("--subset", metavar="NAME",
                        help="Load subset NAME as navigation source on startup")
    args = parser.parse_args()

    if args.base_dir:
        BASE_DIR    = Path(args.base_dir).expanduser().resolve()
        PNG_DIR     = BASE_DIR / "PNG_fullsize"
        SORT_DIR    = BASE_DIR / "features" / "sorting"
        DATASET_DIR = BASE_DIR / "dataset"
        QC_CSV      = BASE_DIR / "qc_annotations.csv"
        COUNTS_CSV  = BASE_DIR / "cell_counts.csv"
        print(f"[INFO] Using base directory: {BASE_DIR}")

    summary_path = COUNTS_CSV.parent / (COUNTS_CSV.stem + "_summary.csv")
    per_rat_path = COUNTS_CSV.parent / (COUNTS_CSV.stem + "_per_rat.csv")

    if args.export:
        export_per_rat_standalone(summary_path, per_rat_path)
        return

    if args.export_dlc:
        counter_id = prompt_counter_id(args.counter)
        store      = CellCountStore(path=COUNTS_CSV, counter_id=counter_id)
        store.export_dlc()
        return

    if args.exclude:
        counter_id = prompt_counter_id(args.counter)
        store      = CellCountStore(path=COUNTS_CSV, counter_id=counter_id)
        new_state  = store.toggle_exclusion(args.exclude,
                                            reason=args.exclude_reason)
        if new_state is not None:
            action = "EXCLUDED" if new_state else "RE-INCLUDED"
            print(f"[INFO] '{args.exclude}' is now {action} "
                  f"for counter '{counter_id}'.")
            store.export_per_rat(per_rat_path)
        return

    for d, label in [(PNG_DIR, "PNG"), (SORT_DIR, "sorting")]:
        if not d.exists():
            print(f"[WARN] Expected directory not found: {d}  ({label})")

    try:
        loader = DatasetLoader(sort_dir=SORT_DIR)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    counter_id = prompt_counter_id(args.counter)
    qc         = QCStore(path=QC_CSV)
    counts     = CellCountStore(path=COUNTS_CSV, counter_id=counter_id)
    subsets    = SubsetStore(base_dir=BASE_DIR)

    viewer_ctrl = HistologyViewer(
        loader         = loader,
        qc             = qc,
        counts         = counts,
        subsets        = subsets,
        counter_id     = counter_id,
        initial_subset = args.subset,
    )
    viewer_ctrl.run()


if __name__ == "__main__":
    main()
