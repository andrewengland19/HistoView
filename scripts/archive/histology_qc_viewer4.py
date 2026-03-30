"""
histology_qc_viewer.py
======================
Napari-based keyboard-driven QC viewer for rat histology datasets.

Features
--------
  * Navigate sections with arrow keys; switch rats with [ / ]
  * Toggle individual channels (1-4) or full overlay (Space)
  * Rotate image 90 deg CW with 'r'; toggle mirror flag with 'm'
  * Right-click anywhere to place a colour-coded dot on the active channel
    and record the cell as counted
  * Backspace undoes the last marker
  * Ctrl+S saves QC annotations
  * On exit, all click counts are saved to cell_counts.csv (rat, xy, layer,
    coords, timestamp)

Directory layout assumed
------------------------
~/Microscopy/Cohort1_TPH2/
    PNG_fullsize/<ratID>/CH1 ... CH4/
    features/sorting/predicted_order_<ratID>.csv  (column: "image")

Output files (relative to BASE_DIR)
-------------------------------------
  qc_annotations.csv     QC flags / notes per section
  cell_counts.csv        One row per right-click:
                           rat, xy_label, layer, coord_y, coord_x,
                           rotation_applied, timestamp

Keyboard map
------------
  <- / ->     Previous / Next section
  [ / ]       Previous / Next rat
  1-4         Show CH1 / CH2 / CH3 / CH4 only
  Space       Toggle overlay (all channels) <-> last single-channel
  m           Toggle MIRROR flag
  r           Rotate image 90 degrees CW
  Backspace   Undo last marker
  Ctrl+S      Save QC annotations
  n           Edit NOTES (console)
  c           Pin current section as comparison reference
  Tab         Flip between pinned reference and current section
  Ctrl+P      Export the two paired image names to a .txt file
  a           Add current section to a named subset
  Ctrl+L      Load a subset as the navigation source
  Ctrl+A      List all subsets

CLI usage
---------
  python histology_qc_viewer.py [BASE_DIR]

Author: generated for Cohort1_TPH2 pipeline
"""

from __future__ import annotations

import argparse
import json
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

BASE_DIR = Path.home() / "Microscopy" / "Cohort1_TPH2"
PNG_DIR  = BASE_DIR / "PNG_fullsize"
SORT_DIR = BASE_DIR / "features" / "sorting"
QC_CSV   = BASE_DIR / "qc_annotations.csv"
COUNTS_CSV = BASE_DIR / "cell_counts.csv"

CHANNELS     = ["CH1", "CH2", "CH3", "CH4"]
CH_LABELS    = {"CH1": "DAPI", "CH2": "TPH2/GFP", "CH3": "RFP", "CH4": "Cy5"}
CH_COLORMAPS = {"CH1": "blue",  "CH2": "green",    "CH3": "red",  "CH4": "magenta"}

# Marker colours per channel (RGBA 0-1)
CH_MARKER_COLORS: dict[str, np.ndarray] = {
    "CH1": np.array([[0.4, 0.6, 1.0, 0.95]]),   # pale blue  — DAPI
    "CH2": np.array([[0.2, 1.0, 0.2, 0.95]]),   # green      — GFP
    "CH3": np.array([[1.0, 0.3, 0.3, 0.95]]),   # red        — RFP
    "CH4": np.array([[0.9, 0.2, 0.9, 0.95]]),   # magenta    — Cy5
}


# ---------------------------------------------------------------------------
# Data-loader abstraction
# ---------------------------------------------------------------------------

class SectionRecord:
    """One XY section for one rat, with derived paths for all channels."""

    __slots__ = ("rat", "stem_ch1", "stems", "png_paths", "xy_label")

    def __init__(self, rat: str, stem_ch1: str):
        self.rat      = rat
        self.stem_ch1 = stem_ch1
        self.xy_label = self._parse_xy(stem_ch1)
        self.stems    = {ch: self._derive_stem(stem_ch1, ch) for ch in CHANNELS}
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
        self.sort_dir = sort_dir
        self.rats: list[str] = []
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

    COLUMNS = ["rat", "image", "mirror", "rotate", "notes"]
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
            return {k: (str(row.get(k, "")) if k == "notes"
                        else bool(row.get(k, False)))
                    for k in ("mirror", "rotate", "notes")}
        return {"mirror": False, "rotate": False, "notes": ""}

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
                       "mirror": False, "rotate": False, "notes": ""}
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
# Cell-count store  (simple click log, no completion/blinding logic)
# ---------------------------------------------------------------------------

class CellCountStore:
    """
    Records every right-click (rat, xy_label, layer, coords, timestamp).
    All pending clicks are written to CSV on session end.
    """

    CLICKS_COLS = [
        "rat", "xy_label", "layer", "layer_label",
        "coord_y", "coord_x", "rotation_applied", "timestamp",
    ]

    def __init__(self, path: Path):
        self.path = path
        # In-memory list; flushed to disk on exit
        self._clicks: list[dict] = []

    # ------------------------------------------------------------------
    # Adding / undoing clicks
    # ------------------------------------------------------------------

    def add_click(self, rat: str, xy_label: str,
                  layer: str,
                  coord_y: float, coord_x: float,
                  rotation_applied: int) -> int:
        """Append a click. Returns total click count so far."""
        self._clicks.append({
            "rat":              rat,
            "xy_label":         xy_label,
            "layer":            layer,
            "layer_label":      CH_LABELS[layer],
            "coord_y":          round(coord_y, 2),
            "coord_x":          round(coord_x, 2),
            "rotation_applied": rotation_applied,
            "timestamp":        datetime.now().isoformat(timespec="seconds"),
        })
        return len(self._clicks)

    def undo_last_click(self) -> Optional[dict]:
        """Remove and return the most recent click, or None if empty."""
        if self._clicks:
            return self._clicks.pop()
        return None

    def total(self) -> int:
        return len(self._clicks)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Write all in-memory clicks to the CSV (append if file exists)."""
        if not self._clicks:
            print("[INFO] No clicks to save.")
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new_df = pd.DataFrame(self._clicks, columns=self.CLICKS_COLS)
        if self.path.exists():
            existing = pd.read_csv(self.path)
            for col in self.CLICKS_COLS:
                if col not in existing.columns:
                    existing[col] = ""
            combined = pd.concat([existing, new_df], ignore_index=True)
        else:
            combined = new_df
        combined.to_csv(self.path, index=False)
        print(f"[INFO] {len(self._clicks)} click(s) saved -> {self.path}")


# ---------------------------------------------------------------------------
# Subset / album store  (unchanged — kept for navigation convenience)
# ---------------------------------------------------------------------------

class SubsetStore:
    """Named collections of sections stored as JSON files."""

    def __init__(self, base_dir: Path):
        self._dir = base_dir / "subsets"
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        safe = re.sub(r"[^\w\-]", "_", name)[:64]
        return self._dir / f"{safe}.json"

    def list_subsets(self) -> list[dict]:
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
                    "path":        str(p),
                })
            except Exception:
                pass
        return out

    def load(self, name: str) -> Optional[dict]:
        p = self._path(name)
        if not p.exists():
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
        p   = self._path(name)
        now = datetime.now().isoformat(timespec="seconds")
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
        existing = self.load(name) or {}
        secs = existing.get("sections", [])
        if stem_ch1 not in secs:
            secs.append(stem_ch1)
        self.save(name, secs,
                  description=existing.get("description", description),
                  created_by=existing.get("created_by", created_by))
        return len(secs)


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
                 initial_subset: Optional[str] = None):
        self.loader  = loader
        self.qc      = qc
        self.counts  = counts
        self.subsets = subsets

        # Normal-mode navigation
        self.rat_idx = 0
        self.sec_idx = 0

        # Subset navigation
        self._subset_name: Optional[str]       = None
        self._subset_seq:  list[SectionRecord] = []
        self._subset_idx:  int                 = 0

        # Channel display
        self._overlay = False
        self._last_ch = "CH1"
        self._layers: dict[str, napari.layers.Image] = {}

        # Rotation
        self._rotation_k = 0

        # One Points layer per channel
        self._marker_layers: dict[str, Optional[napari.layers.Points]] = {
            ch: None for ch in CHANNELS}

        # Cross-rat comparison
        self._cmp_ref: Optional[SectionRecord] = None
        self._cmp_showing_ref: bool = False

        self.viewer = napari.Viewer(title="Histology QC Viewer")
        self._init_layers()
        self._bind_keys()
        self._bind_mouse()

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
        if self._in_subset:
            return self._subset_seq
        return self.loader.get_sections(self.loader.rats[self.rat_idx])

    @property
    def current_section(self) -> SectionRecord:
        if self._in_subset:
            return self._subset_seq[self._subset_idx]
        return self.loader.get_sections(self.loader.rats[self.rat_idx])[self.sec_idx]

    @property
    def _current_sec_idx(self) -> int:
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
        """Return the channel key of the currently selected layer, or fallback."""
        sel = self.viewer.layers.selection.active
        if sel is not None:
            for ch, layer in self._layers.items():
                if layer is sel:
                    return ch
        # Fallback: visible single-channel layer
        if not self._overlay:
            return self._last_ch
        return "CH1"

    # ------------------------------------------------------------------
    # Marker layer helpers
    # ------------------------------------------------------------------

    def _ensure_marker_layer(self, ch: str) -> napari.layers.Points:
        ml = self._marker_layers.get(ch)
        if ml is None or ml not in self.viewer.layers:
            color = CH_MARKER_COLORS[ch]
            ml = self.viewer.add_points(
                np.empty((0, 2), dtype=float),
                name=f"Markers {ch} - {CH_LABELS[ch]}",
                face_color=color,
                #edge_color="white",
                #edge_width=0.5,
                size=18,
                opacity=0.9,
                symbol="disc",
            )
            self._marker_layers[ch] = ml
        return ml

    def _add_point_to_layer(self, ch: str, coord_y: float, coord_x: float) -> None:
        ml       = self._ensure_marker_layer(ch)
        new_pt   = np.array([[coord_y, coord_x]])
        existing = ml.data
        ml.data  = new_pt if existing.shape[0] == 0 else np.vstack([existing, new_pt])

    def _remove_last_point_from_layer(self, ch: str) -> None:
        ml = self._marker_layers.get(ch)
        if ml is not None and ml in self.viewer.layers and ml.data.shape[0] > 0:
            ml.data = ml.data[:-1]

    def _clear_all_marker_layers(self) -> None:
        for ch in CHANNELS:
            ml = self._marker_layers.get(ch)
            if ml is not None and ml in self.viewer.layers:
                ml.data = np.empty((0, 2), dtype=float)

    # ------------------------------------------------------------------
    # Section loading
    # ------------------------------------------------------------------

    def _load_section(self) -> None:
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
            arr           = apply_rotation(arr, self._rotation_k)
            layer         = self._layers[ch]
            layer.data    = arr
            if arr.max() > 0:
                layer.contrast_limits = (0, float(arr.max()))

        # Ensure marker layers exist (persist across sections)
        for ch in CHANNELS:
            self._ensure_marker_layer(ch)

        self._update_title()
        self._update_visibility()
        self._print_section_info(anns)

    def _print_section_info(self, anns: dict) -> None:
        sec   = self.current_section
        idx   = self._current_sec_idx
        total = len(self.sections)
        src   = f"SUBSET:{self._subset_name}" if self._in_subset else sec.rat
        print(
            f"\n{'─'*62}\n"
            f"  Rat    : {sec.rat}  ({src})\n"
            f"  XY     : {sec.xy_label}  [{idx+1}/{total}]\n"
            f"  Rot    : {self._rotation_k * 90} deg\n"
            f"  Flags  : mirror={anns['mirror']}  rotate={anns['rotate']}\n"
            f"  Notes  : {anns['notes'] or '(no notes)'}\n"
            f"  Clicks : {self.counts.total()} total this session\n"
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
        anns  = self.qc.get(sec.rat, sec.stem_ch1)
        flags = " ".join(f"[{f.upper()}]"
                         for f in ("mirror", "rotate") if anns[f])
        src   = f"[SUBSET:{self._subset_name}] " if self._in_subset else ""
        title = (f"HistQC | {src}{sec.rat} | {sec.xy_label} | "
                 f"Section {idx+1}/{total} | clicks={self.counts.total()}")
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
    # Undo last marker
    # ------------------------------------------------------------------

    def _undo_last(self) -> None:
        removed = self.counts.undo_last_click()
        if removed:
            ch = removed["layer"]
            self._remove_last_point_from_layer(ch)
            show_info(f"Undo — removed {CH_LABELS[ch]} marker  "
                      f"({self.counts.total()} total)")
            print(f"[UNDO] Removed {CH_LABELS[ch]} at "
                  f"({removed['coord_y']}, {removed['coord_x']}). "
                  f"Total remaining: {self.counts.total()}")
        else:
            show_info("Nothing to undo.")
        self._update_title()

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _go_section(self, delta: int) -> None:
        if self._in_subset:
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
        sec = self.current_section
        print(f"\n[SUBSET] Current: {sec.rat} / {sec.xy_label}")
        print("[SUBSET] Enter subset name to add to:")
        try:
            name = input("  >> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("[SUBSET] Cancelled.")
            return
        if not name:
            show_info("No subset name entered; cancelled.")
            return

        existing = self.subsets.load(name)
        desc = existing.get("description", "") if existing else ""
        if not existing:
            print(f"[SUBSET] Creating new subset '{name}'. "
                  "Enter a description (blank = none):")
            try:
                desc = input("  >> ").strip()
            except (EOFError, KeyboardInterrupt):
                desc = ""

        n = self.subsets.add_section(name, sec.stem_ch1, description=desc)
        show_info(f"Added to subset '{name}' ({n} sections total)")

    def _load_subset_by_name(self, name: str, quiet: bool = False) -> bool:
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
        return True

    def _prompt_load_subset(self) -> None:
        subs = self.subsets.list_subsets()
        if subs:
            print("\n[SUBSET] Available subsets:")
            for s in subs:
                print(f"   {s['name']:30s}  {s['n_sections']:3d} sections  "
                      f"|  {s['description'] or ''}")
        print("\n[SUBSET] Enter subset name (blank = return to full dataset):")
        try:
            name = input("  >> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("[SUBSET] Cancelled.")
            return
        self._load_subset_by_name(name)
        self._load_section()

    def _list_subsets(self) -> None:
        subs = self.subsets.list_subsets()
        if not subs:
            print("\n[SUBSET] No subsets defined yet. Press 'a' to add sections.")
            return
        print(f"\n{'─'*62}")
        print(f"  {'NAME':<30} {'N':>4}  DESCRIPTION")
        print(f"{'─'*62}")
        for s in subs:
            print(f"  {s['name']:<30} {s['n_sections']:>4}  {s['description'] or ''}")
        print(f"{'─'*62}")

    # ------------------------------------------------------------------
    # QC annotation helpers
    # ------------------------------------------------------------------

    def _toggle_flag(self, flag: str) -> None:
        sec     = self.current_section
        new_val = self.qc.toggle(sec.rat, sec.stem_ch1, flag)
        show_info(f"{flag.upper()} -> {'ON' if new_val else 'OFF'}")
        self._update_title()

    def _enter_notes(self) -> None:
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
        sec = self.current_section
        self._cmp_ref         = sec
        self._cmp_showing_ref = False
        show_info(f"[CMP] Pinned: {sec.rat}  {sec.xy_label}")
        print(
            f"\n{'─'*62}\n"
            f"  [COMPARE] Reference pinned\n"
            f"  Rat  : {sec.rat}\n"
            f"  XY   : {sec.xy_label}\n"
            f"  Navigate to another section, then press Tab to flip.\n"
            f"{'─'*62}"
        )

    def _toggle_comparison_view(self) -> None:
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
        label = (f"[REF] {target.rat}  {target.xy_label}"
                 if self._cmp_showing_ref
                 else f"[CUR] {target.rat}  {target.xy_label}")
        show_info(f"[CMP] {label}  |  Tab to flip  |  Ctrl+P to save pair")

    def _load_images_into_layers(self, sec: "SectionRecord") -> None:
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
        if self._cmp_ref is None:
            show_info("[CMP] No reference pinned.")
            return
        sec = self.current_section
        if self._cmp_ref.rat == sec.rat and self._cmp_ref.stem_ch1 == sec.stem_ch1:
            show_info("[CMP] Reference and current are the same — navigate away first.")
            return
        now   = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest  = BASE_DIR / f"comparison_pair_{now}.txt"
        lines = [
            "# Histology comparison pair",
            f"# Saved: {datetime.now().isoformat(timespec='seconds')}",
            "",
            "[Reference]",
            f"rat      = {self._cmp_ref.rat}",
            f"xy_label = {self._cmp_ref.xy_label}",
            f"stem_ch1 = {self._cmp_ref.stem_ch1}",
            "",
            "[Current]",
            f"rat      = {sec.rat}",
            f"xy_label = {sec.xy_label}",
            f"stem_ch1 = {sec.stem_ch1}",
        ]
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text("\n".join(lines) + "\n")
            show_info(f"[CMP] Pair saved -> {dest.name}")
        except Exception as exc:
            show_warning(f"[CMP] Could not save pair: {exc}")

    # ------------------------------------------------------------------
    # Key bindings
    # ------------------------------------------------------------------

    def _bind_keys(self) -> None:
        v = self.viewer

        # Navigation
        v.bind_key("Right", lambda _: self._go_section(+1))
        v.bind_key("Left",  lambda _: self._go_section(-1))
        v.bind_key("]",     lambda _: self._go_rat(+1))
        v.bind_key("[",     lambda _: self._go_rat(-1))

        # Channel / display
        v.bind_key("1",     lambda _: self._show_channel("CH1"))
        v.bind_key("2",     lambda _: self._show_channel("CH2"))
        v.bind_key("3",     lambda _: self._show_channel("CH3"))
        v.bind_key("4",     lambda _: self._show_channel("CH4"))
        v.bind_key("Space", lambda _: self._toggle_overlay())

        # QC flags
        v.bind_key("m",     lambda _: self._toggle_flag("mirror"))
        v.bind_key("r",     lambda _: self._rotate_image())

        # Undo last marker
        v.bind_key("Backspace", lambda _: self._undo_last(), overwrite=True)

        # Save QC annotations
        v.bind_key("Control-s", lambda _: self._save_qc(), overwrite=True)

        # Notes
        v.bind_key("n",     lambda _: self._enter_notes())

        # Subsets
        v.bind_key("a",         lambda _: self._add_to_subset())
        v.bind_key("Control-l", lambda _: self._prompt_load_subset())
        v.bind_key("Control-a", lambda _: self._list_subsets())

        # Cross-rat comparison
        v.bind_key("c",         lambda _: self._pin_comparison_ref())
        v.bind_key("Tab",       lambda _: self._toggle_comparison_view(),
                   overwrite=True)
        v.bind_key("Control-p", lambda _: self._export_comparison_pair(),
                   overwrite=True)

    def _save_qc(self) -> None:
        self.qc.save()
        show_info("QC annotations saved.")

    # ------------------------------------------------------------------
    # Mouse callback — right-click places a marker
    # ------------------------------------------------------------------

    def _bind_mouse(self) -> None:
        viewer = self.viewer

        @viewer.mouse_drag_callbacks.append
        def on_mouse_drag(v, event):
            # Only act on right-click press
            if event.type != "mouse_press" or event.button != 2:
                return

            try:
                pos     = event.position
                coord_y = float(pos[-2])
                coord_x = float(pos[-1])
                sec     = self.current_section
                ch      = self._active_channel()

                total = self.counts.add_click(
                    rat              = sec.rat,
                    xy_label         = sec.xy_label,
                    layer            = ch,
                    coord_y          = coord_y,
                    coord_x          = coord_x,
                    rotation_applied = self._rotation_k * 90,
                )
                self._add_point_to_layer(ch, coord_y, coord_x)
                self._update_title()
                print(f"[COUNT] {CH_LABELS[ch]} marker at "
                      f"({coord_y:.1f}, {coord_x:.1f})  "
                      f"rat={sec.rat}  xy={sec.xy_label}  "
                      f"total={total}")
                show_info(f"{CH_LABELS[ch]} +1  (total: {total})")

            except Exception as exc:
                print(f"[WARN] Click callback error: {exc}")

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self) -> None:
        print(
            "\n"
            "╔═══════════════════════════════════════════════════════════╗\n"
            "║         Histology QC Viewer  --  Key Map                  ║\n"
            "╠═══════════════════════════════════════════════════════════╣\n"
            "║  <- / ->     Previous / Next section                      ║\n"
            "║  [ / ]       Previous / Next rat  (full dataset only)     ║\n"
            "║  1-4         Show CH1 / CH2 / CH3 / CH4 only              ║\n"
            "║  Space       Toggle overlay <-> single channel            ║\n"
            "║  m           Toggle MIRROR flag                           ║\n"
            "║  r           Rotate image 90 deg CW                      ║\n"
            "║  Right-click Place a marker and count the cell            ║\n"
            "║  Backspace   Undo last marker                             ║\n"
            "║  Ctrl+S      Save QC annotations                         ║\n"
            "║  n           Edit NOTES (console)                         ║\n"
            "╠═══════════════════════════════════════════════════════════╣\n"
            "║  CROSS-RAT COMPARISON                                      ║\n"
            "║  c           Pin current section as reference             ║\n"
            "║  Tab         Flip between reference and current section    ║\n"
            "║  Ctrl+P      Save comparison pair to .txt                 ║\n"
            "╠═══════════════════════════════════════════════════════════╣\n"
            "║  SUBSETS                                                   ║\n"
            "║  a           Add current section to a subset              ║\n"
            "║  Ctrl+L      Load a subset as navigation source           ║\n"
            "║  Ctrl+A      List all subsets                             ║\n"
            "╚═══════════════════════════════════════════════════════════╝\n"
            "  Counts are saved to cell_counts.csv on exit.\n"
        )
        try:
            napari.run()
        finally:
            self.qc.save()
            self.counts.save()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    global BASE_DIR, PNG_DIR, SORT_DIR, QC_CSV, COUNTS_CSV

    parser = argparse.ArgumentParser(
        description="Napari histology QC viewer with right-click cell counting.")
    parser.add_argument("base_dir", nargs="?", help="Override BASE_DIR")
    parser.add_argument("--subset", metavar="NAME",
                        help="Load subset NAME as navigation source on startup")
    args = parser.parse_args()

    if args.base_dir:
        BASE_DIR   = Path(args.base_dir).expanduser().resolve()
        PNG_DIR    = BASE_DIR / "PNG_fullsize"
        SORT_DIR   = BASE_DIR / "features" / "sorting"
        QC_CSV     = BASE_DIR / "qc_annotations.csv"
        COUNTS_CSV = BASE_DIR / "cell_counts.csv"
        print(f"[INFO] Using base directory: {BASE_DIR}")

    for d, label in [(PNG_DIR, "PNG"), (SORT_DIR, "sorting")]:
        if not d.exists():
            print(f"[WARN] Expected directory not found: {d}  ({label})")

    try:
        loader = DatasetLoader(sort_dir=SORT_DIR)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    qc      = QCStore(path=QC_CSV)
    counts  = CellCountStore(path=COUNTS_CSV)
    subsets = SubsetStore(base_dir=BASE_DIR)

    viewer_ctrl = HistologyViewer(
        loader         = loader,
        qc             = qc,
        counts         = counts,
        subsets        = subsets,
        initial_subset = args.subset,
    )
    viewer_ctrl.run()


if __name__ == "__main__":
    main()
