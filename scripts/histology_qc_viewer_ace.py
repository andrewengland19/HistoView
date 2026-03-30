"""
histology_qc_viewer.py
======================
Napari-based keyboard-driven QC viewer for rat histology datasets,
extended with per-channel cell counting and quantification.

Directory layout assumed
------------------------
~/Microscopy/Cohort1_TPH2/
    dataset/
        <ratID>/
            <region>/
                XY##/
                    CH1/ ... CH4/
    PNG/
        <ratID>/
            CH1/
                <ratID>_<region>_XY##_CH1.png
            CH2/ ... CH4/
    features/
        sorting/
            predicted_order_<ratID>.csv   (column: "image", CH1 stems)
            predicted_order.csv           (optional combined)

Keyboard map
------------
  ←  / →      previous / next section
  [  / ]      previous / next rat
  1-4         show CH1-CH4 only (also sets active counting channel)
  Space       toggle overlay (all channels) ↔ last single-channel
  m           toggle mirror flag
  r           toggle rotate flag
  q           toggle quantify flag
  n           enter/edit notes for current section (console prompt)
  z           undo last cell count marker on active channel
  p           print current section cell count summary to console
  Ctrl+S      force-save QC annotations and cell counts

Cell counting
-------------
  Right-click  Place a marker dot on the current image and record it
               under the currently active channel. The marker is shown
               as a coloured dot on a dedicated Points layer.
               Each count is written to cell_counts.csv with columns:
                   rat, section, xy_label, channel, ch_label, x, y,
                   timestamp

Author: generated for Cohort1_TPH2 pipeline
"""

from __future__ import annotations

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

BASE_DIR        = Path.home() / "Microscopy" / "Cohort1_TPH2"
PNG_DIR         = BASE_DIR / "PNG_fullsize"
SORT_DIR        = BASE_DIR / "features" / "sorting"
DATASET_DIR     = BASE_DIR / "dataset"
QC_CSV          = BASE_DIR / "qc_annotations.csv"
COUNTS_CSV      = BASE_DIR / "cell_counts.csv"
MANIFEST_CSV    = BASE_DIR / "section_manifest.csv"

# Set True (or pass --manifest-only on the CLI) to load only sections
# marked include=1 in section_manifest.csv.
MANIFEST_ONLY   = False

CHANNELS        = ["CH1", "CH2", "CH3", "CH4"]
CH_LABELS       = {
    "CH1": "DAPI",
    "CH2": "TPH2/GFP",
    "CH3": "RFP",
    "CH4": "Cy5",
}
# Colormaps per channel (napari built-ins)
CH_COLORMAPS    = {
    "CH1": "blue",
    "CH2": "green",
    "CH3": "red",
    "CH4": "magenta",
}
# Point marker colours per channel (RGBA, 0-255)
CH_POINT_COLORS = {
    "CH1": (100, 149, 237, 220),   # cornflower blue
    "CH2": ( 50, 230,  50, 220),   # bright green
    "CH3": (255,  80,  80, 220),   # bright red
    "CH4": (255,  50, 255, 220),   # magenta
}
POINT_SIZE      = 18   # diameter in screen pixels


# ---------------------------------------------------------------------------
# Data-loader abstraction
# ---------------------------------------------------------------------------

class SectionRecord:
    """One XY section for one rat, with derived paths for all channels."""

    __slots__ = ("rat", "stem_ch1", "stems", "png_paths", "xy_label")

    def __init__(self, rat: str, stem_ch1: str):
        self.rat      = rat
        self.stem_ch1 = stem_ch1                            # e.g. rat761_RBS_XY35_CH1
        self.xy_label = self._parse_xy(stem_ch1)            # e.g. XY35
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
        paths = {}
        for ch in CHANNELS:
            paths[ch] = PNG_DIR / self.rat / ch / f"{self.stems[ch]}.png"
        return paths

    def __repr__(self) -> str:
        return f"<SectionRecord rat={self.rat} xy={self.xy_label}>"


class DatasetLoader:
    """
    Discovers all rats from predicted_order_<ratID>.csv files and builds
    ordered SectionRecord lists per rat.
    """

    def __init__(self, sort_dir: Path = SORT_DIR):
        self.sort_dir   = sort_dir
        self.rats: list[str]                      = []
        self.sections: dict[str, list[SectionRecord]] = {}
        self._load()

    def _load(self) -> None:
        # ------------------------------------------------------------------
        # Optional manifest filter
        # ------------------------------------------------------------------
        include_set: Optional[set[tuple[str, str]]] = None
        if MANIFEST_ONLY:
            if not MANIFEST_CSV.exists():
                raise FileNotFoundError(
                    f"--manifest-only requested but manifest not found: {MANIFEST_CSV}\n"
                    f"Run  generate_manifest.py  first."
                )
            try:
                mdf = pd.read_csv(MANIFEST_CSV, dtype={"include": int})
            except Exception as exc:
                raise RuntimeError(f"Could not read manifest CSV: {exc}") from exc
            included = mdf[mdf["include"] == 1]
            include_set = set(
                zip(included["rat"].astype(str), included["section"].astype(str))
            )
            print(
                f"[INFO] Manifest filter active — "
                f"{len(include_set)} section(s) included across "
                f"{included['rat'].nunique()} rat(s)."
            )

        # ------------------------------------------------------------------
        # Load per-rat ordering CSVs
        # ------------------------------------------------------------------
        pattern = "predicted_order_*.csv"
        csvs = sorted(self.sort_dir.glob(pattern))
        if not csvs:
            raise FileNotFoundError(
                f"No predicted_order_*.csv files found in {self.sort_dir}"
            )
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
            stems = df["image"].dropna().tolist()
            # Apply manifest filter if active
            if include_set is not None:
                stems = [s for s in stems if (rat, str(s)) in include_set]
                if not stems:
                    print(f"[INFO] Rat {rat}: no included sections after manifest filter, skipping.")
                    continue
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


# ---------------------------------------------------------------------------
# QC annotation store
# ---------------------------------------------------------------------------

class QCStore:
    """
    Reads/writes per-section QC annotations to a CSV.

    Columns: rat, image, mirror, rotate, quantify, notes
    """

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
            return {
                "mirror":   bool(row.get("mirror", False)),
                "rotate":   bool(row.get("rotate", False)),
                "quantify": bool(row.get("quantify", False)),
                "notes":    str(row.get("notes", "")),
            }
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
                       "mirror": False, "rotate": False, "quantify": False, "notes": ""}
                row.update({k: v for k, v in kwargs.items() if k in self.COLUMNS})
                self._df = pd.concat(
                    [self._df, pd.DataFrame([row])], ignore_index=True
                )

    def toggle(self, rat: str, image: str, flag: str) -> bool:
        current = self.get(rat, image)
        new_val = not current.get(flag, False)
        self.set(rat, image, **{flag: new_val})
        return new_val

    def save(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._df.to_csv(self.path, index=False)
            print(f"[INFO] QC annotations saved → {self.path}")


# ---------------------------------------------------------------------------
# Cell count store
# ---------------------------------------------------------------------------

COUNTS_COLUMNS = [
    "rat", "section", "xy_label", "channel", "ch_label", "x", "y", "timestamp"
]


class CellCountStore:
    """
    Reads/writes per-cell point records to a CSV.

    Each row is one right-click annotation: rat, section stem, xy_label,
    channel key, human channel label, image x/y coordinates, and timestamp.

    Also maintains an in-memory list ordered by insertion time so that undo
    (remove last) can be applied per (rat, section, channel) triple.
    """

    _lock = threading.Lock()

    def __init__(self, path: Path = COUNTS_CSV):
        self.path = path
        if path.exists():
            try:
                self._df = pd.read_csv(path)
                for col in COUNTS_COLUMNS:
                    if col not in self._df.columns:
                        self._df[col] = ""
            except Exception as exc:
                print(f"[WARN] Could not read counts CSV ({exc}); starting fresh.")
                self._df = pd.DataFrame(columns=COUNTS_COLUMNS)
        else:
            self._df = pd.DataFrame(columns=COUNTS_COLUMNS)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add(
        self,
        rat: str,
        section: str,
        xy_label: str,
        channel: str,
        x: float,
        y: float,
    ) -> None:
        row = {
            "rat":       rat,
            "section":   section,
            "xy_label":  xy_label,
            "channel":   channel,
            "ch_label":  CH_LABELS.get(channel, channel),
            "x":         round(float(x), 2),
            "y":         round(float(y), 2),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        with self._lock:
            self._df = pd.concat(
                [self._df, pd.DataFrame([row])], ignore_index=True
            )

    def undo_last(self, rat: str, section: str, channel: str) -> bool:
        """Remove the most recent point for (rat, section, channel).

        Returns True if a point was removed, False if none existed.
        """
        with self._lock:
            mask = (
                (self._df["rat"]     == rat)
                & (self._df["section"] == section)
                & (self._df["channel"] == channel)
            )
            if not mask.any():
                return False
            last_idx = self._df[mask].index[-1]
            self._df = self._df.drop(index=last_idx).reset_index(drop=True)
            return True

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_points(self, rat: str, section: str, channel: str) -> np.ndarray:
        """Return (N, 2) array of [y, x] image coords for a given key."""
        mask = (
            (self._df["rat"]     == rat)
            & (self._df["section"] == section)
            & (self._df["channel"] == channel)
        )
        sub = self._df[mask]
        if sub.empty:
            return np.empty((0, 2), dtype=float)
        # napari Points layer expects [row, col] = [y, x]
        return sub[["y", "x"]].to_numpy(dtype=float)

    def summary(self, rat: Optional[str] = None) -> pd.DataFrame:
        """
        Return a count summary DataFrame.

        Columns: rat, xy_label, channel, ch_label, count
        Optionally filtered to a single rat.
        """
        df = self._df.copy()
        if rat is not None:
            df = df[df["rat"] == rat]
        if df.empty:
            return pd.DataFrame(
                columns=["rat", "xy_label", "channel", "ch_label", "count"]
            )
        grp = (
            df.groupby(["rat", "xy_label", "channel", "ch_label"])
            .size()
            .reset_index(name="count")
        )
        return grp.sort_values(["rat", "xy_label", "channel"])

    # ------------------------------------------------------------------
    # Persist
    # ------------------------------------------------------------------

    def save(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._df.to_csv(self.path, index=False)
            print(f"[INFO] Cell counts saved → {self.path}")


# ---------------------------------------------------------------------------
# Image loading helpers
# ---------------------------------------------------------------------------

def load_png_as_array(path: Path) -> Optional[np.ndarray]:
    """Load a PNG; return None (with warning) if missing or unreadable."""
    if not path.exists():
        show_warning(f"Missing: {path.name}")
        print(f"[WARN] Missing PNG: {path}")
        return None
    try:
        img = iio.imread(path)
        if img.ndim == 3 and img.shape[-1] in (3, 4):
            img = img[..., :3].mean(axis=-1).astype(np.float32)
        return img.astype(np.float32)
    except Exception as exc:
        show_warning(f"Failed to read {path.name}: {exc}")
        print(f"[WARN] Could not read {path}: {exc}")
        return None


def placeholder_array(shape: tuple = (512, 512)) -> np.ndarray:
    return np.zeros(shape, dtype=np.float32)


# ---------------------------------------------------------------------------
# Napari viewer controller
# ---------------------------------------------------------------------------

class HistologyViewer:
    """
    Manages a napari viewer instance and all keyboard/mouse callbacks,
    including right-click cell counting on any channel.
    """

    def __init__(
        self,
        loader: DatasetLoader,
        qc: QCStore,
        counts: CellCountStore,
    ):
        self.loader      = loader
        self.qc          = qc
        self.counts      = counts
        self.rat_idx     = 0
        self.sec_idx     = 0
        self._overlay    = False
        self._last_ch    = "CH1"
        self._layers:       dict[str, napari.layers.Image]  = {}
        self._pt_layers:    dict[str, napari.layers.Points] = {}

        self.viewer = napari.Viewer(title="Histology QC Viewer")
        self._init_layers()
        self._init_point_layers()
        self._bind_keys()
        self._bind_mouse()
        self._load_section()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_rat(self) -> str:
        return self.loader.rats[self.rat_idx]

    @property
    def sections(self) -> list[SectionRecord]:
        return self.loader.get_sections(self.current_rat)

    @property
    def current_section(self) -> SectionRecord:
        return self.sections[self.sec_idx]

    @property
    def active_channel(self) -> str:
        """The channel currently selected for single-channel display / counting."""
        return self._last_ch

    # ------------------------------------------------------------------
    # Layer initialisation
    # ------------------------------------------------------------------

    def _init_layers(self) -> None:
        blank = placeholder_array()
        for ch in CHANNELS:
            layer = self.viewer.add_image(
                blank,
                name=f"{ch} — {CH_LABELS[ch]}",
                colormap=CH_COLORMAPS[ch],
                blending="additive",
                visible=(ch == "CH1"),
            )
            self._layers[ch] = layer

    def _init_point_layers(self) -> None:
        """Create one Points layer per channel for cell count markers."""
        for ch in CHANNELS:
            rgba = np.array(CH_POINT_COLORS[ch], dtype=float) / 255.0
            layer = self.viewer.add_points(
                np.empty((0, 2), dtype=float),
                name=f"Counts {ch} — {CH_LABELS[ch]}",
                face_color=[rgba],
                border_color="white",
                border_width=0.08,    # fraction of point size
                size=POINT_SIZE,
                symbol="disc",
                blending="translucent",
                visible=True,
                # Prevent accidental dragging: put in pan/zoom mode
                # (Users right-click to add via our callback, not napari's add mode)
            )
            # Lock the layer so the user can't accidentally move points via GUI
            layer.mode = "pan_zoom"
            self._pt_layers[ch] = layer

    # ------------------------------------------------------------------
    # Section loading
    # ------------------------------------------------------------------

    def _load_section(self) -> None:
        sec  = self.current_section
        anns = self.qc.get(sec.rat, sec.stem_ch1)

        ref_shape = None
        for ch in CHANNELS:
            path = sec.png_paths[ch]
            arr  = load_png_as_array(path)
            if arr is None:
                arr = placeholder_array() if ref_shape is None else placeholder_array(ref_shape)
            else:
                ref_shape = arr.shape

            layer = self._layers[ch]
            layer.data = arr
            if arr.max() > 0:
                layer.contrast_limits = (0, float(arr.max()))

        self._refresh_point_layers()
        self._update_title()
        self._update_visibility()
        self._print_section_info(anns)

    # ------------------------------------------------------------------
    # Point layer management
    # ------------------------------------------------------------------

    def _refresh_point_layers(self) -> None:
        """Reload all Points layers from the count store for the current section."""
        sec = self.current_section
        for ch in CHANNELS:
            pts = self.counts.get_points(sec.rat, sec.stem_ch1, ch)
            layer = self._pt_layers[ch]
            if pts.shape[0] > 0:
                layer.data = pts
            else:
                layer.data = np.empty((0, 2), dtype=float)

    def _add_point(self, y: float, x: float) -> None:
        """Record one cell at image coordinates (y, x) on the active channel."""
        sec = self.current_section
        ch  = self.active_channel
        self.counts.add(
            rat=sec.rat,
            section=sec.stem_ch1,
            xy_label=sec.xy_label,
            channel=ch,
            x=x,
            y=y,
        )
        # Update the napari point layer live
        layer = self._pt_layers[ch]
        existing = layer.data
        new_pt   = np.array([[y, x]], dtype=float)
        if existing.shape[0] == 0:
            layer.data = new_pt
        else:
            layer.data = np.vstack([existing, new_pt])

        total = layer.data.shape[0]
        show_info(
            f"[{ch} {CH_LABELS[ch]}] Cell #{total} — "
            f"({x:.0f}, {y:.0f})"
        )

    def _undo_last_point(self) -> None:
        """Remove the last placed point on the active channel."""
        sec = self.current_section
        ch  = self.active_channel
        removed = self.counts.undo_last(sec.rat, sec.stem_ch1, ch)
        if removed:
            pts = self.counts.get_points(sec.rat, sec.stem_ch1, ch)
            layer = self._pt_layers[ch]
            layer.data = pts if pts.shape[0] > 0 else np.empty((0, 2), dtype=float)
            show_info(f"[{ch}] Undo — {layer.data.shape[0]} point(s) remaining")
        else:
            show_info(f"[{ch}] Nothing to undo.")

    # ------------------------------------------------------------------
    # Section info print
    # ------------------------------------------------------------------

    def _print_section_info(self, anns: dict) -> None:
        sec   = self.current_section
        idx   = self.sec_idx
        total = len(self.sections)
        flags = (
            f"mirror={anns['mirror']}  rotate={anns['rotate']}  "
            f"quantify={anns['quantify']}"
        )
        notes = anns["notes"] or "(no notes)"
        # Per-channel counts for this section
        count_parts = []
        for ch in CHANNELS:
            pts = self.counts.get_points(sec.rat, sec.stem_ch1, ch)
            count_parts.append(f"{ch}={pts.shape[0]}")
        count_str = "  ".join(count_parts)
        print(
            f"\n{'─'*60}\n"
            f"  Rat     : {sec.rat}\n"
            f"  XY      : {sec.xy_label}  [{idx+1}/{total}]\n"
            f"  Flags   : {flags}\n"
            f"  Notes   : {notes}\n"
            f"  Counts  : {count_str}\n"
            f"  Counting on: {self.active_channel} ({CH_LABELS[self.active_channel]})\n"
            f"{'─'*60}"
        )

    def _print_count_summary(self) -> None:
        """Print a table of counts for the current rat to console."""
        sec = self.current_section
        df  = self.counts.summary(rat=sec.rat)
        if df.empty:
            print(f"\n[COUNTS] No counts recorded yet for {sec.rat}.")
            return
        print(f"\n[COUNTS] Summary for {sec.rat}:")
        print(df.to_string(index=False))

        # Also print per-section × channel pivot for quick reading
        pivot = df.pivot_table(
            index="xy_label", columns="ch_label", values="count", fill_value=0
        )
        print("\n[COUNTS] Per-section pivot:")
        print(pivot.to_string())
        print()

    # ------------------------------------------------------------------
    # Visibility helpers
    # ------------------------------------------------------------------

    def _update_visibility(self) -> None:
        if self._overlay:
            for ch, layer in self._layers.items():
                layer.visible = True
        else:
            for ch, layer in self._layers.items():
                layer.visible = (ch == self._last_ch)
        # Point layers are always visible regardless of image display mode
        for ch, layer in self._pt_layers.items():
            layer.visible = True

    def _show_channel(self, ch: str) -> None:
        self._overlay = False
        self._last_ch = ch
        self._update_visibility()
        show_info(
            f"Showing {ch} — {CH_LABELS[ch]}  "
            f"[right-click to count on this channel]"
        )

    def _toggle_overlay(self) -> None:
        self._overlay = not self._overlay
        self._update_visibility()
        state = "OVERLAY (all channels)" if self._overlay else f"Single: {self._last_ch}"
        show_info(state)

    # ------------------------------------------------------------------
    # Window title
    # ------------------------------------------------------------------

    def _update_title(self) -> None:
        sec   = self.current_section
        idx   = self.sec_idx
        total = len(self.sections)
        anns  = self.qc.get(sec.rat, sec.stem_ch1)
        flags = " ".join(
            f"[{f.upper()}]" for f in ("mirror", "rotate", "quantify") if anns[f]
        )

        # Inline count badge
        ch_counts = []
        for ch in CHANNELS:
            pts = self.counts.get_points(sec.rat, sec.stem_ch1, ch)
            n = pts.shape[0]
            if n > 0:
                ch_counts.append(f"{CH_LABELS[ch]}:{n}")
        count_badge = "  [" + " | ".join(ch_counts) + "]" if ch_counts else ""

        title = (
            f"HistQC | {sec.rat} | {sec.xy_label} | "
            f"Section {idx+1}/{total} | "
            f"Counting: {self.active_channel}"
        )
        if flags:
            title += f"  {flags}"
        title += count_badge
        self.viewer.title = title

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _go_section(self, delta: int) -> None:
        secs = self.sections
        if not secs:
            return
        self.sec_idx = (self.sec_idx + delta) % len(secs)
        self._load_section()

    def _go_rat(self, delta: int) -> None:
        self.qc.save()
        self.counts.save()
        self.rat_idx = (self.rat_idx + delta) % len(self.loader.rats)
        self.sec_idx = 0
        show_info(f"Switched to {self.current_rat}")
        self._load_section()

    # ------------------------------------------------------------------
    # QC annotation helpers
    # ------------------------------------------------------------------

    def _toggle_flag(self, flag: str) -> None:
        sec     = self.current_section
        new_val = self.qc.toggle(sec.rat, sec.stem_ch1, flag)
        show_info(f"{flag.upper()} → {'ON' if new_val else 'OFF'}")
        self._update_title()

    def _enter_notes(self) -> None:
        sec  = self.current_section
        anns = self.qc.get(sec.rat, sec.stem_ch1)
        print(f"\n[NOTES] Current: {anns['notes'] or '(empty)'}")
        print("[NOTES] Enter new notes (blank = keep existing, 'clear' = delete):")
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
    # Key bindings
    # ------------------------------------------------------------------

    def _bind_keys(self) -> None:
        v = self.viewer

        # Navigation
        v.bind_key("Right",   lambda _: self._go_section(+1))
        v.bind_key("Left",    lambda _: self._go_section(-1))
        v.bind_key("]",       lambda _: self._go_rat(+1))
        v.bind_key("[",       lambda _: self._go_rat(-1))

        # Channel selection (also sets active counting channel)
        v.bind_key("1", lambda _: self._show_channel("CH1"))
        v.bind_key("2", lambda _: self._show_channel("CH2"))
        v.bind_key("3", lambda _: self._show_channel("CH3"))
        v.bind_key("4", lambda _: self._show_channel("CH4"))

        # Overlay toggle
        v.bind_key("Space", lambda _: self._toggle_overlay())

        # QC flags
        v.bind_key("m", lambda _: self._toggle_flag("mirror"))
        v.bind_key("r", lambda _: self._toggle_flag("rotate"))
        v.bind_key("q", lambda _: self._toggle_flag("quantify"))

        # Notes
        v.bind_key("n", lambda _: self._enter_notes())

        # Cell count helpers
        v.bind_key("z", lambda _: self._undo_last_point())
        v.bind_key("p", lambda _: self._print_count_summary())

        # Force save (QC + counts)
        v.bind_key("Control-s", lambda _: self._save_all())

    def _save_all(self) -> None:
        self.qc.save()
        self.counts.save()

    # ------------------------------------------------------------------
    # Mouse callback — right-click to place count marker
    # ------------------------------------------------------------------

    def _bind_mouse(self) -> None:

        @self.viewer.mouse_drag_callbacks.append
        def on_mouse_press(viewer, event):
            # Fire only on the initial press event, not during drag
            if event.type != "mouse_press":
                return
            # Right-click only (button 2)
            if event.button != 2:
                return
            try:
                pos = event.position
                if pos is None or len(pos) < 2:
                    return
                y, x = float(pos[-2]), float(pos[-1])
                self._add_point(y, x)
                self._update_title()
            except Exception as exc:
                print(f"[WARN] Right-click callback error: {exc}")

        # Keep legacy double-click for coordinate inspection
        @self.viewer.mouse_double_click_callbacks.append
        def on_double_click(viewer, event):
            try:
                coords = tuple(round(c, 2) for c in event.position)
                sec    = self.current_section
                anns   = self.qc.get(sec.rat, sec.stem_ch1)
                print(
                    f"\n[CLICK] Coordinates : {coords}\n"
                    f"        Rat          : {sec.rat}\n"
                    f"        XY label     : {sec.xy_label}\n"
                    f"        Stem (CH1)   : {sec.stem_ch1}\n"
                    f"        Annotations  : {anns}"
                )
            except Exception as exc:
                print(f"[WARN] Double-click callback error: {exc}")

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self) -> None:
        print(
            "\n"
            "╔══════════════════════════════════════════════════════╗\n"
            "║      Histology QC + Cell Counter  —  Key Map         ║\n"
            "╠══════════════════════════════════════════════════════╣\n"
            "║  ← / →     Previous / Next section                   ║\n"
            "║  [ / ]     Previous / Next rat                        ║\n"
            "║  1-4       Show CH1-4 only  (sets counting channel)   ║\n"
            "║  Space     Toggle overlay (all channels) ↔ single    ║\n"
            "║  m         Toggle MIRROR flag                         ║\n"
            "║  r         Toggle ROTATE flag                         ║\n"
            "║  q         Toggle QUANTIFY flag                       ║\n"
            "║  n         Enter / edit NOTES (console)               ║\n"
            "║  z         Undo last count on active channel          ║\n"
            "║  p         Print count summary for current rat        ║\n"
            "║  Ctrl+S    Force-save QC annotations + cell counts   ║\n"
            "║                                                        ║\n"
            "║  RIGHT-CLICK   Place cell marker on active channel    ║\n"
            "║  Dbl-click     Print image coords to console          ║\n"
            "╚══════════════════════════════════════════════════════╝\n"
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
    global BASE_DIR, PNG_DIR, SORT_DIR, DATASET_DIR, QC_CSV, COUNTS_CSV
    global MANIFEST_CSV, MANIFEST_ONLY

    args = sys.argv[1:]
    manifest_only_flag = "--manifest-only" in args
    args = [a for a in args if a != "--manifest-only"]

    if args:
        BASE_DIR     = Path(args[0]).expanduser().resolve()
        PNG_DIR      = BASE_DIR / "PNG_fullsize"
        SORT_DIR     = BASE_DIR / "features" / "sorting"
        DATASET_DIR  = BASE_DIR / "dataset"
        QC_CSV       = BASE_DIR / "qc_annotations.csv"
        COUNTS_CSV   = BASE_DIR / "cell_counts.csv"
        MANIFEST_CSV = BASE_DIR / "section_manifest.csv"
        print(f"[INFO] Using base directory: {BASE_DIR}")

    if manifest_only_flag:
        MANIFEST_ONLY = True
        print(f"[INFO] --manifest-only: sections filtered to include=1 in {MANIFEST_CSV}")

    for d, label in [(PNG_DIR, "PNG"), (SORT_DIR, "sorting")]:
        if not d.exists():
            print(f"[WARN] Expected directory not found: {d}  ({label})")

    try:
        loader = DatasetLoader(sort_dir=SORT_DIR)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    qc     = QCStore(path=QC_CSV)
    counts = CellCountStore(path=COUNTS_CSV)

    viewer_ctrl = HistologyViewer(loader=loader, qc=qc, counts=counts)
    viewer_ctrl.run()


if __name__ == "__main__":
    main()
