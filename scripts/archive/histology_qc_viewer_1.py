"""
histology_qc_viewer.py
======================
Napari-based keyboard-driven QC viewer for rat histology datasets.

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
  1-4         show CH1-CH4 only
  Space       toggle overlay (all channels) ↔ last single-channel
  m           toggle mirror flag
  r           toggle rotate flag
  q           toggle quantify flag
  n           enter/edit notes for current section (console prompt)
  Ctrl+S      force-save QC annotations now

Author: generated for Cohort1_TPH2 pipeline
"""

from __future__ import annotations

import re
import sys
import threading
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
            records = [SectionRecord(rat, str(s)) for s in stems]
            self.rats.append(rat)
            self.sections[rat] = records

        if not self.rats:
            raise RuntimeError("No valid rat datasets loaded.")

        print(f"[INFO] Loaded {len(self.rats)} rat(s): {self.rats}")

    @staticmethod
    def _rat_from_filename(name: str) -> str:
        # predicted_order_rat761.csv → rat761
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
                # Ensure all expected columns exist
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
            # Convert RGB(A) → luminance for consistency
            img = img[..., :3].mean(axis=-1).astype(np.float32)
        return img.astype(np.float32)
    except Exception as exc:
        show_warning(f"Failed to read {path.name}: {exc}")
        print(f"[WARN] Could not read {path}: {exc}")
        return None


def placeholder_array(shape: tuple = (512, 512)) -> np.ndarray:
    """Black placeholder image when a channel PNG is missing."""
    return np.zeros(shape, dtype=np.float32)


# ---------------------------------------------------------------------------
# Napari viewer controller
# ---------------------------------------------------------------------------

class HistologyViewer:
    """
    Manages a napari viewer instance and all keyboard/mouse callbacks.
    """

    def __init__(self, loader: DatasetLoader, qc: QCStore):
        self.loader      = loader
        self.qc          = qc
        self.rat_idx     = 0
        self.sec_idx     = 0
        self._overlay    = False          # True = all channels visible
        self._last_ch    = "CH1"         # last single-channel shown
        self._layers: dict[str, napari.layers.Image] = {}

        self.viewer = napari.Viewer(title="Histology QC Viewer")
        self._init_layers()
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

    # ------------------------------------------------------------------
    # Layer initialisation
    # ------------------------------------------------------------------

    def _init_layers(self) -> None:
        """Add one layer per channel to the viewer (empty at first)."""
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

    # ------------------------------------------------------------------
    # Section loading
    # ------------------------------------------------------------------

    def _load_section(self) -> None:
        sec  = self.current_section
        anns = self.qc.get(sec.rat, sec.stem_ch1)

        # Determine reference shape from CH1 (or fallback)
        ref_shape = None

        for ch in CHANNELS:
            path  = sec.png_paths[ch]
            arr   = load_png_as_array(path)
            if arr is None:
                if ref_shape is None:
                    arr = placeholder_array()
                else:
                    arr = placeholder_array(ref_shape)
            else:
                ref_shape = arr.shape

            layer = self._layers[ch]
            layer.data = arr
            # Reset contrast limits per channel
            if arr.max() > 0:
                layer.contrast_limits = (0, float(arr.max()))

        self._update_title()
        self._update_visibility()
        self._print_section_info(anns)

    def _print_section_info(self, anns: dict) -> None:
        sec = self.current_section
        idx = self.sec_idx
        total = len(self.sections)
        flags = (
            f"mirror={anns['mirror']}  rotate={anns['rotate']}  "
            f"quantify={anns['quantify']}"
        )
        notes = anns["notes"] or "(no notes)"
        print(
            f"\n{'─'*60}\n"
            f"  Rat   : {sec.rat}\n"
            f"  XY    : {sec.xy_label}  [{idx+1}/{total}]\n"
            f"  Flags : {flags}\n"
            f"  Notes : {notes}\n"
            f"{'─'*60}"
        )

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

    def _show_channel(self, ch: str) -> None:
        self._overlay   = False
        self._last_ch   = ch
        self._update_visibility()
        show_info(f"Showing {ch} — {CH_LABELS[ch]}")

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
        title = (
            f"HistQC | {sec.rat} | {sec.xy_label} | "
            f"Section {idx+1}/{total}"
        )
        if flags:
            title += f"  {flags}"
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
        self.qc.save()  # auto-save when switching rats
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
        """Prompt for notes on the current section (console input)."""
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

        # Channel selection
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

        # Force save
        v.bind_key("Control-s", lambda _: self.qc.save())

    # ------------------------------------------------------------------
    # Mouse callback
    # ------------------------------------------------------------------

    def _bind_mouse(self) -> None:
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
                print(f"[WARN] Mouse callback error: {exc}")

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self) -> None:
        print(
            "\n"
            "╔══════════════════════════════════════════════════════╗\n"
            "║          Histology QC Viewer  —  Key Map             ║\n"
            "╠══════════════════════════════════════════════════════╣\n"
            "║  ← / →     Previous / Next section                   ║\n"
            "║  [ / ]     Previous / Next rat                        ║\n"
            "║  1-4       Show CH1 / CH2 / CH3 / CH4 only           ║\n"
            "║  Space     Toggle overlay (all channels) ↔ single    ║\n"
            "║  m         Toggle MIRROR flag                         ║\n"
            "║  r         Toggle ROTATE flag                         ║\n"
            "║  q         Toggle QUANTIFY flag                       ║\n"
            "║  n         Enter / edit NOTES (console)               ║\n"
            "║  Ctrl+S    Force-save QC annotations                  ║\n"
            "║  Dbl-click Print image coords + section metadata      ║\n"
            "╚══════════════════════════════════════════════════════╝\n"
        )
        try:
            napari.run()
        finally:
            # Always save on exit
            self.qc.save()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    # Allow overriding BASE_DIR via first CLI argument
    global BASE_DIR, PNG_DIR, SORT_DIR, DATASET_DIR, QC_CSV
    if len(sys.argv) > 1:
        BASE_DIR    = Path(sys.argv[1]).expanduser().resolve()
        PNG_DIR     = BASE_DIR / "PNG"
        SORT_DIR    = BASE_DIR / "features" / "sorting"
        DATASET_DIR = BASE_DIR / "dataset"
        QC_CSV      = BASE_DIR / "qc_annotations.csv"
        print(f"[INFO] Using base directory: {BASE_DIR}")

    # Validate critical directories
    for d, label in [(PNG_DIR, "PNG"), (SORT_DIR, "sorting")]:
        if not d.exists():
            print(f"[WARN] Expected directory not found: {d}  ({label})")

    try:
        loader = DatasetLoader(sort_dir=SORT_DIR)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    qc = QCStore(path=QC_CSV)

    viewer_ctrl = HistologyViewer(loader=loader, qc=qc)
    viewer_ctrl.run()


if __name__ == "__main__":
    main()
