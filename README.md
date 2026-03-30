# histoview

A napari-based multichannel fluorescence histology QC viewer with integrated cell counting and quantification.  Built for rat spinal cord cohorts but dataset-agnostic.

---

## What it does

| Script | Purpose |
|---|---|
| `histology_qc_viewer.py` | Load, navigate, and QC multichannel section images; right-click to count cells per channel |
| `generate_manifest.py` | Scan your dataset and produce a CSV where you select which sections to quantify |
| `analyze_counts.py` | Normalize counts across unequal section numbers and plot per-rat or per-group bar graphs |

---

## Requirements

```
python >= 3.9
napari
numpy
pandas
imageio
matplotlib
tomli          # only if Python < 3.11
```

Install into a fresh environment:

```bash
pip install napari[all] numpy pandas imageio matplotlib
# Python < 3.11 only:
pip install tomli
```

---

## Quick start

### 1. Clone and configure

```bash
git clone https://github.com/yourname/histoview.git
cd histoview
python histology_qc_viewer.py --init-config
```

This writes a `config.toml` next to the script.  Open it and set `base_dir` to your cohort root:

```toml
[paths]
base_dir   = "~/Microscopy/MyCohort"
png_subdir = "PNG_fullsize"

[channels]
CH1 = "DAPI"
CH2 = "GFP"
CH3 = "RFP"
CH4 = "Cy5"
```

### 2. Expected directory layout

```
<base_dir>/
    PNG_fullsize/               ← configurable via png_subdir
        <ratID>/
            CH1/  CH2/  CH3/  CH4/
                <ratID>_<region>_XY##_CH1.png
                ...
    features/
        sorting/
            predicted_order_<ratID>.csv   ← one per rat; column: "image"
```

The `predicted_order_<ratID>.csv` files drive section ordering.  Each file needs an `image` column containing the CH1 stem string (e.g. `rat761_RBS_XY35_CH1`).

### 3. Launch the viewer

```bash
python histology_qc_viewer.py
```

Or override the base directory at runtime without editing config:

```bash
python histology_qc_viewer.py --base ~/Microscopy/Cohort2
```

---

## Keyboard map

| Key | Action |
|---|---|
| `←` / `→` | Previous / next section |
| `[` / `]` | Previous / next rat |
| `1` – `4` | Show CH1–CH4 only (also sets active counting channel) |
| `Space` | Toggle overlay (all channels) ↔ single-channel |
| `m` | Toggle MIRROR flag |
| `r` | Toggle ROTATE flag |
| `q` | Toggle QUANTIFY flag |
| `n` | Enter / edit notes (console prompt) |
| `z` | Undo last count marker on active channel |
| `p` | Print count summary for current rat |
| `Ctrl+S` | Force-save QC annotations + cell counts |

**Right-click** anywhere on the image to place a cell marker on the active channel.

---

## Cell counting workflow

1. Press `1`–`4` to select the channel you are counting (shown in the title bar).
2. Zoom in with the scroll wheel.
3. Right-click each cell body to place a marker.  A coloured dot appears and the count is recorded in `cell_counts.csv`.
4. Press `z` to undo the last marker on the active channel.
5. Press `p` or `Ctrl+S` to review / save.

Output file: `<base_dir>/cell_counts.csv`

```
rat, section, xy_label, channel, ch_label, x, y, timestamp
```

The `x` / `y` coordinates are in image pixel space and are suitable as input to CellProfiler pipelines or DLC model training.

---

## Section manifest

Generate a manifest of all sections across all rats:

```bash
python generate_manifest.py
# or for a different cohort:
python generate_manifest.py ~/Microscopy/Cohort2
```

This writes `<base_dir>/section_manifest.csv`.  Open it in Excel or any CSV editor, set `include = 0` on any section you want to skip, then launch the viewer in manifest-only mode:

```bash
python histology_qc_viewer.py --manifest-only
```

Re-running `generate_manifest.py` after adding new rats preserves all existing `include` and `notes` values.

---

## Analysis

```bash
# Per-rat bar graph (GFP channel, normalized to sections counted)
python analyze_counts.py --per-rat

# Per-group comparison (prompts for group assignment in CLI)
python analyze_counts.py --per-group

# Both at once, saved as PDF
python analyze_counts.py --per-rat --per-group --save

# Different channel
python analyze_counts.py --per-rat --channel CH3

# Raw totals instead of normalized
python analyze_counts.py --per-rat --no-normalise
```

### Normalization

Many sections will have zero counted cells.  These are real zero observations, not missing data, and they are included in each rat's denominator.  The primary metric is:

```
cells_per_section = total_clicks / n_included_sections
```

where `n_included_sections` comes from the manifest (`include = 1` rows), not from `cell_counts.csv`.  This keeps per-rat values comparable regardless of how many sections were imaged per animal.

---

## CLI reference

```
histology_qc_viewer.py
  --base PATH         Override base cohort directory
  --png-subdir DIR    PNG subfolder name (default: PNG_fullsize)
  --sort-subdir DIR   Sorting CSV subfolder (default: features/sorting)
  --manifest-only     Load only manifest-included sections
  --config PATH       Path to a specific config.toml
  --init-config       Write starter config.toml and exit

generate_manifest.py
  [BASE_DIR]          Optional base directory override

analyze_counts.py
  --base PATH         Base cohort directory
  --channel CH        CH1–CH4 (default: CH2)
  --per-rat           Per-rat bar graph
  --per-group         Per-group bar graph (interactive group definition)
  --save              Save figures to <base_dir>/figures/
  --no-normalise      Plot raw totals instead of cells_per_section
  --dpi N             Figure DPI (default: 300)
```

---

## Repository suggestions

See [SUGGESTIONS.md](SUGGESTIONS.md) for notes on where this fits in a larger pipeline repo versus a standalone tool.
