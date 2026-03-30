"""
analyze_counts.py
=================
Analysis companion for histology_qc_viewer.py.

Loads cell_counts.csv and section_manifest.csv, normalizes GFP+ neuron
counts per rat to account for unequal section sampling, then generates
publication-style bar graphs with individual data points and SEM error bars.

Normalization rationale
-----------------------
Not every rat contributes the same number of sections to the dataset.
Sections with zero counted cells are valid observations, not missing data —
they just never produced a row in cell_counts.csv because nothing was
clicked.  The denominator for each rat is therefore the number of sections
marked include=1 in section_manifest.csv, NOT the number of rows in
cell_counts.csv.

Primary normalised metric:
    cells_per_section = total_clicked_cells / n_included_sections

This gives a mean cells-per-section value for each rat that is comparable
across animals regardless of how many sections were imaged.

Usage
-----
  # Per-rat bar graph (one bar per rat, GFP channel by default)
  python analyze_counts.py --per-rat

  # Per-rat for a specific channel
  python analyze_counts.py --per-rat --channel CH3

  # Per-group comparison (prompts for group definitions in CLI)
  python analyze_counts.py --per-group

  # Both plots in one run
  python analyze_counts.py --per-rat --per-group

  # Point at a different cohort directory
  python analyze_counts.py --per-rat --base ~/Microscopy/Cohort2_5HT

  # Save figures instead of displaying
  python analyze_counts.py --per-rat --save

  # Override default GFP channel key
  python analyze_counts.py --per-rat --channel CH2

Optional flags
--------------
  --base PATH       Base cohort directory (default: ~/Microscopy/Cohort1_TPH2)
  --channel KEY     Channel to analyse, e.g. CH1 CH2 CH3 CH4 (default: CH2)
  --per-rat         Generate per-rat normalised bar graph
  --per-group       Generate per-group comparison bar graph (prompts for groups)
  --save            Write figures to <BASE_DIR>/figures/ instead of showing
  --no-normalise    Plot raw total counts instead of per-section-normalised values
  --dpi N           Figure DPI for saved files (default: 300)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib import rcParams

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

BASE_DIR     = Path.home() / "Microscopy" / "Cohort1_TPH2"
COUNTS_CSV   = BASE_DIR / "cell_counts.csv"
MANIFEST_CSV = BASE_DIR / "section_manifest.csv"
FIGURES_DIR  = BASE_DIR / "figures"

DEFAULT_CHANNEL = "CH2"   # GFP / TPH2

CH_LABELS = {
    "CH1": "DAPI",
    "CH2": "TPH2/GFP",
    "CH3": "RFP",
    "CH4": "Cy5",
}

# ---------------------------------------------------------------------------
# Plot aesthetics
# ---------------------------------------------------------------------------

BAR_COLOR      = "#4C9BE8"   # muted blue
DOT_COLOR      = "#1A1A2E"   # near-black
DOT_ALPHA      = 0.75
DOT_SIZE       = 7
DOT_JITTER     = 0.07        # horizontal jitter width for individual dots
BAR_ALPHA      = 0.72
BAR_WIDTH      = 0.55
CAPSIZE        = 4
ERROR_COLOR    = "#222222"
SPINE_COLOR    = "#333333"
FONT_FAMILY    = "sans-serif"

GROUP_PALETTE  = [
    "#4C9BE8",  # blue
    "#E8704C",  # orange
    "#4CE87A",  # green
    "#E84C9B",  # pink
    "#9B4CE8",  # purple
    "#E8C94C",  # gold
]

def _apply_style() -> None:
    rcParams["font.family"]      = FONT_FAMILY
    rcParams["axes.spines.top"]  = False
    rcParams["axes.spines.right"]= False
    rcParams["axes.linewidth"]   = 1.2
    rcParams["xtick.major.width"]= 1.2
    rcParams["ytick.major.width"]= 1.2
    rcParams["xtick.direction"]  = "out"
    rcParams["ytick.direction"]  = "out"
    rcParams["figure.dpi"]       = 120


# ---------------------------------------------------------------------------
# Data loading & normalisation
# ---------------------------------------------------------------------------

def load_data(
    counts_csv: Path,
    manifest_csv: Path,
    channel: str,
) -> pd.DataFrame:
    """
    Return a per-rat DataFrame with columns:
        rat, n_sections, raw_count, cells_per_section

    n_sections  : number of sections marked include=1 in the manifest
                  (the denominator — includes zero-count sections)
    raw_count   : total clicked cells for this rat × channel
    cells_per_section : raw_count / n_sections
    """

    # --- manifest: section denominator per rat ---
    if not manifest_csv.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_csv}\n"
            f"Run  generate_manifest.py  first."
        )
    manifest = pd.read_csv(manifest_csv, dtype={"include": int})
    # Only count sections the user marked for quantification
    included = manifest[manifest["include"] == 1]
    section_counts = (
        included.groupby("rat")
        .size()
        .reset_index(name="n_sections")
    )

    # --- cell counts: sum per rat × channel ---
    if not counts_csv.exists():
        raise FileNotFoundError(
            f"Cell counts not found: {counts_csv}\n"
            f"Run the viewer and count some cells first."
        )
    counts = pd.read_csv(counts_csv)

    if counts.empty:
        raise RuntimeError("cell_counts.csv is empty — no cells have been counted yet.")

    ch_counts = counts[counts["channel"] == channel].copy()

    if ch_counts.empty:
        ch_label = CH_LABELS.get(channel, channel)
        raise RuntimeError(
            f"No counts found for channel {channel} ({ch_label}).\n"
            f"Available channels in file: {sorted(counts['channel'].unique().tolist())}"
        )

    raw = (
        ch_counts.groupby("rat")
        .size()
        .reset_index(name="raw_count")
    )

    # Merge: left join on section_counts so every rat with included sections
    # appears, even if they have zero counts in cell_counts.csv
    df = section_counts.merge(raw, on="rat", how="left")
    df["raw_count"] = df["raw_count"].fillna(0).astype(int)
    df["cells_per_section"] = df["raw_count"] / df["n_sections"]

    df = df.sort_values("rat").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Shared plot helpers
# ---------------------------------------------------------------------------

def _draw_bars_with_dots(
    ax: plt.Axes,
    positions: list[float],
    values: list[float],        # one value per bar (mean or single rat value)
    errors: list[float],        # SEM per bar (0 for single-rat case)
    labels: list[str],
    colors: list[str],
    ylabel: str,
    title: str,
    dot_values: list[list[float]] | None = None,  # raw dots per bar group
) -> None:
    """
    Core bar + dot + error-bar renderer used by both --per-rat and --per-group.
    """
    rng = np.random.default_rng(seed=42)

    for i, (pos, val, err, label, color) in enumerate(
        zip(positions, values, errors, labels, colors)
    ):
        ax.bar(
            pos, val,
            width=BAR_WIDTH,
            color=color,
            alpha=BAR_ALPHA,
            zorder=2,
        )
        if err > 0:
            ax.errorbar(
                pos, val,
                yerr=err,
                fmt="none",
                color=ERROR_COLOR,
                capsize=CAPSIZE,
                capthick=1.4,
                linewidth=1.4,
                zorder=3,
            )
        # Individual dots
        dots = dot_values[i] if dot_values is not None else [val]
        jitter = rng.uniform(-DOT_JITTER, DOT_JITTER, size=len(dots))
        ax.scatter(
            [pos + j for j in jitter],
            dots,
            color=DOT_COLOR,
            s=DOT_SIZE ** 2 / 4,
            alpha=DOT_ALPHA,
            zorder=4,
            linewidths=0,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, pad=10)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())

    # Push x-axis down slightly
    ax.tick_params(axis="x", pad=4)
    # Ensure y starts at 0
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(bottom=0, top=ymax * 1.12)

    for spine in ax.spines.values():
        spine.set_color(SPINE_COLOR)


# ---------------------------------------------------------------------------
# --per-rat
# ---------------------------------------------------------------------------

def plot_per_rat(
    df: pd.DataFrame,
    channel: str,
    normalise: bool,
    save: bool,
    figures_dir: Path,
    dpi: int,
) -> None:
    ch_label = CH_LABELS.get(channel, channel)
    metric   = "cells_per_section" if normalise else "raw_count"
    ylabel   = (
        f"{ch_label}+ cells per section"
        if normalise
        else f"{ch_label}+ cells (raw total)"
    )
    title = (
        f"{'Normalised' if normalise else 'Raw'} {ch_label}+ neuron count per rat"
    )

    _apply_style()
    fig, ax = plt.subplots(figsize=(max(5, len(df) * 0.9 + 1.5), 4.5))

    positions  = list(range(len(df)))
    values     = df[metric].tolist()
    errors     = [0.0] * len(df)          # single point per rat — no SEM
    labels     = df["rat"].tolist()
    colors     = [BAR_COLOR] * len(df)

    _draw_bars_with_dots(
        ax, positions, values, errors, labels, colors,
        ylabel=ylabel,
        title=title,
        dot_values=[[v] for v in values],  # one dot per bar (the value itself)
    )

    # Annotate n_sections below each bar label
    for pos, row in zip(positions, df.itertuples()):
        ax.text(
            pos, -ax.get_ylim()[1] * 0.06,
            f"n={row.n_sections}",
            ha="center", va="top", fontsize=7.5, color="#666666",
        )

    fig.tight_layout()
    _finish(fig, save, figures_dir, f"per_rat_{channel}.pdf", dpi)


# ---------------------------------------------------------------------------
# --per-group
# ---------------------------------------------------------------------------

def prompt_groups(rats: list[str]) -> dict[str, list[str]]:
    """
    Interactive CLI prompt. Returns { group_name: [rat, rat, ...] }.
    """
    print("\n" + "─" * 55)
    print("  GROUP DEFINITION")
    print("─" * 55)
    print(f"  Rats available: {', '.join(rats)}")
    print("  Enter group name then the rats belonging to it.")
    print("  Press Enter on an empty group name when done.\n")

    groups: dict[str, list[str]] = {}
    rat_set = set(rats)

    while True:
        try:
            gname = input("  Group name (blank = done): ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not gname:
            break

        raw = input(f"  Rats in '{gname}' (comma-separated): ").strip()
        members = [r.strip() for r in raw.split(",") if r.strip()]
        unknown = [r for r in members if r not in rat_set]
        if unknown:
            print(f"  [WARN] Unknown rat(s) ignored: {unknown}")
            members = [r for r in members if r in rat_set]
        if not members:
            print(f"  [WARN] No valid rats — group '{gname}' skipped.")
            continue
        groups[gname] = members
        print(f"  → '{gname}': {members}\n")

    if not groups:
        raise RuntimeError("No groups defined — aborting per-group plot.")

    # Confirm
    print("\n  Groups confirmed:")
    for g, members in groups.items():
        print(f"    {g}: {members}")
    print()
    return groups


def plot_per_group(
    df: pd.DataFrame,
    channel: str,
    normalise: bool,
    save: bool,
    figures_dir: Path,
    dpi: int,
) -> None:
    ch_label = CH_LABELS.get(channel, channel)
    metric   = "cells_per_section" if normalise else "raw_count"
    ylabel   = (
        f"{ch_label}+ cells per section"
        if normalise
        else f"{ch_label}+ cells (raw total)"
    )

    rats   = df["rat"].tolist()
    groups = prompt_groups(rats)

    # Build per-group stats
    group_names  = list(groups.keys())
    group_means  = []
    group_sems   = []
    group_dots   = []   # raw per-rat values for each group
    group_ns     = []

    for gname in group_names:
        members = groups[gname]
        vals = df.loc[df["rat"].isin(members), metric].tolist()
        # Rats listed in a group but missing from df get 0 — shouldn't happen
        # but protects against typos that survived the prompt filter
        n    = len(vals)
        mean = float(np.mean(vals)) if vals else 0.0
        sem  = float(np.std(vals, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        group_means.append(mean)
        group_sems.append(sem)
        group_dots.append(vals)
        group_ns.append(n)

    title = (
        f"{'Normalised' if normalise else 'Raw'} {ch_label}+ neuron count by group"
    )

    _apply_style()
    fig, ax = plt.subplots(figsize=(max(5, len(group_names) * 1.6 + 1.5), 4.5))

    positions = list(range(len(group_names)))
    colors    = [GROUP_PALETTE[i % len(GROUP_PALETTE)] for i in range(len(group_names))]

    _draw_bars_with_dots(
        ax, positions, group_means, group_sems, group_names, colors,
        ylabel=ylabel,
        title=title,
        dot_values=group_dots,
    )

    # Annotate n per group
    for pos, n in zip(positions, group_ns):
        ax.text(
            pos, -ax.get_ylim()[1] * 0.06,
            f"n={n}",
            ha="center", va="top", fontsize=8, color="#666666",
        )

    fig.tight_layout()
    _finish(fig, save, figures_dir, f"per_group_{channel}.pdf", dpi)


# ---------------------------------------------------------------------------
# Save / show helper
# ---------------------------------------------------------------------------

def _finish(
    fig: plt.Figure,
    save: bool,
    figures_dir: Path,
    filename: str,
    dpi: int,
) -> None:
    if save:
        figures_dir.mkdir(parents=True, exist_ok=True)
        out = figures_dir / filename
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"[INFO] Figure saved → {out}")
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Analyse and plot normalised cell counts from histology_qc_viewer."
    )
    p.add_argument(
        "--base",
        type=Path,
        default=None,
        metavar="PATH",
        help="Base cohort directory (default: ~/Microscopy/Cohort1_TPH2)",
    )
    p.add_argument(
        "--channel",
        default=DEFAULT_CHANNEL,
        metavar="CH",
        help=f"Channel key to analyse (default: {DEFAULT_CHANNEL}  = GFP/TPH2)",
    )
    p.add_argument(
        "--per-rat",
        action="store_true",
        help="Generate per-rat normalised bar graph",
    )
    p.add_argument(
        "--per-group",
        action="store_true",
        help="Generate per-group comparison bar graph (prompts for group definitions)",
    )
    p.add_argument(
        "--save",
        action="store_true",
        help="Save figures to <BASE_DIR>/figures/ instead of displaying",
    )
    p.add_argument(
        "--no-normalise",
        action="store_true",
        help="Plot raw total counts instead of per-section-normalised values",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=300,
        metavar="N",
        help="DPI for saved figures (default: 300)",
    )
    return p


def main() -> None:
    global BASE_DIR, COUNTS_CSV, MANIFEST_CSV, FIGURES_DIR

    parser = build_parser()
    args   = parser.parse_args()

    if not args.per_rat and not args.per_group:
        parser.print_help()
        print("\n[ERROR] Specify at least one of --per-rat or --per-group.")
        sys.exit(1)

    if args.base is not None:
        BASE_DIR     = args.base.expanduser().resolve()
        COUNTS_CSV   = BASE_DIR / "cell_counts.csv"
        MANIFEST_CSV = BASE_DIR / "section_manifest.csv"
        FIGURES_DIR  = BASE_DIR / "figures"

    normalise = not args.no_normalise
    channel   = args.channel.upper()

    print(f"[INFO] Base dir  : {BASE_DIR}")
    print(f"[INFO] Channel   : {channel} ({CH_LABELS.get(channel, '?')})")
    print(f"[INFO] Normalise : {normalise}")

    try:
        df = load_data(COUNTS_CSV, MANIFEST_CSV, channel)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"\n[ERROR] {exc}")
        sys.exit(1)

    print(f"\n[INFO] Per-rat summary ({channel}):")
    print(df[["rat", "n_sections", "raw_count", "cells_per_section"]].to_string(index=False))
    print()

    if args.per_rat:
        plot_per_rat(
            df,
            channel=channel,
            normalise=normalise,
            save=args.save,
            figures_dir=FIGURES_DIR,
            dpi=args.dpi,
        )

    if args.per_group:
        plot_per_group(
            df,
            channel=channel,
            normalise=normalise,
            save=args.save,
            figures_dir=FIGURES_DIR,
            dpi=args.dpi,
        )


if __name__ == "__main__":
    main()
