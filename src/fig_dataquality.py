#!/usr/bin/env python3
"""
Data-quality figure -- v2, updated for the two-arm Italy design.

CHANGES FROM v1 (fig_dataquality.py as originally attached)
    1. v1 was Ghana-ONLY, hardcoded to ghana_national_master.csv and its
       Ghana-specific columns (daily_incidence, observed_simple, gap_length --
       produced by prepare_ghana.py's missingness audit). Italy's ISS series
       has no such gap-audit columns because it is a complete daily feed;
       plotting Italy the same way would either crash on missing columns or
       misleadingly show "0% missing" without context.
    2. Now renders THREE panels: Ghana (gap-annotated, unchanged from v1),
       Italy PRIMARY (single-wave, r=6-valid window), and Italy SECONDARY
       (full multi-wave range) -- with the secondary panel explicitly
       annotated as a regime-shift stress-test arm, not a like-for-like
       "high-quality" comparison. This keeps the figure consistent with the
       two-tier Italy design agreed in the pipeline discussion.
    3. Ghana's plot logic (fill_between, gap bars, annotation) is preserved
       EXACTLY as in v1 -- only the layout (now 3 stacked mini-figures) and
       the two new Italy panels are added.

Usage: python3 fig_dataquality.py
    (reads data/processed/ghana_national_master.csv,
           data/processed/italy_primary_national_master.csv,
           data/processed/italy_secondary_national_master.csv)
"""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

OUT = Path("results/figures/fig_data_quality.png")
OUT.parent.mkdir(parents=True, exist_ok=True)

ghana_path = Path("data/processed/ghana_national_master.csv")
italy_primary_path = Path("data/processed/italy_primary_national_master.csv")
italy_secondary_path = Path("data/processed/italy_secondary_national_master.csv")

n_panels = 1 + int(italy_primary_path.exists()) + int(italy_secondary_path.exists())
fig = plt.figure(figsize=(9, 3.6 * n_panels))
gs_top = fig.add_gridspec(n_panels, 1, hspace=0.55)
row = 0


def panel_ghana(outer_gs):
    m = pd.read_csv(ghana_path, index_col=0, parse_dates=True)
    inner = outer_gs.subgridspec(2, 1, height_ratios=[3, 1], hspace=0.12)
    ax0 = fig.add_subplot(inner[0]); ax1 = fig.add_subplot(inner[1], sharex=ax0)

    ax0.fill_between(m.index, m.daily_incidence, color='#c8d8e8', lw=0)
    ax0.plot(m.index, m.daily_incidence, color='#2c5f8a', lw=0.8)
    for d in m.index[m.observed_simple == 0]:
        ax0.axvline(d, color='#d94f4f', alpha=0.16, lw=1.4)
    ax0.set_ylabel('Daily reported cases')
    ax0.set_title('Ghana COVID-19 national incidence with unreported days shaded '
                  '(HERA, 12 Mar 2020 - 19 May 2021)', fontsize=10, loc='left')
    unreported_pct = 100 * (m.observed_simple == 0).mean()
    h = [plt.Line2D([], [], color='#2c5f8a', lw=1.2, label='Reported incidence'),
         plt.Line2D([], [], color='#d94f4f', lw=4, alpha=0.3,
                    label=f'No report ({(m.observed_simple==0).sum()} d, {unreported_pct:.1f}%)')]
    ax0.legend(handles=h, fontsize=8, frameon=False, loc='upper left')

    gl = m.gap_length.where(m.gap_length > 0, 0) if "gap_length" in m.columns else pd.Series(0, index=m.index)
    ax1.bar(m.index, gl, color='#d94f4f', width=1.0)
    ax1.set_ylabel('Gap\nlength (d)', fontsize=8)
    ax1.set_xlabel('Date')
    if gl.max() > 0:
        peak_date = gl.idxmax()
        ax1.annotate(f'{int(gl.max())}-day gap', xy=(peak_date, gl.max()),
                    xytext=(peak_date, gl.max() * 0.85), fontsize=7.5,
                    arrowprops=dict(arrowstyle='->', lw=0.7, color='#555'))
    for a in (ax0, ax1):
        a.spines[['top', 'right']].set_visible(False)


def panel_italy(outer_gs, path, label, note, color):
    m = pd.read_csv(path, index_col=0, parse_dates=True)
    ax = fig.add_subplot(outer_gs)
    ax.fill_between(m.index, m.I_proxy, color=color, alpha=0.25, lw=0)
    ax.plot(m.index, m.I_proxy, color=color, lw=0.9)
    ax.set_ylabel('I_proxy (corrected)')
    ax.set_title(label, fontsize=10, loc='left')
    ax.annotate(note, xy=(0.99, 0.92), xycoords='axes fraction',
               ha='right', va='top', fontsize=7.5, style='italic', color='#555')
    ax.spines[['top', 'right']].set_visible(False)


panel_ghana(gs_top[row]); row += 1

if italy_primary_path.exists():
    m = pd.read_csv(italy_primary_path, index_col=0, parse_dates=True)
    date_range = f"{m.index.min().date()} to {m.index.max().date()}"
    panel_italy(gs_top[row], italy_primary_path,
               f'Italy PRIMARY arm -- single-wave, r=6-valid window ({date_range})',
               'Clean benchmark: Objectives 1 & 2', '#2c8a5f')
    row += 1

if italy_secondary_path.exists():
    m = pd.read_csv(italy_secondary_path, index_col=0, parse_dates=True)
    date_range = f"{m.index.min().date()} to {m.index.max().date()}"
    panel_italy(gs_top[row], italy_secondary_path,
               f'Italy SECONDARY arm -- full multi-wave range ({date_range})',
               'Regime-shift robustness stress test ONLY -- Objective 3, '
               'NOT a clean data-richness comparison', '#8a2c5f')
    row += 1

fig.savefig(OUT, dpi=170, bbox_inches='tight')
print(f'ok: wrote {OUT}')
