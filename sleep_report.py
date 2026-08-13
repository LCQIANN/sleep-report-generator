#!/usr/bin/env python3
"""
Sleep Report Generator
Reads a PSG .edf and a Hypnogram .edf, produces an HTML sleep report.
"""

import sys
import os
import datetime
import warnings
import base64
from io import BytesIO

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe
import mne
from mne.time_frequency import psd_array_welch

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Configuration ───────────────────────────────────────────────────────────
PSG_FILE = os.environ.get("PSG_FILE", "data/SC4002E0-PSG.edf")
HYPNO_FILE = os.environ.get("HYPNO_FILE", "data/SC4002EC-Hypnogram.edf")
OUTPUT_HTML = os.environ.get("OUTPUT_HTML", "sleep_report.html")
EPOCH_SEC = 30

STAGE_MAP = {
    "Sleep stage W": "W",
    "Sleep stage 1": "N1",
    "Sleep stage 2": "N2",
    "Sleep stage 3": "N3",
    "Sleep stage 4": "N3",
    "Sleep stage R": "REM",
    "Sleep stage ?": "?",
    "Movement time": "MT",
}

STAGE_ORDER = ["W", "N1", "N2", "N3", "REM"]
STAGE_LABELS = {"W": "Wake", "N1": "N1", "N2": "N2", "N3": "N3 (SWS)", "REM": "REM"}
STAGE_NUMERIC = {"W": 0, "N1": -1, "N2": -2, "N3": -3, "REM": -4}

COLORS = {
    "W":   "#d9534f",
    "N1":  "#d4a24e",
    "N2":  "#4a90c4",
    "N3":  "#2a5f8a",
    "REM": "#8b6fbf",
    "MT":  "#9ca3af",
    "?":   "#d1d5db",
}

CHART_BG = "#f5f4f0"
CHART_TEXT = "#3b3a37"
CHART_GRID = "#d5d3ce"
CHART_SPINE = "#b8b5ad"


# ── Helpers ─────────────────────────────────────────────────────────────────
def read_hypnogram(hypno_path):
    annot = mne.read_annotations(hypno_path)
    stages, onsets = [], []
    for desc, onset, dur in zip(annot.description, annot.onset, annot.duration):
        label = STAGE_MAP.get(desc)
        if label is None:
            continue
        n_epochs = max(1, int(round(dur / EPOCH_SEC)))
        for i in range(n_epochs):
            stages.append(label)
            onsets.append(onset + i * EPOCH_SEC)
    return np.array(stages), np.array(onsets)


def sleep_metrics(stages, onsets):
    total_epochs = len(stages)
    total_time_min = total_epochs * EPOCH_SEC / 60
    sleep_set = {"N1", "N2", "N3", "REM"}

    stage_minutes = {}
    for s in STAGE_ORDER:
        stage_minutes[s] = np.sum(stages == s) * EPOCH_SEC / 60

    tst = sum(stage_minutes.get(s, 0) for s in sleep_set)
    sleep_mask = np.isin(stages, list(sleep_set))

    if not np.any(sleep_mask):
        return dict(tst=0, total_time=total_time_min, spt=0, efficiency=0,
                    sol=total_time_min, rem_latency=None, awakenings=0,
                    waso=0, stage_minutes=stage_minutes)

    first_sleep = np.argmax(sleep_mask)
    last_sleep = len(sleep_mask) - 1 - np.argmax(sleep_mask[::-1])
    sol = first_sleep * EPOCH_SEC / 60
    spt = (last_sleep - first_sleep + 1) * EPOCH_SEC / 60
    waso = np.sum(stages[first_sleep:last_sleep + 1] == "W") * EPOCH_SEC / 60
    efficiency = (tst / total_time_min) * 100 if total_time_min > 0 else 0

    rem_mask = stages[first_sleep:] == "REM"
    rem_latency = np.argmax(rem_mask) * EPOCH_SEC / 60 if np.any(rem_mask) else None

    awakenings = 0
    ps = stages[first_sleep:last_sleep + 1]
    for i in range(1, len(ps)):
        if ps[i] == "W" and ps[i - 1] != "W":
            awakenings += 1

    return dict(tst=tst, total_time=total_time_min, spt=spt, efficiency=efficiency,
                sol=sol, rem_latency=rem_latency, awakenings=awakenings,
                waso=waso, stage_minutes=stage_minutes)


def fmt_min(m):
    if m is None:
        return "N/A"
    h, mi = divmod(int(m), 60)
    return f"{h}h {mi:02d}m" if h > 0 else f"{mi}m"


def setup_chart_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "sans-serif"],
        "font.size": 10,
        "axes.facecolor": CHART_BG,
        "figure.facecolor": CHART_BG,
        "axes.edgecolor": CHART_SPINE,
        "axes.labelcolor": CHART_TEXT,
        "xtick.color": CHART_TEXT,
        "ytick.color": CHART_TEXT,
        "text.color": CHART_TEXT,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
    })


def fig_to_b64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=170, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none", pad_inches=0.15)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


# ── Plots ───────────────────────────────────────────────────────────────────
def make_hypnogram(stages, onsets):
    fig, ax = plt.subplots(figsize=(14, 3.2))
    numeric = np.array([STAGE_NUMERIC.get(s, 0.5) for s in stages])

    for i in range(len(onsets) - 1):
        x0 = onsets[i] / 3600
        x1 = onsets[i + 1] / 3600
        y = numeric[i]
        c = COLORS.get(stages[i], "#aaa")
        ax.fill_between([x0, x1], y - 0.38, y + 0.38,
                        color=c, alpha=0.55, linewidth=0)

    ax.step(onsets / 3600, numeric, where="post", color="#3b3a37",
            linewidth=0.9, alpha=0.8)

    ax.set_yticks([0, -1, -2, -3, -4])
    ax.set_yticklabels(["Wake", "N1", "N2", "N3", "REM"], fontsize=10, fontweight="500")
    ax.set_xlabel("Hours from recording start", fontsize=10)
    ax.set_xlim(onsets[0] / 3600 - 0.1, onsets[-1] / 3600 + 0.1)
    ax.set_ylim(-4.7, 0.7)
    ax.grid(axis="x", alpha=0.25, color=CHART_GRID, linewidth=0.5)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(left=False, bottom=True)

    fig.tight_layout()
    return fig_to_b64(fig)


def make_donut(stage_minutes):
    fig, ax = plt.subplots(figsize=(5, 5))
    labels, sizes, cols = [], [], []
    for s in STAGE_ORDER:
        m = stage_minutes.get(s, 0)
        if m > 0:
            labels.append(STAGE_LABELS[s])
            sizes.append(m)
            cols.append(COLORS[s])

    wedges, texts, autotexts = ax.pie(
        sizes, labels=None, colors=cols, autopct="%1.1f%%",
        startangle=90, pctdistance=0.82,
        wedgeprops=dict(width=0.38, edgecolor=CHART_BG, linewidth=2.5),
        textprops=dict(fontsize=9),
    )
    for t in autotexts:
        t.set_fontsize(8.5)
        t.set_fontweight("600")
        t.set_color("#3b3a37")

    total = sum(sizes)
    ax.text(0, 0.08, f"{total:.0f}", ha="center", va="center",
            fontsize=24, fontweight="700", color=CHART_TEXT)
    ax.text(0, -0.12, "minutes", ha="center", va="center",
            fontsize=9, color="#7a786f")

    ax.legend(wedges, labels, loc="lower center", ncol=len(labels),
              fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, -0.05),
              handlelength=1.2, handletextpad=0.4, columnspacing=1.2)

    fig.tight_layout()
    return fig_to_b64(fig)


def make_stage_bars(stage_minutes):
    fig, ax = plt.subplots(figsize=(7, 2.8))
    stages_rev = list(reversed(STAGE_ORDER))
    bars = [stage_minutes.get(s, 0) for s in stages_rev]
    bar_colors = [COLORS[s] for s in stages_rev]
    y = range(len(stages_rev))

    bh = ax.barh(y, bars, color=bar_colors, height=0.55, edgecolor="none")
    ax.set_yticks(y)
    ax.set_yticklabels([STAGE_LABELS[s] for s in stages_rev], fontsize=10)
    ax.set_xlabel("Duration (minutes)", fontsize=9.5)

    for i, (v, s) in enumerate(zip(bars, stages_rev)):
        if v > 0:
            ax.text(v + max(bars) * 0.02, i, f"{v:.0f} min",
                    va="center", fontsize=9, color="#5a584f")

    ax.set_xlim(0, max(bars) * 1.18)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.tick_params(left=False)
    ax.grid(axis="x", alpha=0.2, color=CHART_GRID, linewidth=0.5)

    fig.tight_layout()
    return fig_to_b64(fig)


def make_psd(raw, stages, onsets):
    eeg_picks = mne.pick_types(raw.info, eeg=True)
    if len(eeg_picks) == 0:
        return None
    ch_idx = eeg_picks[0]
    ch_name = raw.ch_names[ch_idx]
    sfreq = raw.info["sfreq"]
    data = raw.get_data(picks=[ch_idx])[0]

    fig, axes = plt.subplots(1, 5, figsize=(16, 3.2), sharey=True)
    band_edges = [0.5, 4, 8, 13, 30]
    band_names = ["δ", "θ", "α", "β"]
    band_colors_alpha = ["#8ecae6", "#a7c957", "#ffb703", "#fb8500"]

    for i, stage in enumerate(STAGE_ORDER):
        ax = axes[i]
        mask = stages == stage
        if not np.any(mask):
            ax.text(0.5, 0.5, f"No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color="#9a9890")
            ax.set_title(STAGE_LABELS[stage], fontsize=11, fontweight="600",
                         color=COLORS[stage])
            continue

        epoch_indices = np.where(mask)[0]
        segments = []
        for idx in epoch_indices:
            start = int(onsets[idx] * sfreq)
            stop = start + int(EPOCH_SEC * sfreq)
            if stop <= len(data):
                segments.append(data[start:stop])

        if not segments:
            continue

        concat = np.concatenate(segments)
        psds, freqs = psd_array_welch(
            concat[np.newaxis, :], sfreq=sfreq,
            fmin=0.5, fmax=35, n_fft=int(sfreq * 4),
            n_overlap=int(sfreq * 2), verbose=False,
        )
        psd_db = 10 * np.log10(psds[0] + 1e-20)

        ax.plot(freqs, psd_db, color=COLORS[stage], linewidth=1.6, alpha=0.9)
        ax.fill_between(freqs, psd_db, alpha=0.12, color=COLORS[stage])

        for j, (bname, bcol) in enumerate(zip(band_names, band_colors_alpha)):
            f1, f2 = band_edges[j], band_edges[j + 1]
            bm = (freqs >= f1) & (freqs <= f2)
            if np.any(bm):
                ax.axvspan(f1, f2, alpha=0.06, color=bcol, zorder=0)

        ax.set_xlim(0.5, 35)
        ax.set_xlabel("Hz", fontsize=8.5)
        if i == 0:
            ax.set_ylabel("Power (dB/Hz)", fontsize=9)
        ax.set_title(STAGE_LABELS[stage], fontsize=11, fontweight="600",
                     color=COLORS[stage])
        ax.grid(alpha=0.2, color=CHART_GRID, linewidth=0.4)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(left=(i == 0), bottom=True, length=3)

    band_legend = "    ".join(
        f"δ 0.5–4 Hz    θ 4–8 Hz    α 8–13 Hz    β 13–30 Hz".split("    ")
    )
    fig.text(0.5, -0.02, f"Channel: {ch_name}  ·  {band_legend}",
             ha="center", fontsize=8.5, color="#7a786f")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18, wspace=0.08)
    return fig_to_b64(fig)


# ── HTML ────────────────────────────────────────────────────────────────────
def build_html(metrics, figs, psg_file, hypno_file):
    sm = metrics["stage_minutes"]
    total = sum(sm.get(s, 0) for s in STAGE_ORDER)

    stage_rows = ""
    for s in STAGE_ORDER:
        m = sm.get(s, 0)
        pct = (m / total * 100) if total > 0 else 0
        bar_w = pct
        stage_rows += f"""
        <tr>
          <td><span class="stage-dot" style="background:{COLORS[s]}"></span>{STAGE_LABELS[s]}</td>
          <td class="num">{fmt_min(m)}</td>
          <td class="num">{pct:.1f}%</td>
          <td class="bar-cell"><div class="bar-fill" style="width:{bar_w}%;background:{COLORS[s]}"></div></td>
        </tr>"""

    def metric_block(value, label, sublabel=""):
        sub_html = f'<div class="metric-sub">{sublabel}</div>' if sublabel else ""
        return f"""
        <div class="metric">
          <div class="metric-val">{value}</div>
          <div class="metric-label">{label}</div>
          {sub_html}
        </div>"""

    eff = metrics["efficiency"]
    eff_class = "good" if eff >= 85 else ("fair" if eff >= 75 else "poor")

    psd_section = ""
    if figs.get("psd"):
        psd_section = f"""
    <section class="section">
      <div class="section-head">
        <h2>EEG Power Spectral Density</h2>
        <p class="section-desc">Average power spectrum for each sleep stage, computed from Fpz-Cz via Welch's method</p>
      </div>
      <div class="chart-frame full">
        <img src="data:image/png;base64,{figs['psd']}" alt="PSD per stage">
      </div>
    </section>"""

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Sleep Analysis Report</title>
<style>
:root {{
  --bg: #efeee9;
  --surface: #f5f4f0;
  --card: #fafaf7;
  --text: #2a2926;
  --text2: #6b6860;
  --text3: #9a9890;
  --border: #dbd8d0;
  --accent: #4a5568;
  --indigo: #4f46e5;
  --radius: 10px;
}}
*,*::before,*::after {{ margin:0;padding:0;box-sizing:border-box; }}
body {{
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
               "Helvetica Neue", Arial, sans-serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.55;
  -webkit-font-smoothing: antialiased;
  font-variant-numeric: tabular-nums;
}}

.page {{
  max-width: 980px;
  margin: 0 auto;
  padding: 2.5rem 1.5rem 3rem;
}}

/* ── Header ── */
header {{
  margin-bottom: 2.2rem;
}}
header h1 {{
  font-size: 1.6rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  color: var(--text);
  margin-bottom: 0.35rem;
}}
.header-meta {{
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem 1.2rem;
  font-size: 0.82rem;
  color: var(--text3);
}}
.header-meta span::before {{
  content: "";
  display: inline-block;
  width: 4px; height: 4px;
  border-radius: 50%;
  background: var(--border);
  margin-right: 0.5rem;
  vertical-align: middle;
}}
.header-meta span:first-child::before {{ display: none; }}

/* ── Metrics strip ── */
.metrics {{
  display: grid;
  grid-template-columns: repeat(6, 1fr);
  gap: 1px;
  background: var(--border);
  border-radius: var(--radius);
  overflow: hidden;
  margin-bottom: 2rem;
  box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}}
.metric {{
  background: var(--card);
  padding: 1.1rem 0.8rem;
  text-align: center;
}}
.metric-val {{
  font-size: 1.55rem;
  font-weight: 700;
  letter-spacing: -0.03em;
  color: var(--text);
  line-height: 1.2;
}}
.metric-val.eff-good {{ color: #2d8659; }}
.metric-val.eff-fair {{ color: #b5830a; }}
.metric-val.eff-poor {{ color: #c0392b; }}
.metric-label {{
  font-size: 0.72rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--text3);
  margin-top: 0.3rem;
}}
.metric-sub {{
  font-size: 0.72rem;
  color: var(--text3);
  margin-top: 0.1rem;
}}

/* ── Sections ── */
.section {{
  margin-bottom: 2rem;
}}
.section-head {{
  margin-bottom: 0.8rem;
}}
.section-head h2 {{
  font-size: 1rem;
  font-weight: 650;
  letter-spacing: -0.01em;
}}
.section-desc {{
  font-size: 0.8rem;
  color: var(--text3);
  margin-top: 0.15rem;
}}

.chart-frame {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 0.6rem;
  overflow-x: auto;
}}
.chart-frame img {{
  width: 100%;
  height: auto;
  display: block;
  border-radius: 6px;
}}

/* ── Two-col layout ── */
.split {{
  display: grid;
  grid-template-columns: 340px 1fr;
  gap: 1.5rem;
  align-items: start;
}}

/* ── Table ── */
.stage-table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 0.88rem;
}}
.stage-table th {{
  font-size: 0.7rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--text3);
  padding: 0.5rem 0.7rem;
  text-align: left;
  border-bottom: 2px solid var(--border);
}}
.stage-table td {{
  padding: 0.55rem 0.7rem;
  border-bottom: 1px solid var(--border);
  vertical-align: middle;
}}
.stage-table tr:last-child td {{
  border-bottom: none;
}}
.stage-table .num {{
  text-align: right;
  font-variant-numeric: tabular-nums;
  font-weight: 500;
}}
.stage-dot {{
  display: inline-block;
  width: 10px; height: 10px;
  border-radius: 3px;
  margin-right: 0.5rem;
  vertical-align: middle;
}}
.bar-cell {{
  width: 35%;
  padding-right: 0;
}}
.bar-fill {{
  height: 8px;
  border-radius: 4px;
  min-width: 3px;
}}

.table-foot {{
  display: flex;
  justify-content: space-between;
  padding: 0.6rem 0.7rem 0;
  font-size: 0.82rem;
  font-weight: 600;
  color: var(--text2);
  border-top: 2px solid var(--border);
  margin-top: 0.2rem;
}}

/* ── Footer ── */
footer {{
  text-align: center;
  font-size: 0.75rem;
  color: var(--text3);
  padding-top: 1.5rem;
  border-top: 1px solid var(--border);
}}

@media (max-width: 720px) {{
  .metrics {{ grid-template-columns: repeat(3, 1fr); }}
  .split {{ grid-template-columns: 1fr; }}
  .page {{ padding: 1.2rem 1rem 2rem; }}
}}
@media (max-width: 480px) {{
  .metrics {{ grid-template-columns: repeat(2, 1fr); }}
}}
</style>
</head>
<body>
<div class="page">

  <header>
    <h1>Sleep Analysis Report</h1>
    <div class="header-meta">
      <span>PSG: {os.path.basename(psg_file)}</span>
      <span>Hypnogram: {os.path.basename(hypno_file)}</span>
      <span>{now_str}</span>
    </div>
  </header>

  <div class="metrics">
    {metric_block(f'<span class="eff-{eff_class}">{eff:.1f}%</span>', "Sleep Efficiency")}
    {metric_block(fmt_min(metrics["tst"]), "Total Sleep Time")}
    {metric_block(fmt_min(metrics["sol"]), "Sleep Onset Latency")}
    {metric_block(fmt_min(metrics["rem_latency"]), "REM Latency")}
    {metric_block(str(metrics["awakenings"]), "Awakenings")}
    {metric_block(fmt_min(metrics["waso"]), "WASO")}
  </div>

  <section class="section">
    <div class="section-head">
      <h2>Hypnogram</h2>
      <p class="section-desc">Sleep stages across the full recording period ({metrics['total_time']:.0f} min)</p>
    </div>
    <div class="chart-frame full">
      <img src="data:image/png;base64,{figs['hypnogram']}" alt="Hypnogram">
    </div>
  </section>

  <section class="section">
    <div class="section-head">
      <h2>Sleep Architecture</h2>
    </div>
    <div class="split">
      <div class="chart-frame">
        <img src="data:image/png;base64,{figs['donut']}" alt="Stage distribution donut">
      </div>
      <div>
        <table class="stage-table">
          <thead>
            <tr><th>Stage</th><th style="text-align:right">Duration</th><th style="text-align:right">%</th><th>Distribution</th></tr>
          </thead>
          <tbody>
            {stage_rows}
          </tbody>
        </table>
        <div class="table-foot">
          <span>Total recording</span>
          <span>{fmt_min(total)}</span>
        </div>
      </div>
    </div>
  </section>

  <section class="section">
    <div class="section-head">
      <h2>Stage Duration</h2>
    </div>
    <div class="chart-frame">
      <img src="data:image/png;base64,{figs['bars']}" alt="Stage duration bar chart">
    </div>
  </section>

  {psd_section}

  <footer>
    Generated with MNE-Python &amp; Matplotlib
  </footer>

</div>
</body>
</html>"""


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    psg_file = PSG_FILE
    hypno_file = HYPNO_FILE
    output = OUTPUT_HTML

    if len(sys.argv) >= 3:
        psg_file, hypno_file = sys.argv[1], sys.argv[2]
    if len(sys.argv) >= 4:
        output = sys.argv[3]

    print(f"Reading PSG:       {psg_file}")
    print(f"Reading Hypnogram: {hypno_file}")

    raw = mne.io.read_raw_edf(psg_file, preload=True, verbose=False)
    stages, onsets = read_hypnogram(hypno_file)

    valid = onsets < raw.times[-1]
    stages, onsets = stages[valid], onsets[valid]
    print(f"Epochs: {len(stages)}, Duration: {len(stages)*EPOCH_SEC/3600:.1f} h")

    metrics = sleep_metrics(stages, onsets)
    print(f"TST: {fmt_min(metrics['tst'])}, Efficiency: {metrics['efficiency']:.1f}%")

    setup_chart_style()
    print("Generating charts...")
    figs = {
        "hypnogram": make_hypnogram(stages, onsets),
        "donut": make_donut(metrics["stage_minutes"]),
        "bars": make_stage_bars(metrics["stage_minutes"]),
        "psd": make_psd(raw, stages, onsets),
    }

    print("Building HTML...")
    html = build_html(metrics, figs, psg_file, hypno_file)
    with open(output, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Report saved to: {output}")


if __name__ == "__main__":
    main()
