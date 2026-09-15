# -*- coding: utf-8 -*-
"""
Per-stride EMG from the RESTORES walking recordings (P4-P6 CSV export).

    python run_walking_emg.py P4 [P5] [--dates 2026-01-28 ...] [--max-seconds N]
                              [--out outputs/walking_emg]

The exercise time tabs in a walking file are per-stride markers: the P4
28 Jan 2026 video shows 22 right heel strikes for the file named "22 full
gaits" with 22 tabs ~5 s apart (see gait_video).  Each tab-to-tab interval
is treated as one stride, kept when its duration lies within 0.5-2x the
file's median, and every channel is cut into those strides.

Per channel the envelope is divided by that channel's own activation
threshold (1.5 x p99 of the condition-matched rest recording, see
emg_pipeline.detect_activation).  Rest and trial share the amplifier range
within a session, so this ratio is gain-free and comparable across sessions
even though absolute microvolts are not (no gain log exists for P4-P6).

Outputs, per walking file in <out>/<patient>_<date>_<stem>/:
    strides.csv    one row per stride x channel: duration, mean / peak
                   envelope (raw and threshold-normalised), active fraction,
                   time of peak (% stride)
    cycles.csv     0-100 % ensemble mean / sd per channel (normalised)
    cycles.png     4x4 grid, left and right of each muscle overlaid
    summary.json   counts, stride timing, per-muscle activation and left /
                   right symmetry, cycle consistency
and <out>/walking_emg_summary.csv over everything processed.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

import emg_pipeline as ep
import restores_csv as rc

ENV_WINDOW_S = 0.1
OUT_FS = 100                       # envelope storage rate (Hz)
N_PTS = 101
# substrings of the acquisition's channel names ('Lt Vastus Laterali' is truncated in the export)
MUSCLES = ["Psoas", "Rectus Femoris", "Vastus Laterali", "Tibialis Anterior",
           "Gluteus Maximus", "Bicep Femoris", "Gastroc", "Soleus"]


RATE_WINDOW_S = 60                 # the rate is constant within a file; detection cost scales with length


def stim_rate(rec, rest):
    cc = list(range(len(rec.channels)))
    n = int(RATE_WINDOW_S * rec.fs)
    rate_t, _, method_t, _ = ep.pulse_train_consensus(rec.data[:, :n], rec.fs, cc)
    rate_r, _, method_r, _ = (ep.pulse_train_consensus(rest.data[:, :n], rest.fs, cc) if rest is not None
                              else (None, 0, None, {}))
    if ep.is_environmental(rate_t, method_t):
        rate_t, method_t = None, None
    if ep.is_environmental(rate_r, method_r):
        rate_r, method_r = None, None
    return (rate_t, method_t) if rate_t is not None else (rate_r, method_r)


def stride_bounds(rec):
    """Tab-to-tab intervals as (start, end) sample pairs, with the odd ones flagged."""
    tabs = rec.tab_samples()
    if len(tabs) < 2:
        return [], np.array([])
    d = np.diff(tabs) / rec.fs
    med = np.median(d)
    ok = (d >= 0.5 * med) & (d <= 2.0 * med)
    return [(a, b, bool(k)) for a, b, k in zip(tabs[:-1], tabs[1:], ok)], d


def analyze_file(sf, rest, rec, out_dir, label):
    stim_hz, method = stim_rate(rec, rest)
    blank = method == "pulses"
    bounds, gaps = stride_bounds(rec)
    used = [(a, b) for a, b, k in bounds if k]
    step = rec.fs // OUT_FS
    grid = np.linspace(0, 1, N_PTS)

    rows, cyc_rows, chan_summary = [], [], {}
    cycles = {}
    for ch, name in enumerate(rec.channels):
        prep = ep.remove_artifact(rec.data[ch], rec.fs, stim_hz=stim_hz, blank=blank)
        if rest is not None and ch < rest.data.shape[0]:
            prep_r = ep.remove_artifact(rest.data[ch], rest.fs, stim_hz=stim_hz, blank=blank)
            mask, thr = ep.detect_activation(prep, prep_r, fs=rec.fs, window_s=ENV_WINDOW_S)
        else:
            mask, thr = np.zeros(len(prep), bool), np.nan
        env = ep.moving_rms(prep, rec.fs, window_s=ENV_WINDOW_S)
        norm = env / thr if np.isfinite(thr) and thr > 0 else np.full_like(env, np.nan)
        curves = []
        for i, (a, b) in enumerate(used):
            seg, segn, segm = env[a:b], norm[a:b], mask[a:b]
            if len(seg) < 10 or not np.isfinite(seg).any():
                continue
            k = int(np.nanargmax(seg))
            rows.append({"stride": i, "t_start": a / rec.fs, "duration_s": (b - a) / rec.fs, "channel": ch,
                         "muscle": name, "env_mean": float(np.nanmean(seg)), "env_peak": float(seg[k]),
                         "norm_mean": float(np.nanmean(segn)), "norm_peak": float(segn[k]),
                         "active_frac": float(segm.mean()), "peak_pct": 100.0 * k / len(seg)})
            x = np.linspace(0, 1, len(segn[::step]))
            y = segn[::step]
            if np.isfinite(y).all() and len(y) > 3:
                curves.append(np.interp(grid, x, y))
        cycles[ch] = np.array(curves) if curves else np.empty((0, N_PTS))
        if len(curves):
            m, sd = cycles[ch].mean(axis=0), cycles[ch].std(axis=0)
            cyc_rows.append(pd.DataFrame({"channel": ch, "muscle": name, "pct": grid * 100, "mean": m, "sd": sd,
                                          "n": len(curves)}))
            r = [np.corrcoef(c, m)[0, 1] for c in cycles[ch]] if len(curves) > 1 else [np.nan]
            chan_summary[name] = {"threshold": thr, "norm_mean": float(np.nanmean(cycles[ch])),
                                  "active_frac": float(np.mean([x["active_frac"] for x in rows if x["channel"] == ch])),
                                  "consistency_r": float(np.nanmean(r)),
                                  "peak_pct_median": float(np.median([x["peak_pct"] for x in rows if x["channel"] == ch]))}

    os.makedirs(out_dir, exist_ok=True)
    strides = pd.DataFrame(rows)
    strides.to_csv(os.path.join(out_dir, "strides.csv"), index=False, float_format="%.4f")
    (pd.concat(cyc_rows) if cyc_rows else pd.DataFrame()).to_csv(os.path.join(out_dir, "cycles.csv"), index=False,
                                                                   float_format="%.4f")
    summary = {**label, "file": sf.name, "stim_on": sf.stim_on, "stim_hz": stim_hz, "stim_method": method,
               "baseline": None if rest is None else rest.name, "gaits_named": sf.gaits, "harness": sf.harness,
               "brace": sf.brace,
               "duration_s": round(rec.duration_s, 1), "tabs_total": len(rec.tabs), "tabs_in_clip": len(rec.tab_samples()),
               "strides_used": len(used), "strides_rejected": len(bounds) - len(used),
               "stride_time_median_s": round(float(np.median(gaps)), 2) if len(gaps) else None,
               "stride_time_cv": round(float(np.std(gaps) / np.mean(gaps)), 3) if len(gaps) > 1 else None}
    for name, d in chan_summary.items():
        key = name.replace(" ", "_")
        summary[f"{key}_norm_mean"] = round(d["norm_mean"], 3)
        summary[f"{key}_active_frac"] = round(d["active_frac"], 3)
        summary[f"{key}_consistency_r"] = round(d["consistency_r"], 3)
        summary[f"{key}_peak_pct"] = round(d["peak_pct_median"], 1)
    for m in MUSCLES:
        l = next((d for n, d in chan_summary.items() if n.startswith("Lt") and m in n), None)
        r = next((d for n, d in chan_summary.items() if n.startswith("Rt") and m in n), None)
        if l and r and (l["norm_mean"] + r["norm_mean"]) > 0:
            summary[f"{m.replace(' ', '_')}_LR_symmetry_pct"] = round(
                100 * abs(l["norm_mean"] - r["norm_mean"]) / (0.5 * (l["norm_mean"] + r["norm_mean"])), 1)
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=lambda o: None if isinstance(o, float) and not np.isfinite(o) else str(o))
    plot_cycles(rec, cycles, os.path.join(out_dir, "cycles.png"),
                f"{label['patient']} {label['date']} {sf.name}: envelope / rest threshold, mean +/- sd over {len(used)} strides")
    return summary


def plot_cycles(rec, cycles, out_png, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    grid = np.linspace(0, 100, N_PTS)
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True)
    for ax, m in zip(axes.ravel(), MUSCLES):
        for side, color in (("Lt", "tab:blue"), ("Rt", "tab:orange")):
            ch = next((i for i, n in enumerate(rec.channels) if n and n.startswith(side) and m in n), None)
            if ch is None or not len(cycles.get(ch, [])):
                continue
            mu, sd = cycles[ch].mean(axis=0), cycles[ch].std(axis=0)
            ax.plot(grid, mu, color=color, label=f"{side} (n={len(cycles[ch])})")
            ax.fill_between(grid, mu - sd, mu + sd, color=color, alpha=0.2)
        ax.axhline(1.0, color="k", lw=0.6, ls=":")
        ax.set_title(m, fontsize=10)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7)
    for ax in axes[1]:
        ax.set_xlabel("% stride (tab to tab)")
    for ax in axes[:, 0]:
        ax.set_ylabel("envelope / rest threshold")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("patients", nargs="+")
    ap.add_argument("--dates", nargs="*", default=None, help="only sessions on these dates (YYYY-MM-DD)")
    ap.add_argument("--max-seconds", type=int, default=None, help="cap the read (default: whole file)")
    ap.add_argument("--out", default=os.path.join("outputs", "walking_emg"))
    args = ap.parse_args(argv)
    if not ep.ROOT:
        sys.exit("RESTORES_ROOT is not set and restores_config.py is missing")

    os.makedirs(args.out, exist_ok=True)
    out_csv = os.path.join(args.out, "walking_emg_summary.csv")
    rows = []

    def flush():                                   # a crash half-way still leaves a usable table
        pd.DataFrame(rows).to_csv(out_csv, index=False)

    for pat in args.patients:
        root = os.path.join(ep.ROOT, f"RESTORESP{pat.strip('Pp')}")
        for date, phase, folder in rc.sessions(root):
            if args.dates and str(date) not in args.dates:
                continue
            files = rc.list_session(folder)
            walks = [f for f in files if f.movement == "walking"]
            if not walks:
                continue
            print(f"--- {pat} {date} {phase}  {len(walks)} walking file(s)", flush=True)
            rests = {}
            for sf in walks:
                try:
                    b = rc.baseline_for(files, sf.stim_on) or rc.baseline_for(files, None) \
                        or next((x for x in files if x.is_baseline), None)
                    rest = None
                    if b is not None:
                        rest = rests.setdefault(b.path, rc.load(b.path, max_seconds=args.max_seconds or 60))
                    rec = rc.load(sf.path, max_seconds=args.max_seconds)
                    stem = os.path.splitext(sf.name)[0].replace(" ", "_").replace(",", "")
                    out_dir = os.path.join(args.out, f"{pat}_{date}_{stem}")
                    s = analyze_file(sf, rest, rec, out_dir, {"patient": pat, "date": str(date), "phase": phase})
                except Exception as e:                      # report, never silently skip
                    print(f"    !! {sf.name}: {type(e).__name__}: {e}", flush=True)
                    rows.append({"patient": pat, "date": str(date), "file": sf.name,
                                 "error": f"{type(e).__name__}: {e}"})
                    flush()
                    continue
                print(f"    {sf.name[:45]:45} stim={s['stim_on']} hz={s['stim_hz']} tabs={s['tabs_in_clip']}/{s['tabs_total']} "
                      f"gaits={s['gaits_named']} strides={s['strides_used']} "
                      f"stride={s['stride_time_median_s']}s cv={s['stride_time_cv']}", flush=True)
                rows.append(s)
                flush()
    flush()
    print(f"\nwrote {len(rows)} rows -> {out_csv}")


if __name__ == "__main__":
    main()
