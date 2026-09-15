# -*- coding: utf-8 -*-
"""
QC pass over RESTORES CSV sessions (P4-P6): one row per exercise recording.

    python run_csv_qc.py P4 [P5 P6] [--phase post|pre|all] [--max-seconds 60]
                         [--sessions N] [--out outputs/csv_qc.csv]

Per recording it reports what the pipeline decided and why, so a bad
decision is visible before any statistic is built on it:

    which baseline was used and whether it matched the stim condition
    the stimulation pulse rate measured on the trial and on the rest
    the activation threshold and the fraction of the clip marked active
    whether the exercise time tabs coincide with detected activity
    the amplifier's own LFF / HFF / Notch settings for the file
    RMS of the active samples on each agonist (un-gain-corrected)

Amplitudes are NOT comparable across sessions until the amplifier range
settings are recovered; see restores_csv.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

import emg_pipeline as ep
import restores_csv as rc


def consensus_channels(n_ch):
    """All channels: on P6's arm montage the artifact is invisible on the
    four leg-chosen channels while stim-on RMS is 3-12x stim-off."""
    return list(range(n_ch))


def qc_session(patient, date, phase, folder, max_seconds):
    files = rc.list_session(folder)
    rests = {}

    def rest_for(stim_on):
        b = rc.baseline_for(files, stim_on)
        if b is None:
            return None, None
        if b.path not in rests:
            rests[b.path] = rc.load(b.path, max_seconds=max_seconds)
        return b, rests[b.path]

    rows = []
    for sf in files:
        if sf.is_baseline:
            continue
        warn = []
        rec = rc.load(sf.path, max_seconds=max_seconds)
        base, rest = rest_for(sf.stim_on)
        if base is None:
            base, rest = rest_for(None)
            if base is None:
                base, rest = next(((b, rests.setdefault(b.path, rc.load(b.path, max_seconds=max_seconds)))
                                   for b in files if b.is_baseline), (None, None))
            warn.append("baseline condition mismatch" if base is not None else "no baseline")
        if rest is not None and rest.channels != rec.channels:
            warn.append("baseline montage differs")

        lower = [c or "" for c in rec.channels] == rc.LOWER_LIMB
        ag = rc.agonists(rec, sf)
        if not lower:
            warn.append("montage not lower-limb")
        elif not ag:
            warn.append("no agonist map for movement")
        trigger = ag[0] if ag else 0
        electrodes = ag or [trigger]

        cc = consensus_channels(len(rec.channels))
        if phase == "pre":                      # no stimulator yet: anything periodic is environmental
            rate_t, agree_t, method_t = None, 0, None
            rate_r, agree_r, method_r = None, 0, None
        else:
            rate_t, agree_t, method_t, _ = ep.pulse_train_consensus(rec.data, rec.fs, cc)
            rate_r, agree_r, method_r, _ = (ep.pulse_train_consensus(rest.data, rest.fs, cc)
                                            if rest is not None else (None, 0, None, {}))
        env_t, env_r = ep.is_environmental(rate_t, method_t), ep.is_environmental(rate_r, method_r)
        if env_t:
            rate_t, method_t = None, None
        if env_r:
            rate_r, method_r = None, None
        stim_hz, method = (rate_t, method_t) if rate_t is not None else (rate_r, method_r)
        if sf.stim_on and stim_hz is None:
            warn.append("stim on: no stimulation detected")
        if sf.stim_on is False and rate_t is not None:
            warn.append(f"stim off but {rate_t} Hz on trial")
        if rate_t is not None and rate_r is not None:
            ratio = rate_t / rate_r
            # 2x apart is the same train read with alternate pulses dropped, not a settings change
            if min(abs(ratio - k) for k in (1.0, 2.0, 0.5)) > 0.1:
                warn.append(f"baseline stim {rate_r} Hz != trial {rate_t} Hz")
        if env_t or env_r:
            warn.append("17 Hz environmental comb present")

        row = {
            "patient": patient, "date": date, "phase": phase, "file": sf.name, "number": sf.number,
            "stim_on": sf.stim_on, "redo": sf.redo, "side": sf.side, "movement": sf.movement,
            "gaits": sf.gaits, "harness": sf.harness, "brace": sf.brace,
            "montage": "lower_limb" if lower else "other", "n_ch": len(rec.channels),
            "duration_s": round(rec.duration_s, 1),
            "LFF": rec.lff, "HFF": rec.hff, "Notch": rec.notch,
            "dropped_rows": rec.meta.get("dropped_rows"), "nan_frac": round(rec.meta.get("nan_fraction", 0), 4),
            "tabs_total": len(rec.tabs), "tabs_in_clip": len(rec.tab_samples()),
            "baseline": None if base is None else base.name,
            "rate_trial": rate_t, "agree_trial": agree_t, "rate_rest": rate_r,
            "stim_hz_used": stim_hz, "stim_method": method, "env_comb": bool(env_t or env_r),
        }
        if rest is None:
            row.update(threshold=None, active_frac=None, tab_hit_frac=None, rms_active="", rms_rest_trigger=None,
                       warnings="; ".join(warn))
            rows.append(row)
            continue

        blank = method == "pulses"
        trials = ep.process_arrays(rec.data, rest.data, trigger, electrodes, fs=rec.fs,
                                   remove_stim=True, stim_hz=stim_hz,
                                   label=(patient, str(date), sf.number, sf.stim_on), paths=[sf.path])
        prep_t = ep.remove_artifact(rec.data[trigger], rec.fs, stim_hz=stim_hz, blank=blank)
        prep_r = ep.remove_artifact(rest.data[trigger], rest.fs, stim_hz=stim_hz, blank=blank)
        mask, thr = ep.detect_activation(prep_t, prep_r, fs=rec.fs)

        tabs = rec.tab_samples()
        hits = [bool(mask[i:i + 3 * rec.fs].any()) for i in tabs]
        if len(rec.tabs) and not tabs:
            warn.append("all tabs outside clip")
        if rec.meta.get("dropped_rows"):
            warn.append("dropped rows")
        if mask.mean() > 0.95:
            warn.append("whole clip active")

        row.update(
            threshold=round(thr, 4), active_frac=round(float(mask.mean()), 3),
            tab_hit_frac=(round(float(np.mean(hits)), 2) if hits else None),
            rms_active="; ".join(f"{rec.channels[t.electrode]}={t.rms():.3f}" for t in trials),
            rms_rest_trigger=round(float(np.sqrt(np.mean(prep_r ** 2))), 4),
            warnings="; ".join(warn),
        )
        rows.append(row)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("patients", nargs="+", help="P4 P5 P6")
    ap.add_argument("--phase", default="post", choices=["post", "pre", "all"])
    ap.add_argument("--max-seconds", type=int, default=60)
    ap.add_argument("--sessions", type=int, default=None, help="only the first N sessions per patient")
    ap.add_argument("--dates", nargs="*", default=None, help="only sessions on these dates (YYYY-MM-DD)")
    ap.add_argument("--out", default=os.path.join("outputs", "csv_qc.csv"))
    args = ap.parse_args(argv)

    if not ep.ROOT:
        sys.exit("RESTORES_ROOT is not set and restores_config.py is missing")

    rows = []
    for pat in args.patients:
        root = os.path.join(ep.ROOT, f"RESTORESP{pat.strip('Pp')}")
        sess = [s for s in rc.sessions(root) if args.phase == "all" or s[1] == args.phase]
        if args.dates:
            sess = [s for s in sess if str(s[0]) in args.dates]
        if args.sessions:
            sess = sess[:args.sessions]
        for date, phase, folder in sess:
            print(f"--- {pat} {date} {phase}  {os.path.basename(folder)}", flush=True)
            try:
                got = qc_session(pat, date, phase, folder, args.max_seconds)
            except Exception as e:                       # finding 21: never silently skip
                print(f"    !! {type(e).__name__}: {e}", flush=True)
                rows.append({"patient": pat, "date": date, "phase": phase, "file": None,
                             "warnings": f"session failed: {type(e).__name__}: {e}"})
                continue
            for r in got:
                print(f"    {r['file'][:40]:40} stim={str(r['stim_on']):5} hz={str(r['stim_hz_used']):6} "
                      f"act={r['active_frac']!s:6} tabhit={r['tab_hit_frac']!s:5} {r['warnings']}", flush=True)
            rows.extend(got)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"\nwrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
