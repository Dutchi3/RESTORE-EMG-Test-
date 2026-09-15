# -*- coding: utf-8 -*-
"""
Loader for the RESTORES CSV export format (P4-P6, and P3's CSV copies).

One file per exercise.  Layout, verified on RS-0103/0104/0105/0106:

    <preamble>   patient id, test start/end, then an "Exercise time tabs:"
                 block of wall-clock event markers (variable length)
    Mod No, Mod Name, Tr No, Tr Name, Date Time, LFF, HFF, Notch, 0, ..., 9999
    <rows>       one row per channel per second; channels cycle 1..N

Things this module takes care of because the raw tree does not:

  * the header row is found by pattern, not a fixed row count
  * channel names are read from each file -- P6 is an upper-limb montage
    and even changes between pre-op and post-op
  * LFF / HFF / Notch (the amplifier's own filters) are recorded per file
    and vary between and within sessions
  * baselines are identified by NAME; session numbering is not reliable
    (one P4 session has stim-on and stim-off baselines swapped)
  * an exercise's stim condition is not in its name: it is whatever the
    nearest preceding baseline (by number) was
  * exercise time tabs are mapped to sample indices from the first data
    row's timestamp and dropped when they fall outside the clip

Amplitudes are NOT gain-corrected.  Verified 2026-09-10: the CSV export is
identical to the un-adjusted 'Extracted' .mat tree, and no gain records
exist for P4-P6.  Any amplitude measure across sessions needs those
records first; scale-invariant measures do not.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

N_META_COLS = 8
FS = 10_000

_HEADER = re.compile(r"^\s*Mod No\s*,")
_TAB = re.compile(r"^\s*(\d+)\.\s*(\d{1,2}:\d{2}:\d{2})\s*$")
_NUMBER = re.compile(r"^\s*(\d+)")
_SIDE = re.compile(r"\b(lt|left|rt|right)\b", re.I)
_STIM = re.compile(r"stim\s*(on|off)", re.I)

# Lower-limb montage, in the order the acquisition writes it.  Used only to
# pick a default trigger channel; every loaded file carries its own names.
LOWER_LIMB = ['Lt Psoas M', 'Lt Rectus Femoris', 'Lt Vastus Laterali', 'Lt Tibialis Anterior',
              'Lt Gluteus Maximus', 'Lt Bicep Femoris', 'Lt Gastroc', 'Lt Soleus',
              'Rt Psoas M', 'Rt Rectus Femoris', 'Rt Vastus Laterali', 'Rt Tibialis Anterior',
              'Rt Gluteus Maximus', 'Rt Bicep Femoris', 'Rt Gastroc', 'Rt Soleus']

# Agonist channel offsets within one leg (add 8 for the right leg), from the
# study's "Muscles for exercises" document.
LOWER_LIMB_AGONISTS = {
    "hip flex": [0, 1, 4], "hip ext": [4, 5], "knee flex": [5, 6], "knee ext": [1, 2],
    "ankle dorsiflex": [3], "ankle plantarflex": [6, 7],
}


# --------------------------------------------------------------------------
# One file
# --------------------------------------------------------------------------
@dataclass
class Recording:
    path: str
    data: np.ndarray                 # (n_channels, n_samples), raw export units
    channels: list[str]
    fs: int
    date_time: list[str]             # per row, 1 s resolution
    tabs: list[str]                  # wall-clock event markers from the preamble
    lff: list[float]
    hff: list[float]
    notch: list[float]
    meta: dict = field(default_factory=dict)

    @property
    def duration_s(self):
        return self.data.shape[1] / self.fs

    @property
    def name(self):
        return os.path.basename(self.path)

    def tab_samples(self):
        """Event markers as sample indices, restricted to the clip.

        The first data row's timestamp is the start of this file's clip,
        which is not the session's 'Test start'.  Some tabs precede the clip
        or follow it; those are dropped.
        """
        t0 = _clock(self.date_time[0]) if self.date_time else None
        if t0 is None:
            return []
        out = []
        for tab in self.tabs:
            t = _clock(tab)
            if t is None:
                continue
            t = t.replace(year=t0.year, month=t0.month, day=t0.day)
            i = int(round((t - t0).total_seconds() * self.fs))
            if 0 <= i < self.data.shape[1]:
                out.append(i)
        return out


def parse_preamble(path):
    """-> (meta dict, line index of the column header)."""
    meta = {"tabs": []}
    with open(path, "r", errors="replace") as f:
        for i, line in enumerate(f):
            if _HEADER.match(line):
                return meta, i
            s = line.strip()
            if not s:
                continue
            m = _TAB.match(s)
            if m:
                meta["tabs"].append(m.group(2))
                continue
            for key, name in (("Patient ID:", "patient_id"), ("Test name:", "test_name"),
                              ("Test start:", "test_start"), ("Test end:", "test_end"),
                              ("Duration:", "duration")):
                if s.startswith(key):
                    meta[name] = s.split(":", 1)[1].strip().strip('"').strip()
    raise ValueError(f"no 'Mod No' header row in {path}")


def load(path, max_seconds=None, fs=FS):
    """Load one CSV export.  max_seconds caps the read; files run to 100s of MB."""
    meta, header_idx = parse_preamble(path)
    probe = pd.read_csv(path, skiprows=header_idx, header=0, nrows=64, usecols=range(N_META_COLS))
    tr_no = pd.to_numeric(probe.iloc[:, 2], errors="coerce").dropna().astype(int)
    n_ch = int(tr_no.max()) if len(tr_no) else 0
    if n_ch == 0:
        raise ValueError(f"no channel rows in {path}")

    nrows = None if max_seconds is None else int(max_seconds) * n_ch
    df = pd.read_csv(path, skiprows=header_idx, header=0, nrows=nrows,
                     skip_blank_lines=True, engine="c", low_memory=False).dropna(how="all")

    tr_no = pd.to_numeric(df.iloc[:, 2], errors="coerce").to_numpy()
    tr_name = df.iloc[:, 3].astype(str).str.strip().str.strip('"').str.strip()
    names = [None] * n_ch
    for k, nm in zip(tr_no, tr_name):
        if np.isfinite(k) and names[int(k) - 1] is None:
            names[int(k) - 1] = nm
    samples = df.iloc[:, N_META_COLS:].to_numpy(dtype=float)
    spr = samples.shape[1]

    # A new one-second cycle starts whenever the channel number stops
    # increasing.  Placing rows by their own channel number (rather than by
    # position) survives a missing row or a row whose channel number failed
    # to parse -- both occur -- instead of shifting every later second.
    good = np.isfinite(tr_no)
    sec_of_row = np.zeros(len(df), dtype=int)
    sec, prev = -1, np.inf
    for i in range(len(df)):
        if not good[i]:
            continue
        if tr_no[i] <= prev:
            sec += 1
        sec_of_row[i], prev = sec, tr_no[i]
    n_sec = sec + 1
    data = np.full((n_ch, max(n_sec, 0) * spr), np.nan)
    for i in np.flatnonzero(good):
        ch = int(tr_no[i]) - 1
        s = sec_of_row[i]
        data[ch, s * spr:(s + 1) * spr] = samples[i]

    def distinct(col):
        return sorted(set(pd.to_numeric(df.iloc[:, col], errors="coerce").dropna().tolist()))

    meta.update(n_rows=len(df), dropped_rows=int((~good).sum()),
                incomplete_seconds=int(n_sec - np.sum(np.bincount(sec_of_row[good], minlength=n_sec) == n_ch))
                if n_sec else 0,
                samples_per_row=spr, nan_fraction=float(np.isnan(data).mean()) if data.size else 0.0)
    return Recording(path, data, names, fs,
                     df.iloc[:, 4].astype(str).str.strip().str.strip('"').str.strip().tolist(),
                     meta.pop("tabs"), distinct(5), distinct(6), distinct(7), meta)


def _clock(s):
    s = str(s).strip().strip('"').strip()
    for fmt in ("%m/%d/%Y %H:%M:%S", "%d/%m/%Y %H:%M:%S", "%H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


# --------------------------------------------------------------------------
# One session folder
# --------------------------------------------------------------------------
@dataclass
class SessionFile:
    path: str
    number: int | None
    is_baseline: bool
    stim_on: bool | None          # None = unknown / pre-op
    side: str | None              # 'L', 'R' or None
    movement: str                 # e.g. 'hip flex', 'elbow ext', 'walking'
    redo: bool
    gaits: int | None = None      # walking files: the therapist's count from the name
    harness: bool = False         # walking files: 'with harness' in the name
    brace: bool = False           # walking files: 'with leg brace' in the name

    @property
    def name(self):
        return os.path.basename(self.path)


_GAITS = re.compile(r"(\d+)\s*full\s*g[ai]{2}ts?", re.I)      # 'gaits', 'gait', and the 'giats' typo


def _classify(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    m = _NUMBER.match(stem)
    number = int(m.group(1)) if m else None
    body = re.sub(r"^\s*\d+\s*[.,]?\s*", "", stem).strip()
    low = body.lower()
    st = _STIM.search(low)
    stim = (st.group(1) == "on") if st else None
    is_base = "baseline" in low
    sm = _SIDE.search(low)
    side = None if not sm else ("L" if sm.group(1).lower() in ("lt", "left") else "R")
    movement = _SIDE.sub("", low)
    movement = re.sub(r"stim\s*(on|off)|baseline|redo|\(.*?\)|@.*$", "", movement)
    movement = re.sub(r"\s+", " ", movement).strip(" -.")
    gaits, harness, brace = None, False, False
    if "walk" in low:
        # 'Walking 22 full gaits', 'walking with harness, 16 full gaits', '..., 23 steps, 12 full gaits',
        # 'walking with leg brace 16 full gaits', 'Attempting 1st walk'
        g = _GAITS.search(low)
        gaits = int(g.group(1)) if g else None
        harness, brace = "harness" in low, "brace" in low
        movement, side = "walking", None
    return SessionFile(path, number, is_base, stim, side, movement, "redo" in low, gaits, harness, brace)


def list_session(folder):
    """Every CSV in a session folder, with its stim condition resolved.

    Baselines carry 'stim on'/'stim off' in their name.  Exercises do not,
    so each inherits the condition of the nearest preceding baseline by
    file number.  This is the only rule that survives the swapped session
    (P4 16 Feb 2026, where 0 is stim-on and 9 is stim-off).
    """
    files = [_classify(os.path.join(folder, f)) for f in os.listdir(folder)
             if f.lower().endswith(".csv")]
    files.sort(key=lambda r: (r.number if r.number is not None else 10 ** 6, r.name))
    baselines = [(r.number, r.stim_on) for r in files if r.is_baseline and r.number is not None]
    for r in files:
        if r.is_baseline or r.number is None:
            continue
        prior = [(n, s) for n, s in baselines if n < r.number]
        if prior:
            r.stim_on = max(prior)[1]
        elif baselines:
            r.stim_on = min(baselines)[1]
        else:
            r.stim_on = None
    return files


def baseline_for(files, stim_on):
    """The rest recording in the requested condition, or None."""
    for r in files:
        if r.is_baseline and (r.stim_on == stim_on or (stim_on is None and r.stim_on is None)):
            return r
    if stim_on is None:                                    # pre-op: any baseline will do
        return next((r for r in files if r.is_baseline), None)
    return None


WALKING_TRIGGERS = [3, 11]      # Lt / Rt Tibialis Anterior: one swing-phase burst per stride


def agonists(rec: Recording, sf: SessionFile):
    """Channel indices for this exercise's agonists, or [] if unknown montage."""
    if [c or "" for c in rec.channels] != LOWER_LIMB:
        return []
    if sf.movement == "walking":
        return list(WALKING_TRIGGERS)
    key = next((k for k in LOWER_LIMB_AGONISTS if k in sf.movement), None)
    if key is None or sf.side is None:
        return []
    return [i + (8 if sf.side == "R" else 0) for i in LOWER_LIMB_AGONISTS[key]]


# --------------------------------------------------------------------------
# Patient trees
# --------------------------------------------------------------------------
_DATE = re.compile(r"(\d{1,2})\s+([A-Za-z]{3,9})\s+(\d{4})")


def session_date(folder_name):
    m = _DATE.search(folder_name)
    if not m:
        return None
    d, mon, y = m.groups()
    for fmt in ("%d %B %Y", "%d %b %Y"):
        try:
            return datetime.strptime(f"{d} {mon[:3] if fmt.endswith('b %Y') else mon} {y}", fmt).date()
        except ValueError:
            continue
    return None


def sessions(patient_root):
    """-> [(date, phase, folder)] for every session under a RESTORESP{n} tree,
    sorted by date.  phase is 'pre' or 'post'.  Numbers in folder names are
    ignored: two copies of P4 disagree on them, and one skips a session."""
    out = []
    for top in os.listdir(patient_root):
        tpath = os.path.join(patient_root, top)
        if not os.path.isdir(tpath):
            continue
        phase = "pre" if re.search(r"pre.?op", top, re.I) else ("post" if re.search(r"post.?op", top, re.I) else None)
        if phase is None:
            continue
        for s in os.listdir(tpath):
            spath = os.path.join(tpath, s)
            if os.path.isdir(spath) and any(f.lower().endswith(".csv") for f in os.listdir(spath)):
                out.append((session_date(s), phase, spath))
    return sorted(out, key=lambda t: (t[0] is None, t[0]))
