# -*- coding: utf-8 -*-
"""
Single-path EMG processing for the RESTORES study (P1-P3).

Replaces the six near-identical per-patient blocks that each analysis script
carried.  Patient differences (exercise numbering, trials-per-exercise, file
naming) are data here, not duplicated code.

Corrections against the original pipeline, with the audit finding they close:

  01  stimulation artifact is now actually removed (apply_filters_P3 shipped
      with its notch/baseline/rectify block commented out)
  02  amplifier gain is applied (get_gains was imported but never called)
  03  segmentation triggers on each exercise's own agonist, not always psoas
  04  activation threshold comes from the rest recording, not from the trial
      it is thresholding
  06  right ankle plantarflexion uses electrodes 14,15 not 13,14
  11  sample rate is honoured instead of hardcoded to 10 kHz
  12  notch series start at the first real harmonic, not 0 Hz
  17  peak normalisation uses a percentile, not a single sample
  20  every trial is kept instead of the last one overwriting the rest
  21  failures are recorded and reported instead of silently swallowed
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import numpy as np
import scipy.io
from scipy import signal

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
# Where the study data lives.  Set RESTORES_ROOT to the share root, e.g.
#     setx RESTORES_ROOT "\\<server>\<share>"          (Windows, persistent)
#     export RESTORES_ROOT='/mnt/<share>'              (POSIX)
# or drop a one-line restores_config.py next to this file containing
#     ROOT = r"..."
# Kept out of source control so the internal path never enters the repository.
def _resolve_root():
    env = os.environ.get("RESTORES_ROOT")
    if env:
        return env
    try:
        from restores_config import ROOT       # untracked, see .gitignore
        return ROOT
    except Exception:
        return ""


ROOT = _resolve_root()
DATA_ROOT = os.path.join(ROOT, "David_Teo_RESTORES", "handover docs", "data files")
GAIN_ROOT = os.path.join(DATA_ROOT, "Gain and Impedance Data")

# Two parallel copies of the recordings exist.
#
#   Extracted Mat Files      - samples still carry the amplifier range scaling
#   Gain Adjusted Mat Files  - the same recordings divided by that range
#
# Every original analysis script reads the *Extracted* tree, so its samples are
# inflated by the range setting, which was raised during stimulation and varies
# by muscle and by week.  Measured on P1 W10: stim-off is /50 on every channel,
# stim-on is /250 on left psoas, /300 on right psoas, /100 on left gastrocnemius
# and /50 elsewhere.  P3 W1 spans /30 stim-off to /700 stim-on.  An on-vs-off
# amplitude comparison built on the Extracted tree therefore carries a scale
# factor of up to 6x (P1) or 23x (P3) that has nothing to do with the muscle.
#
# Coherence is scale-invariant and is unaffected by this choice.  Any amplitude
# measure is not.
GAINADJ_ROOT = os.path.join(ROOT, "Gain Adjusted Mat Files")
EXTRACTED_ROOT = os.path.join(DATA_ROOT, "Extracted Mat Files")

USE_GAIN_ADJUSTED = True
MAT_ROOT = GAINADJ_ROOT if USE_GAIN_ADJUSTED else EXTRACTED_ROOT

# The gain-adjusted tree is not quite complete; these weeks exist only in the
# Extracted tree and have to be either skipped or gain-corrected by hand.
GAINADJ_MISSING = {"P1": {3}, "P2": {13}, "P3": {14, 26}}

FS = 10_000                 # acquisition rate, Hz
BAND = (30.0, 450.0)        # EMG passband
STIM_HZ = 40.0              # stimulator fundamental, confirmed from the recordings
MAINS_HZ = 50.0
NOTCH_Q = 30.0

ELECTRODES = ['L Psoas Major', 'L Rectus Femoris', 'L Vastus Lateralis', 'L Tibialis Anterior',
              'L Gluteus Maximus', 'L Bicep Femoris', 'L Gastrocnemius', 'L Soleus',
              'R Psoas Major', 'R Rectus Femoris', 'R Vastus Lateralis', 'R Tibialis Anterior',
              'R Gluteus Maximus', 'R Bicep Femoris', 'R Gastrocnemius', 'R Soleus']

EXERCISES = {2: 'Left Hip Flexion', 3: 'Left Hip Extension',
             4: 'Left Knee Flexion', 5: 'Left Knee Extension',
             6: 'Left Ankle Dorsiflexion', 7: 'Left Ankle Plantarflexion',
             8: 'Right Hip Flexion', 9: 'Right Hip Extension',
             10: 'Right Knee Flexion', 11: 'Right Knee Extension',
             12: 'Right Ankle Dorsiflexion', 13: 'Right Ankle Plantarflexion'}

# Agonists per exercise, from the study's own "Muscles for exercises" document.
# The original pipeline used electrode 0 or 8 (psoas) for every exercise; psoas
# is the correct agonist for hip flexion only.
AGONISTS = {2: [0, 1, 4], 3: [4, 5], 4: [5, 6], 5: [1, 2], 6: [3], 7: [6, 7],
            8: [8, 9, 12], 9: [12, 13], 10: [13, 14], 11: [9, 10], 12: [11], 13: [14, 15]}

STIM_ON_OFFSET = 13         # stim-on exercise number = stim-off number + 13


@dataclass(frozen=True)
class Patient:
    """Per-patient recording conventions."""
    name: str
    exercise_shift: int      # added to the canonical exercise number
    weeks: range             # weeks with stimulation delivered
    gain_folder: str

    def exercise_number(self, canonical: int, stim_on: bool) -> int:
        return canonical + self.exercise_shift + (STIM_ON_OFFSET if stim_on else 0)


PATIENTS = {
    "P1": Patient("P1", 0, range(7, 27), "RS-0101 Post op gain impedance"),
    "P2": Patient("P2", 0, range(9, 28), "RS-0102 Post op gain impedance"),
    "P3": Patient("P3", -1, range(1, 26), "RS-0103_Post Op gain impedance"),
}


# --------------------------------------------------------------------------
# Filtering
# --------------------------------------------------------------------------
def _notch_sections(freqs, q, fs):
    rows = []
    for f0 in freqs:
        if 0 < f0 < fs / 2:                      # finding 12: 0 Hz is degenerate
            b, a = signal.iirnotch(f0, q, fs)
            rows.append(np.concatenate([b, a]))
    return np.array(rows).reshape(-1, 6)


def detect_stim_frequency(x, fs=FS, candidates=(20, 25, 30, 40, 47.5, 60, 80, 100, 120),
                          exclude_mains=True, min_prominence_db=6.0):
    """Find the stimulation line in a recording, or None if there isn't one.

    STIM_HZ is 40 and that is correct wherever the artifact is strong, but the
    artifact is not always present: surveying stim-on recordings, the dominant
    20-100 Hz line is 40 Hz in P1 W7/W10 and P3 W1, and merely mains (50 or
    100 Hz) in P1 W18, P2 W9/W20 and P3 W3.  A blanket 40 Hz notch therefore
    carves a hole in recordings that never had an artifact.  Detect instead.

    17 Hz - the series the original notched 30 times - is not present at all:
    measured prominence is +1.6 dB stim-off and +0.6 dB stim-on, i.e. noise.
    """
    f, p = signal.welch(np.nan_to_num(x), fs=fs, nperseg=min(20000, len(x)))
    best, best_prom = None, 0.0
    for f0 in candidates:
        if exclude_mains and abs(f0 % MAINS_HZ) < 1e-6:
            continue
        i = int(np.argmin(np.abs(f - f0)))
        side = np.r_[p[max(0, i - 60):max(0, i - 12)], p[i + 12:i + 60]]
        if side.size == 0 or p[i] <= 0:
            continue
        prom = 10 * np.log10(p[i] / np.median(side))
        if prom > best_prom:
            best, best_prom = f0, prom
    return (best, best_prom) if best_prom >= min_prominence_db else (None, best_prom)


def build_filter(fs=FS, band=BAND, stim_hz=STIM_HZ, mains_hz=MAINS_HZ, q=NOTCH_Q):
    """Bandpass plus stimulation and mains harmonics, as one second-order-section
    cascade.  Applied with sosfiltfilt it is zero-phase; the original applied
    52 notches through lfilter, which is causal and distorts phase."""
    sos = [signal.butter(4, band, btype='band', fs=fs, output='sos')]
    if stim_hz:
        sos.append(_notch_sections(np.arange(stim_hz, fs / 2, stim_hz), q, fs))
    if mains_hz:
        sos.append(_notch_sections(np.arange(mains_hz, fs / 2, mains_hz), q, fs))
    return np.vstack(sos)


_FILTER_CACHE: dict = {}


def clean(x, fs=FS, remove_stim=True, remove_mains=True):
    """Filter one channel.  remove_stim=False reproduces the original behaviour."""
    key = (fs, remove_stim and STIM_HZ or None, remove_mains and MAINS_HZ or None)
    if key not in _FILTER_CACHE:
        _FILTER_CACHE[key] = build_filter(
            fs=fs,
            stim_hz=STIM_HZ if remove_stim else None,
            mains_hz=MAINS_HZ if remove_mains else None,
        )
    return signal.sosfiltfilt(_FILTER_CACHE[key], np.nan_to_num(np.asarray(x, dtype=float)))


def rms_envelope(x, fs=FS, window_s=0.05, overlap=0.5):
    """Windowed RMS.  window_s=0.05 matches the original's 500-sample window
    (its comment said 0.01 s, which was wrong - finding 10)."""
    w = max(1, int(round(window_s * fs)))
    step = max(1, int(round(w * (1 - overlap))))
    if len(x) < w:
        return np.array([np.sqrt(np.mean(np.square(x)))]) if len(x) else np.array([0.0])
    idx = range(0, len(x) - w + 1, step)
    return np.array([np.sqrt(np.mean(np.square(x[i:i + w]))) for i in idx])


def robust_peak(x, percentile=99.0):
    """Finding 17: max() is a single sample and one artifact spike rescales a
    whole trial."""
    return float(np.percentile(np.abs(x), percentile))


# --------------------------------------------------------------------------
# Activation detection
# --------------------------------------------------------------------------
def detect_activation(trial, rest, fs=FS, k=3.0, smooth_hz=0.7, pad_s=0.5, min_dur_s=0.2):
    """Samples where the muscle is active.

    The threshold is derived from the *rest* recording (mean + k*sd of its
    envelope).  The original used 1.2 * mean of the trial's own envelope, which
    is circular: a weak trial lowers its own bar, so noise is admitted as
    movement exactly in the sessions where the patient is weakest (finding 04).
    """
    b, a = signal.butter(2, smooth_hz, btype='low', fs=fs)
    env_trial = signal.filtfilt(b, a, np.abs(trial))
    env_rest = signal.filtfilt(b, a, np.abs(rest))

    threshold = float(np.mean(env_rest) + k * np.std(env_rest))
    active = env_trial > threshold

    pad, min_len = int(pad_s * fs), int(min_dur_s * fs)
    edges = np.diff(active.astype(np.int8))
    starts = list(np.flatnonzero(edges == 1) + 1)
    stops = list(np.flatnonzero(edges == -1) + 1)
    if active[0]:
        starts.insert(0, 0)
    if active[-1]:
        stops.append(len(active))

    mask = np.zeros(len(trial), dtype=bool)
    for s, e in zip(starts, stops):
        if e - s >= min_len:
            mask[max(0, s - pad):min(len(trial), e + pad)] = True
    return mask, threshold


# --------------------------------------------------------------------------
# File discovery and loading
# --------------------------------------------------------------------------
_NUM = re.compile(r'^\s*(\d+)\s*(?:\.\s*(\d+))?')


def parse_name(filename):
    """-> (exercise_number, subtrial or None), or (None, None) if unnumbered."""
    m = _NUM.match(filename)
    if not m:
        return None, None
    return int(m.group(1)), (int(m.group(2)) if m.group(2) else None)


def find_trials(folder, exercise_number, max_subtrial=3):
    """Every file for one exercise, in subtrial order.

    Subtrial 4 is the 'controlled release' variant the original excluded.
    Unlike the original this keeps *all* remaining trials rather than letting
    each overwrite the last (finding 20).
    """
    hits = []
    for fn in os.listdir(folder):
        if not fn.lower().endswith('.mat'):
            continue
        num, sub = parse_name(fn)
        if num != exercise_number:
            continue
        if sub is not None and sub > max_subtrial:
            continue
        hits.append((sub if sub is not None else 0, os.path.join(folder, fn)))
    return [p for _, p in sorted(hits)]


def load_recording(path):
    """Load a .mat once and return the full (16, N) array.

    The original called loadmat separately for the trigger channel and for each
    analysed electrode, re-reading the same 20 MB file two or three times.
    """
    return np.asarray(scipy.io.loadmat(path)['data'], dtype=float)


def load_concatenated(paths):
    """Concatenate the trials of one exercise into a single (16, N) array."""
    parts = [load_recording(p) for p in paths]
    n_ch = min(p.shape[0] for p in parts)
    return np.concatenate([p[:n_ch] for p in parts], axis=1)


BASELINE_EXERCISE = 1       # canonical number of the rest recording


def find_baseline(folder, patient, stim_on):
    """The rest recording *recorded in the same stimulation condition*.

    This matters more than it looks.  With stimulation on, the signal never
    settles back to the stim-off rest level, so thresholding a stim-on trial
    against a stim-off baseline marks the entire recording as active - the
    on/off comparison then contrasts movement-only epochs against whole
    recordings.  Both baselines were recorded (exercise 14 for P1/P2, 13 for
    P3); the original loaded the stim-on one into `baseline_on_signal` and then
    thresholded against the stim-off one regardless.
    """
    pat = PATIENTS[patient]
    wanted = pat.exercise_number(BASELINE_EXERCISE, stim_on)
    fallback, fallback_num = None, None
    for fn in os.listdir(folder):
        if not fn.lower().endswith('.mat'):
            continue
        num, _ = parse_name(fn)
        if num is None:
            continue
        if num == wanted:
            return os.path.join(folder, fn)
        if fallback_num is None or num < fallback_num:
            fallback, fallback_num = os.path.join(folder, fn), num
    return fallback


# --------------------------------------------------------------------------
# Gain
# --------------------------------------------------------------------------
def gain_scale(gain_uv, reference_uv=None):
    """Divide out the amplifier range so amplitudes are comparable across
    conditions and weeks.

    Prefer reading the Gain Adjusted tree (USE_GAIN_ADJUSTED = True), which
    already has this applied and was produced by the study itself.  This
    function exists for the handful of weeks missing from that tree.

    A note on how this was established, because an earlier version of this
    module got it wrong.  Comparing the two trees file by file shows
    gain_adjusted = extracted / range, with `range` taken per muscle and per
    stimulation condition, exactly matching the session gain notes.  An earlier
    attempt to settle the question from the quantisation step of the stored
    samples was invalid: the export rounds to 0.01 uV *after* the range scaling,
    so both trees show a 0.01 step whatever the range, and the measurement can
    never distinguish the two cases.
    """
    if gain_uv is None or not np.isfinite(gain_uv) or gain_uv == 0:
        return 1.0
    return 1.0 / float(gain_uv)


# --------------------------------------------------------------------------
# Trial-level processing
# --------------------------------------------------------------------------
@dataclass
class Trial:
    patient: str
    week: int
    exercise: int
    stim_on: bool
    electrode: int
    signal: np.ndarray            # cleaned, gain-corrected, active samples only
    n_active: int
    n_total: int
    threshold: float
    gain_uv: float | None
    paths: list = field(default_factory=list)

    @property
    def active_fraction(self):
        return self.n_active / self.n_total if self.n_total else 0.0

    def rms(self):
        return float(np.sqrt(np.mean(np.square(self.signal)))) if self.signal.size else np.nan


def process_trial(patient, week, canonical_exercise, stim_on, electrodes,
                  remove_stim=True, gains=None, fs=FS):
    """Load one exercise for one session and return a Trial per electrode.

    Segmentation triggers on the exercise's own agonist (finding 03) using a
    threshold derived from the session's rest recording (finding 04).
    """
    pat = PATIENTS[patient]
    folder = os.path.join(MAT_ROOT, patient, f"W{week}")
    number = pat.exercise_number(canonical_exercise, stim_on)

    paths = find_trials(folder, number)
    if not paths:
        raise FileNotFoundError(f"{patient} W{week}: no files for exercise {number}")
    baseline_path = find_baseline(folder, patient, stim_on)
    if baseline_path is None:
        raise FileNotFoundError(f"{patient} W{week}: no baseline recording")

    data = load_concatenated(paths)
    rest = load_recording(baseline_path)

    trigger_ch = AGONISTS[canonical_exercise][0]
    mask, threshold = detect_activation(
        clean(data[trigger_ch], fs, remove_stim=remove_stim),
        clean(rest[trigger_ch], fs, remove_stim=remove_stim),   # same condition, same filtering
        fs=fs)

    out = []
    for ch in electrodes:
        g = None if gains is None else gains.get(ch)
        x = clean(data[ch], fs, remove_stim=remove_stim) * gain_scale(g)
        out.append(Trial(patient, week, number, stim_on, ch, x[mask],
                         int(mask.sum()), int(mask.size), threshold, g, paths))
    return out


# --------------------------------------------------------------------------
# Quality control
# --------------------------------------------------------------------------
def artifact_prominence(x, fs=FS, f0=STIM_HZ):
    """Height of the stimulation line above local spectral background, in dB."""
    f, p = signal.welch(x, fs=fs, nperseg=min(20000, len(x)))
    i = int(np.argmin(np.abs(f - f0)))
    lo, hi = max(0, i - 60), min(len(p), i + 60)
    side = np.r_[p[lo:max(lo, i - 12)], p[min(hi, i + 12):hi]]
    if side.size == 0 or p[i] <= 0:
        return np.nan
    return float(10 * np.log10(p[i] / np.median(side)))


def qc_row(trial, raw_channel, fs=FS):
    """Per-trial quality metrics.  The original emitted nothing like this, so a
    26x amplitude jump or a session with no detected movement was invisible
    (finding 12 in the Tier 3 list)."""
    return {
        "patient": trial.patient, "week": trial.week, "exercise": trial.exercise,
        "stim_on": trial.stim_on, "electrode": ELECTRODES[trial.electrode],
        "n_trials": len(trial.paths),
        "duration_s": round(trial.n_total / fs, 1),
        "active_fraction": round(trial.active_fraction, 3),
        "rms": round(trial.rms(), 4) if np.isfinite(trial.rms()) else None,
        "threshold": round(trial.threshold, 4),
        "artifact_dB": round(artifact_prominence(raw_channel, fs), 1),
        "gain_uv": trial.gain_uv,
    }


def check_duplicate_recordings(folder, patient, canonical_exercise):
    """True when a session's stim-on recording is a byte-copy of its stim-off one.

    Twenty such duplicates exist in P3 W1 and W3; the pipeline compared those
    recordings against themselves, yielding a difference of exactly zero, which
    scipy's Wilcoxon then discarded - shrinking n with no warning.
    """
    pat = PATIENTS[patient]
    off = find_trials(folder, pat.exercise_number(canonical_exercise, False))
    on = find_trials(folder, pat.exercise_number(canonical_exercise, True))
    dups = []
    for a, b in zip(off, on):
        if os.path.getsize(a) == os.path.getsize(b):
            if np.array_equal(load_recording(a), load_recording(b)):
                dups.append((os.path.basename(a), os.path.basename(b)))
    return dups


if __name__ == "__main__":
    # Self-test against a session known to carry a large stimulation artifact.
    for stim_on in (False, True):
        trials = process_trial("P1", 10, 2, stim_on, electrodes=[0, 1])
        raw = load_concatenated(find_trials(
            os.path.join(MAT_ROOT, "P1", "W10"),
            PATIENTS["P1"].exercise_number(2, stim_on)))
        tag = "STIM ON " if stim_on else "STIM OFF"
        for t in trials:
            row = qc_row(t, raw[t.electrode])
            print(f"{tag} {row['electrode']:<20} rms={row['rms']:>10} "
                  f"active={row['active_fraction']:.2f} artifact={row['artifact_dB']:>6} dB")
