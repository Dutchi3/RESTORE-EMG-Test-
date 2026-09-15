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

Second pass (2026-09-10), after running the above against P4-P6 CSV data:

  A   the stimulation rate is measured per recording from pulse spacing
      (detect_pulse_train) and passed into clean(); the earlier spectral
      detect_stim_frequency() existed but was never wired in, so every
      recording was notched at 40 Hz whatever its true rate (20 Hz in P4,
      ~50 Hz in P1 W18 -- which spectral detection cannot see past mains)
  B   pulses are blanked in the time domain before filtering; a notch comb
      only removes the line components of a spike train, and at 20 Hz the
      21 in-band notches remove more EMG than artifact
  C   activation threshold is a percentile of the rest envelope, not
      mean + 3 sd of a 0.7 Hz-smoothed one, which sat below the trial's own
      rest floor and marked whole recordings active
  D   0.5 s at each end of a clip is never counted as active (filtfilt
      edge transients)
  E   process_arrays() takes plain arrays so the CSV loader and the .mat
      loader share the same processing
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


MIN_NOTCH_HZ = 15.0         # below this a comb is blanked, never notched
NOTCH_MAX_HZ = 500.0        # no point notching above the passband


def build_filter(fs=FS, band=BAND, stim_hz=STIM_HZ, mains_hz=MAINS_HZ, q=NOTCH_Q):
    """Bandpass plus stimulation and mains harmonics, as one second-order-section
    cascade.  Applied with sosfiltfilt it is zero-phase; the original applied
    52 notches through lfilter, which is causal and distorts phase.

    Harmonics stop at NOTCH_MAX_HZ (the original ran them to Nyquist, 124
    sections for nothing).  A stimulation rate under MIN_NOTCH_HZ gets no
    comb at all: at 2.5 Hz the notches sit closer together than their own
    width above ~75 Hz and the cascade removes the whole band.  Low-rate
    pulses are sparse in time, which is what blank_pulses is for.
    """
    sos = [signal.butter(4, band, btype='band', fs=fs, output='sos')]
    top = min(NOTCH_MAX_HZ, fs / 2)
    if stim_hz and stim_hz >= MIN_NOTCH_HZ:
        sos.append(_notch_sections(np.arange(stim_hz, top, stim_hz), q, fs))
    if mains_hz:
        sos.append(_notch_sections(np.arange(mains_hz, top, mains_hz), q, fs))
    return np.vstack(sos)


_FILTER_CACHE: dict = {}


def clean(x, fs=FS, remove_stim=True, remove_mains=True, stim_hz=None):
    """Filter one channel.  remove_stim=False reproduces the original behaviour.

    stim_hz is the rate measured on this recording (detect_pulse_train); the
    module default is only a fallback, because the rate differs by session.
    """
    f_stim = (STIM_HZ if stim_hz is None else float(stim_hz)) if remove_stim else None
    f_mains = MAINS_HZ if remove_mains else None
    key = (fs, f_stim, f_mains)
    if key not in _FILTER_CACHE:
        _FILTER_CACHE[key] = build_filter(fs=fs, stim_hz=f_stim, mains_hz=f_mains)
    return signal.sosfiltfilt(_FILTER_CACHE[key], np.nan_to_num(np.asarray(x, dtype=float)))


def moving_rms(x, fs=FS, window_s=0.2):
    """Same-length RMS envelope."""
    w = max(1, int(round(window_s * fs)))
    x = np.asarray(x, dtype=float)
    return np.sqrt(np.convolve(x * x, np.ones(w) / w, mode="same"))


def rms_envelope(x, fs=FS, window_s=0.05, overlap=0.5):
    """Windowed RMS.  window_s=0.05 matches the original's 500-sample window
    (its comment said 0.01 s, which was wrong - finding 10)."""
    w = max(1, int(round(window_s * fs)))
    step = max(1, int(round(w * (1 - overlap))))
    x = np.asarray(x, dtype=float)
    if len(x) < w:
        return np.array([np.sqrt(np.mean(np.square(x)))]) if len(x) else np.array([0.0])
    c = np.concatenate([[0.0], np.cumsum(x * x)])
    starts = np.arange(0, len(x) - w + 1, step)
    return np.sqrt((c[starts + w] - c[starts]) / w)


# --------------------------------------------------------------------------
# Stimulation pulses
# --------------------------------------------------------------------------
def spike_times(x, fs=FS, k_slope=10.0, k_amp=4.0, refractory_s=0.010):
    """Sample indices of stimulation spikes.  -> (indices, |dx|, settled-slope level)

    A spike must be steep AND tall.  Steepness (|dx| above k_slope MADs)
    is what survives an artifact-dominated channel: the amplitude MAD
    inflates with the artifact and an amplitude-only threshold clears the
    whole trace (P4 22 Dec 2025, R psoas: MAD 33 uV, zero spikes), whereas
    a pulse edge is extreme for only a few samples so the slope's MAD stays
    near the EMG floor.  Height (|x| above k_amp MADs) is what rejects
    spiky noise, which is steep but small (P1 W10 stim-off: slope alone
    found a dense 5 ms "train" on every channel).

    The 10 ms refractory matters: a pulse is followed by echo lobes at
    +4/+5/+8 ms that are steep and tall too, and with a 2 ms refractory
    each pulse counted 2-3 times -- P1 W18's 20 ms period read as 5 + 15 ms
    and the "rate" came out as 1/15 ms = 68 Hz.  It caps detectable rates
    at 100 Hz, above anything this study delivers.
    """
    x = np.nan_to_num(np.asarray(x, dtype=float))
    d = np.abs(np.diff(x))
    mad_d = np.median(np.abs(d - np.median(d))) * 1.4826
    if mad_d <= 0:
        return np.array([], dtype=int), d, 0.0
    peaks, _ = signal.find_peaks(d, height=np.median(d) + k_slope * mad_d,
                                 distance=int(refractory_s * fs))
    a = np.abs(x - np.median(x))
    mad_a = np.median(np.abs(a - np.median(a))) * 1.4826
    if mad_a > 0 and len(peaks):
        tall = np.maximum(a[peaks], a[np.minimum(peaks + 1, len(a) - 1)]) >= np.median(a) + k_amp * mad_a
        peaks = peaks[tall]
    return peaks, d, float(np.median(d) + 3.0 * mad_d)


def detect_pulse_train(x, fs=FS, fmin=1.5, fmax=100.0, min_consistency=0.5, min_density=0.5):
    """-> (pulse rate in Hz or None, consistency).

    Measured from the spacing of artifact spikes rather than from the
    spectrum.  On these recordings the spectrum misleads: the mains ladder
    is stronger than the stimulation fundamental, a 30 Hz acquisition
    high-pass removes a 20 Hz fundamental outright leaving only harmonics,
    and a 50 Hz stimulator (P1 W18) is indistinguishable from mains.  A
    stimulation pulse is a sharp biphasic spike, so its spacing is direct.

    A train is accepted only when most gaps agree with the modal gap AND
    the spike count is at least half of what that rate predicts over the
    clip.  Noise excursions fail both.
    """
    x = np.nan_to_num(np.asarray(x, dtype=float))
    peaks, _, _ = spike_times(x, fs)
    if len(peaks) < 20:
        return None, 0.0
    gaps = np.diff(peaks) / fs
    gaps = gaps[(gaps >= 1.0 / fmax) & (gaps <= 1.0 / fmin)]
    if len(gaps) < 10:
        return None, 0.0
    hist, edges = np.histogram(gaps, bins=np.arange(1.0 / fmax, 1.0 / fmin + 5e-4, 5e-4))
    i = int(np.argmax(hist))
    if i == 0 or i == len(hist) - 1:          # piled against a boundary: not a train
        return None, 0.0
    modal = float(edges[i] + edges[i + 1]) / 2
    # a doubled gap is one missed (or sub-threshold) pulse, not a different train:
    # alternating-amplitude trains (P4 5 Jan 2026) drop every other pulse below the gate
    consistent = (np.abs(gaps - modal) < 0.1 * modal) | (np.abs(gaps - 2 * modal) < 0.1 * modal)
    consistency = float(np.mean(consistent))
    rate = 1.0 / modal
    density = len(peaks) / (len(x) / fs) / rate
    if consistency < min_consistency or density < min_density:
        return None, round(consistency, 3)
    return round(rate, 2), round(consistency, 3)


def detect_stim_comb(x, fs=FS, f0_range=(1.5, 100.0), step=0.1, band_hz=120.0, max_harm=80,
                     line_db=6.0, presence=0.6, min_harmonics=4, cap_db=30.0,
                     min_mean_db=8.0, mains_hz=MAINS_HZ):
    """Fundamental of a harmonic comb in the spectrum, or None.  -> (f0, mean dB)

    Complements detect_pulse_train for low-rate stimulation (2.5 Hz in P4
    22 Dec 2025), where each pulse evokes a multi-peaked response so spike
    spacing is ambiguous, but the spectrum shows a dense, unmistakable comb.

    Scoring: every candidate is judged over the same band (0..band_hz), so
    a candidate with a smaller f0 has more harmonic slots.  It is valid
    when at least `presence` of those slots carry a line (> line_db), and
    its score is the sum of their prominences (each capped).  The sum
    rewards explaining more lines, so a super-harmonic (7.5 Hz for a 2.5 Hz
    train) loses to the fundamental; the presence rule rejects
    sub-harmonics (half their slots are empty).  Scoring a fixed number of
    harmonics per candidate instead lets a super-harmonic win whenever its
    lines happen to be the strong ones.

    Harmonics within 1 Hz of a mains multiple are skipped, so a 50 Hz
    stimulator is invisible here; that case belongs to the time domain.
    A comb can also be environmental (a ~68 Hz ladder appears in some
    stim-off recordings): notching it is harmless, labelling it as
    stimulation is not, so QC reports these per recording.
    """
    x = np.nan_to_num(np.asarray(x, dtype=float))
    f, p = signal.welch(x, fs=fs, nperseg=min(200_000, len(x)))
    df = f[1] - f[0]
    kern = min(max(3, int(round(8.0 / df)) | 1), (len(p) - 1) | 1)
    bg = signal.medfilt(p, kernel_size=kern)
    with np.errstate(divide="ignore", invalid="ignore"):
        prom = np.nan_to_num(10 * np.log10(p / bg), neginf=0.0, posinf=0.0)

    f0s = np.arange(f0_range[0], f0_range[1] + 1e-9, step)
    fh = f0s[:, None] * np.arange(1, max_harm + 1)[None, :]
    # a 25 Hz comb has 2 slots under 120 Hz once mains is skipped, so widen with f0
    top = np.minimum(fs / 2, np.maximum(band_hz, 8.0 * f0s))[:, None]
    valid = fh <= top
    if mains_hz:
        valid &= np.abs(fh - mains_hz * np.round(fh / mains_hz)) >= 1.0
    idx = np.clip(np.round(fh / df).astype(int), 1, len(prom) - 2)
    valid &= np.round(fh / df) < len(prom) - 1
    pm = np.clip(np.maximum(np.maximum(prom[idx - 1], prom[idx]), prom[idx + 1]), 0.0, cap_db)
    n_valid = valid.sum(axis=1)
    present = ((pm > line_db) & valid).sum(axis=1)
    score = (pm * valid).sum(axis=1)
    ok = (n_valid >= min_harmonics) & (present >= presence * n_valid)
    if not ok.any():
        return None, 0.0
    best = score[ok].max()
    i = int(np.flatnonzero(ok & (score >= best / 1.02))[0])     # ties within 2% -> lowest f0
    mean = float(score[i] / n_valid[i])
    if mean < min_mean_db:
        return None, round(mean, 1)
    return round(float(f0s[i]), 2), round(mean, 1)


ENV_COMB_HZ = 16.97
# An interference ladder at multiples of ~16.97 Hz (17, 34, 51, 68, 85 Hz)
# appears in stim-OFF baselines of P2, P3, P4, P5 and P6.  It is what the
# original pipeline's "17 Hz notch series" was chasing.  It is not
# stimulation, it is present in both conditions, and 29 notches at Q=30
# would remove half the upper band to kill a few 8-14 dB lines.


def is_environmental(rate, method, tol=0.03):
    """True for a spectral comb sitting on the ENV_COMB_HZ ladder.  A spike
    train (method 'pulses') is never environmental: the ladder is not spiky,
    and a genuine 50 Hz stimulator (P1 W18, P2 W9, P3 W12) lands near its
    third harmonic."""
    if rate is None or method != "comb":
        return False
    k = round(rate / ENV_COMB_HZ)
    return k >= 1 and abs(rate - k * ENV_COMB_HZ) <= tol * rate


def measure_stim(x, fs=FS):
    """-> (rate or None, method) where method is 'pulses', 'comb' or None."""
    rate, _ = detect_pulse_train(x, fs)
    if rate is not None:
        return rate, "pulses"
    rate, _ = detect_stim_comb(x, fs)
    return (rate, "comb") if rate is not None else (None, None)


def pulse_train_consensus(data, fs=FS, channels=None, tol=0.1):
    """Stimulation rate agreed by at least two channels, else None.

    -> (rate, n_agreeing, method, {channel: (rate, method)})
    Time-domain spacing is tried first; the spectral comb only if no two
    channels agree on a spike train.
    """
    channels = list(range(data.shape[0])) if channels is None else list(channels)
    per = {ch: measure_stim(data[ch], fs) for ch in channels}
    octaves = np.array([1.0, 2.0, 0.5])

    def related(rates, cand):
        return np.min(np.abs((rates / cand)[:, None] - octaves), axis=1) <= tol

    found = {}
    for method in ("pulses", "comb"):
        rates = np.array([r for r, m in per.values() if r is not None and m == method])
        if rates.size == 0:
            continue
        # rates related by 2x are the same train read at different amplitude
        # gates (alternating pulses); they agree, and the higher one is the
        # pulse rate.  Pick the candidate that most channels agree with.
        best, best_n = None, 0
        for cand in rates:
            n = int(related(rates, cand).sum())
            if n > best_n or (n == best_n and best is not None and cand > best):
                best, best_n = float(cand), n
        found[method] = (round(float(rates[related(rates, best)].max()), 2), best_n)
    pulses, comb = found.get("pulses"), found.get("comb")
    if pulses and pulses[1] >= 2:
        return pulses[0], pulses[1], "pulses", per
    # one channel's spike train corroborated by the other channels' spectral
    # comb is two independent methods agreeing -- take it, so blanking runs
    if pulses and comb and comb[1] >= 2 and related(np.array([pulses[0]]), comb[0])[0]:
        return pulses[0], pulses[1] + comb[1], "pulses", per
    if comb and comb[1] >= 2:
        return comb[0], comb[1], "comb", per
    return None, 0, None, per


def blank_pulses(x, fs=FS, pre_ms=1.0, max_post_ms=12.0, anchor_ms=2.0):
    """Cut each stimulation spike out of the trace and bridge the gap.

    -> (blanked signal, number of spikes)

    A notch comb removes only the line components of a spike train; the
    spike itself is broadband and survives.  Removing it in time is the
    standard approach for stimulation artifact.  Spikes are located on the
    slope (spike_times); the window after each extends until the slope has
    settled (up to max_post_ms); and each end of the bridge is anchored on
    the median of a short stretch rather than a single sample -- anchoring
    on one sample lands on the pulse tail and the bridge ramps up to it,
    which *adds* energy.
    """
    x = np.nan_to_num(np.asarray(x, dtype=float))
    y = x.copy()
    peaks, d, quiet = spike_times(x, fs)
    if not len(peaks):
        return y, 0
    pre, maxpost, anc = (int(v * fs / 1000) for v in (pre_ms, max_post_ms, anchor_ms))
    n = len(x)
    for p in peaks:
        if p < anc + pre or p > n - anc - 3:      # no room for an anchor on that side
            continue
        lo = p - pre
        hi, run = p + 2, 0
        limit = min(n - anc - 1, p + maxpost)
        while hi < limit:
            run = run + 1 if d[hi - 1] < quiet else 0
            if run >= 3:
                break
            hi += 1
        left = float(np.median(y[lo - anc:lo]))
        right = float(np.median(x[hi:hi + anc]))
        y[lo:hi] = np.linspace(left, right, hi - lo, endpoint=False)
    return y, int(len(peaks))


def remove_artifact(x, fs=FS, stim_hz=None, blank=True):
    """Blank the pulses (if a rate was found), then bandpass + notches.

    stim_hz=None means no stimulation was detected on this recording, so
    only the mains notches are applied.  Notching at a rate the recording
    does not have carves holes in clean EMG (finding A).  blank=False when
    the rate came from the spectral comb: there are no sharp spikes to cut.
    """
    x = np.nan_to_num(np.asarray(x, dtype=float))
    if stim_hz is not None and blank:
        x, _ = blank_pulses(x, fs)
    return clean(x, fs, remove_stim=stim_hz is not None, remove_mains=True, stim_hz=stim_hz)


def robust_peak(x, percentile=99.0):
    """Finding 17: max() is a single sample and one artifact spike rescales a
    whole trial."""
    return float(np.percentile(np.abs(x), percentile))


# --------------------------------------------------------------------------
# Activation detection
# --------------------------------------------------------------------------
def detect_activation(trial, rest, fs=FS, method="rest_p99", k=3.0, smooth_hz=0.7,
                      window_s=0.2, percentile=99.0, scale=1.5,
                      pad_s=0.5, min_dur_s=0.2, edge_s=0.5):
    """Samples where the muscle is active.

    The threshold is derived from the *rest* recording.  The original used
    1.2 * mean of the trial's own envelope, which is circular: a weak trial
    lowers its own bar, so noise is admitted as movement exactly in the
    sessions where the patient is weakest (finding 04).

    method="rest_p99" (default): scale * the `percentile` of the rest
    recording's moving-RMS envelope (window_s).  On P5 pre-op this is the
    only rule that both catches reps sitting 1.5x above rest and stays
    silent on a flat recording (finding C).

    method="rest_sd": mean + k*sd of a smooth_hz-lowpassed |x|.  Kept for
    reproducing earlier numbers.  That envelope is nearly flat so its sd is
    tiny, and the threshold lands below the trial's own rest floor whenever
    the two recordings differ slightly -- whole trials come out active.

    The first and last edge_s seconds are never active: zero-phase filtering
    rings there (finding D).
    """
    if method == "rest_sd":
        b, a = signal.butter(2, smooth_hz, btype='low', fs=fs)
        env_trial = signal.filtfilt(b, a, np.abs(trial))
        env_rest = signal.filtfilt(b, a, np.abs(rest))
        threshold = float(np.mean(env_rest) + k * np.std(env_rest))
    elif method == "rest_p99":
        env_trial = moving_rms(trial, fs, window_s)
        env_rest = moving_rms(rest, fs, window_s)
        threshold = float(scale * np.percentile(env_rest, percentile))
    else:
        raise ValueError(f"unknown method {method!r}")
    active = env_trial > threshold
    edge = int(edge_s * fs)
    if edge and len(active) > 2 * edge:
        active[:edge] = False
        active[-edge:] = False

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
    stim_hz: float | None = None      # pulse rate measured on this recording

    @property
    def active_fraction(self):
        return self.n_active / self.n_total if self.n_total else 0.0

    def rms(self):
        return float(np.sqrt(np.mean(np.square(self.signal)))) if self.signal.size else np.nan


def process_arrays(data, rest, trigger_ch, electrodes, fs=FS, remove_stim=True,
                   gains=None, stim_hz="auto", label=("", 0, 0, False), paths=()):
    """Process one exercise given its (n_ch, N) array and its rest recording.

    Format-agnostic core shared by the .mat path (process_trial) and the CSV
    path (restores_csv / run_csv_qc).  stim_hz="auto" measures the pulse
    rate on the trigger channel of the trial and of the rest recording and
    uses whichever is found (finding A); None means "there is no train";
    a number forces that rate.  label = (patient, week, exercise, stim_on).
    """
    blank = True
    if stim_hz == "auto" and remove_stim:
        stim_hz, method = measure_stim(data[trigger_ch], fs)
        if stim_hz is None:
            stim_hz, method = measure_stim(rest[trigger_ch], fs)
        if is_environmental(stim_hz, method):
            stim_hz, method = None, None
        blank = method == "pulses"
    elif not remove_stim:
        stim_hz = None

    def prep(x):
        return (remove_artifact(x, fs, stim_hz=stim_hz, blank=blank) if remove_stim
                else clean(x, fs, remove_stim=False))

    mask, threshold = detect_activation(prep(data[trigger_ch]), prep(rest[trigger_ch]), fs=fs)

    patient, week, number, stim_on = label
    out = []
    for ch in electrodes:
        g = None if gains is None else gains.get(ch)
        x = prep(data[ch]) * gain_scale(g)
        out.append(Trial(patient, week, number, stim_on, ch, x[mask],
                         int(mask.sum()), int(mask.size), threshold, g, list(paths), stim_hz))
    return out


def process_trial(patient, week, canonical_exercise, stim_on, electrodes,
                  remove_stim=True, gains=None, fs=FS, stim_hz="auto"):
    """Load one exercise for one session (.mat tree) and return a Trial per electrode.

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

    return process_arrays(load_concatenated(paths), load_recording(baseline_path),
                          AGONISTS[canonical_exercise][0], electrodes, fs=fs,
                          remove_stim=remove_stim, gains=gains, stim_hz=stim_hz,
                          label=(patient, week, number, stim_on), paths=paths)


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


def qc_row(trial, raw_channel, fs=FS, electrode_names=None):
    """Per-trial quality metrics.  The original emitted nothing like this, so a
    26x amplitude jump or a session with no detected movement was invisible
    (finding 12 in the Tier 3 list)."""
    names = ELECTRODES if electrode_names is None else electrode_names
    f0 = trial.stim_hz if trial.stim_hz else STIM_HZ
    return {
        "patient": trial.patient, "week": trial.week, "exercise": trial.exercise,
        "stim_on": trial.stim_on, "electrode": names[trial.electrode],
        "n_trials": len(trial.paths),
        "duration_s": round(trial.n_total / fs, 1),
        "active_fraction": round(trial.active_fraction, 3),
        "rms": round(trial.rms(), 4) if np.isfinite(trial.rms()) else None,
        "threshold": round(trial.threshold, 4),
        "stim_hz": trial.stim_hz,
        "artifact_dB": round(artifact_prominence(raw_channel, fs, f0=f0), 1),
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
