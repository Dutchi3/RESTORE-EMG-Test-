# -*- coding: utf-8 -*-
"""
Markerless gait kinematics from the RESTORES walking videos (MediaPipe Pose).

Ported from Kezia Susanto's ``gait_analysis_mediapipe.py`` (Kezia_Data/
RESTORES_Walk on the share) and extended.  Her script targets the legacy
``mp.solutions.pose`` API, which no longer exists in mediapipe >= 1.0 (the
only version with wheels for this venv's Python 3.14), so the tracking here
uses the Tasks ``PoseLandmarker`` with the same model family (heavy = her
``model_complexity=2``).

What was wrong or missing in the original, and what this module does instead
(each point verified on the two videos on the share, 2026-09-15):

  * The videos are FRONTAL: the patient walks a 15 m track towards a camera
    operator walking backwards.  Hip/knee/ankle flexion happen in the
    sagittal plane, which is the depth axis of that view, so 2-D image-plane
    angles are foreshortening artefacts there.  Flexion angles are therefore
    computed from MediaPipe's 3-D *world* landmarks (metres, origin at the
    hip midpoint, y down, z away from the camera).  The 2-D angles are still
    produced -- in a side-on video they are the better estimate -- and every
    frame carries a ``view_ratio`` so the reader knows which plane a 2-D
    angle actually lives in.
  * The original computed angles from *normalised* coordinates (x/width,
    y/height).  On a 720x1280 portrait frame that shrinks vertical distances
    by 0.56 relative to horizontal ones and bends every angle.  Angles here
    are computed in pixels; the normalised version is kept as ``*_2dnorm``
    only so the old CSVs can be reconciled.
  * No smoothing: landmark trajectories are low-passed (zero-phase
    Butterworth, 6 Hz -- the usual cut-off for gait kinematics) before any
    angle or event is derived.  Short visibility gaps are interpolated,
    long ones stay NaN.
  * No gait events: heel strike and toe off are detected from the heel /
    toe displacement along the body's own forward axis (Zeni et al. 2008,
    Gait & Posture 27:710), which does not care which way the camera looks.
    A 2-D fallback for frontal video uses the heel's image height instead.
  * Strides are validated (duration, visibility, exactly one toe-off), time-
    normalised to 0-100 % and summarised: stride time and its variability,
    cadence, stance fraction, joint ranges, left/right symmetry, cycle
    consistency, step width and trunk sway (the frontal-plane measures this
    camera angle is actually good at).

Nothing here is synchronised to the EMG: the videos carry no clock or sync
marker, so video and EMG from the same session can be compared per stride
statistics, not frame by frame.

Validation, P4 28 Jan 2026 (3209 frames):
  * port check -- our ``*_2dnorm`` angles vs Kezia's CSV: r 0.83-0.98,
    MAE 2.6-5.1 deg (residual = Tasks vs legacy model and her temporal
    landmark smoothing); pixel-space angles differ from normalised ones by
    8-22 deg on average; 3-D vs 2-D flexion on this frontal video: r only
    0.24-0.66, MAE 12-44 deg.
  * events -- 21 L / 22 R heel strikes, stride 4.97 / 5.05 s (CV 0.125),
    cadence 24 steps/min, checked frame by frame on 5.5-14.5 s; the EMG file
    of the same session is named "22 full gaits".  A fixed 0.5 s minimum
    event spacing (healthy cadence) fired 3-4x per stride, hence the
    adaptive period.
  * stance fraction is approximate on frontal video (toe off = onset of the
    toe's lift in the image); stride and step timing are the robust outputs.
P5 4 Sep 2026 (2128 frames): port check r 0.57-0.96, MAE 2.1-4.0 deg;
16 L / 17 R heel strikes, stride 4.05 / 4.28 s, cadence 29 steps/min, knee
3-D range 21 deg left vs 12 deg right.
"""
from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
from scipy import signal

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_URLS = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}
LEGACY_COMPLEXITY = {0: "lite", 1: "full", 2: "heavy"}   # mp.solutions.pose model_complexity

# MediaPipe pose landmark indices (left, right)
LM = {"SH": (11, 12), "HP": (23, 24), "KN": (25, 26), "AN": (27, 28), "HL": (29, 30), "FT": (31, 32)}
SIDES = ("L", "R")
SMOOTH_HZ = 6.0
MIN_VIS = 0.35
UP = np.array([0.0, -1.0, 0.0])    # world y points down


def model_path(name="heavy", model_dir=MODEL_DIR):
    """Local path of a pose model, downloading it on first use (~10-30 MB)."""
    if name not in MODEL_URLS:
        raise ValueError(f"model must be one of {list(MODEL_URLS)}, got {name!r}")
    os.makedirs(model_dir, exist_ok=True)
    p = os.path.join(model_dir, f"pose_landmarker_{name}.task")
    if not os.path.exists(p) or os.path.getsize(p) < 1_000_000:
        print(f"downloading {name} pose model -> {p}", flush=True)
        urllib.request.urlretrieve(MODEL_URLS[name], p)
    return p


# --------------------------------------------------------------------------
# Pass 1: landmark tracking
# --------------------------------------------------------------------------
@dataclass
class Track:
    path: str
    fps: float
    width: int
    height: int
    t: np.ndarray            # (n,) seconds from first frame
    px: np.ndarray           # (n, 33, 2) pixel coordinates, NaN when no pose
    vis: np.ndarray          # (n, 33) visibility
    world: np.ndarray        # (n, 33, 3) metres, origin hip midpoint
    n_poses: np.ndarray      # (n,) poses returned by the detector
    model: str

    @property
    def n_frames(self):
        return len(self.t)

    @property
    def detected(self):
        return np.isfinite(self.px[:, LM["HP"][0], 0])


def _pick_pose(result, prev_hip, width, height):
    """Index of the pose to follow: nearest to the last hip position, else largest."""
    best, best_score = None, None
    for i, lms in enumerate(result.pose_landmarks):
        hl, hr, sl, sr = (lms[k] for k in (LM["HP"][0], LM["HP"][1], LM["SH"][0], LM["SH"][1]))
        hip = np.array([(hl.x + hr.x) / 2 * width, (hl.y + hr.y) / 2 * height])
        if prev_hip is not None and np.all(np.isfinite(prev_hip)):
            score = -np.linalg.norm(hip - prev_hip)
        else:
            sh = np.array([(sl.x + sr.x) / 2 * width, (sl.y + sr.y) / 2 * height])
            score = np.linalg.norm(hip - sh)
        if best_score is None or score > best_score:
            best, best_score = i, score
    return best


def track_video(path, model="heavy", num_poses=1, min_detection=0.5, min_presence=0.5,
                min_tracking=0.5, max_frames=None, progress=True):
    """Run the pose landmarker over every frame.  ~8 fps with 'heavy' on CPU."""
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python import vision

    if not os.path.exists(path):
        raise FileNotFoundError(path)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames:
        n_total = min(n_total, int(max_frames))

    opts = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path(model)),
        running_mode=vision.RunningMode.VIDEO, num_poses=num_poses,
        min_pose_detection_confidence=min_detection, min_pose_presence_confidence=min_presence,
        min_tracking_confidence=min_tracking)
    lm = vision.PoseLandmarker.create_from_options(opts)

    px = np.full((n_total, 33, 2), np.nan)
    vis = np.full((n_total, 33), np.nan)
    world = np.full((n_total, 33, 3), np.nan)
    n_poses = np.zeros(n_total, dtype=int)
    t = np.arange(n_total) / fps
    prev_hip, last_ts, t0 = None, -1, time.time()
    i = 0
    while i < n_total:
        ok, frame = cap.read()
        if not ok:
            break
        ts = int(round(i * 1000.0 / fps))
        if ts <= last_ts:                       # the API demands strictly increasing stamps
            ts = last_ts + 1
        last_ts = ts
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
        n_poses[i] = len(res.pose_landmarks)
        if res.pose_landmarks:
            j = _pick_pose(res, prev_hip, width, height)
            L, W = res.pose_landmarks[j], res.pose_world_landmarks[j]
            px[i] = [(l.x * width, l.y * height) for l in L]
            vis[i] = [l.visibility for l in L]
            world[i] = [(w.x, w.y, w.z) for w in W]
            prev_hip = px[i, list(LM["HP"])].mean(axis=0)
        i += 1
        if progress and i % 300 == 0:
            el = time.time() - t0
            print(f"  {i}/{n_total} frames  {i/el:.1f} fps  eta {(n_total-i)/(i/el):.0f}s", flush=True)
    lm.close()
    cap.release()
    if i < n_total:                             # container header over-reported the count
        px, vis, world, n_poses, t = px[:i], vis[:i], world[:i], n_poses[:i], t[:i]
    return Track(path, fps, width, height, t, px, vis, world, n_poses, model)


# MediaPipe's 33 pose landmarks, in index order.  All of them are cached so
# that a re-render from the cache draws the same full skeleton as a fresh
# track (an earlier cache kept only the 12 gait points, which silently
# dropped the arms and face from the annotated video).
POSE_NAMES = ["nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye",
              "right_eye_outer", "left_ear", "right_ear", "mouth_left", "mouth_right",
              "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
              "left_pinky", "right_pinky", "left_index", "right_index", "left_thumb", "right_thumb",
              "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
              "left_heel", "right_heel", "left_foot_index", "right_foot_index"]
_LM_FIELDS = ("x", "y", "vis", "wx", "wy", "wz")


def save_track(track: Track, path):
    """Raw per-frame landmarks (pixels, visibility, world metres), all 33 points."""
    cols = {"frame": np.arange(track.n_frames), "t": track.t, "n_poses": track.n_poses}
    for k, name in enumerate(POSE_NAMES):
        cols[f"{name}_x"] = track.px[:, k, 0]
        cols[f"{name}_y"] = track.px[:, k, 1]
        cols[f"{name}_vis"] = track.vis[:, k]
        cols[f"{name}_wx"] = track.world[:, k, 0]
        cols[f"{name}_wy"] = track.world[:, k, 1]
        cols[f"{name}_wz"] = track.world[:, k, 2]
    pd.DataFrame(cols).to_csv(path, index=False, float_format="%.4f")


def cache_is_current(landmarks_csv):
    """False for caches written before all 33 landmarks were stored."""
    try:
        header = pd.read_csv(landmarks_csv, nrows=0).columns
    except Exception:
        return False
    return all(f"{name}_{f}" in header for name in POSE_NAMES for f in _LM_FIELDS)


def load_track(landmarks_csv, path, fps, width, height, model="?"):
    """Rebuild a Track from save_track output, to skip re-tracking."""
    df = pd.read_csv(landmarks_csv)
    n = len(df)
    px, vis, world = np.full((n, 33, 2), np.nan), np.full((n, 33), np.nan), np.full((n, 33, 3), np.nan)
    for k, name in enumerate(POSE_NAMES):
        px[:, k, 0], px[:, k, 1] = df[f"{name}_x"], df[f"{name}_y"]
        vis[:, k] = df[f"{name}_vis"]
        world[:, k] = df[[f"{name}_wx", f"{name}_wy", f"{name}_wz"]].to_numpy()
    return Track(path, fps, width, height, df["t"].to_numpy(), px, vis, world,
                 df["n_poses"].to_numpy(), model)


# --------------------------------------------------------------------------
# Signal helpers
# --------------------------------------------------------------------------
def _long_gaps(ok, max_len):
    """Mask of samples inside runs of ~ok longer than max_len."""
    out = np.zeros(len(ok), dtype=bool)
    bad = ~np.asarray(ok, bool)
    if not bad.any():
        return out
    edges = np.flatnonzero(np.diff(np.r_[False, bad, False]))
    for a, b in zip(edges[::2], edges[1::2]):
        if b - a > max_len:
            out[a:b] = True
    return out


def smooth_series(x, fps, cutoff=SMOOTH_HZ, order=4, max_gap_s=0.5):
    """Zero-phase low-pass of a series with NaN gaps.  Gaps shorter than
    max_gap_s are bridged by interpolation; longer ones (and the unobserved
    ends) come back as NaN."""
    x = np.asarray(x, float)
    n = len(x)
    ok = np.isfinite(x)
    out = np.full(n, np.nan)
    if ok.sum() < 3 * (2 * order + 1) + 1:
        return out
    idx = np.arange(n)
    xi = np.interp(idx, idx[ok], x[ok])
    sos = signal.butter(order, cutoff, btype="low", fs=fps, output="sos")
    xs = signal.sosfiltfilt(sos, xi)
    xs[_long_gaps(ok, int(round(max_gap_s * fps)))] = np.nan
    first, last = idx[ok][0], idx[ok][-1]
    xs[:first] = np.nan
    xs[last + 1:] = np.nan
    return xs


def joint_angle(a, b, c):
    """Interior angle ABC in degrees at vertex b; arrays (n, d) with d = 2 or 3.
    180 = segments in line (straight knee); NaN propagates."""
    ba, bc = a - b, c - b
    denom = np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.sum(ba * bc, axis=-1) / denom
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def _peaks(x, fps, min_dist_s, prom_frac):
    """Peak indices of a NaN-bearing series; peaks inside NaN stretches are dropped."""
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    if ok.sum() < 10:
        return np.array([], dtype=int)
    idx = np.arange(len(x))
    xi = np.interp(idx, idx[ok], x[ok])
    lo, hi = np.nanpercentile(x, [5, 95])
    pk, _ = signal.find_peaks(xi, distance=max(1, int(min_dist_s * fps)), prominence=prom_frac * (hi - lo))
    return pk[ok[pk]]


# --------------------------------------------------------------------------
# Pass 2: kinematics
# --------------------------------------------------------------------------
def compute_kinematics(track: Track, min_vis=MIN_VIS, cutoff=SMOOTH_HZ):
    """One row per frame: validity, view, joint angles (2-D pixel, 3-D world,
    legacy 2-D normalised), frontal-plane measures and the event signals."""
    n, fps = track.n_frames, track.fps
    used = sorted({k for pair in LM.values() for k in pair})
    px = track.px.copy()
    world = track.world.copy()
    low = ~(track.vis >= min_vis)                       # NaN vis counts as low
    px[low] = np.nan
    world[low] = np.nan
    for k in used:                                       # filter trajectories, not angles
        for d in range(2):
            px[:, k, d] = smooth_series(px[:, k, d], fps, cutoff)
        for d in range(3):
            world[:, k, d] = smooth_series(world[:, k, d], fps, cutoff)

    out = {"frame": np.arange(n), "t": track.t, "detected": track.detected, "n_poses": track.n_poses}
    hipL, hipR = px[:, LM["HP"][0]], px[:, LM["HP"][1]]
    shL, shR = px[:, LM["SH"][0]], px[:, LM["SH"][1]]
    hip_mid, sh_mid = (hipL + hipR) / 2, (shL + shR) / 2
    pelvis_w = np.linalg.norm(hipL - hipR, axis=1)
    torso = np.linalg.norm(sh_mid - hip_mid, axis=1)
    shoulder_w = np.linalg.norm(shL - shR, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["view_ratio"] = shoulder_w / torso          # ~0.8 facing the camera, ~0.2 side-on
        out["step_width_ratio"] = np.abs(px[:, LM["AN"][0], 0] - px[:, LM["AN"][1], 0]) / pelvis_w
        out["trunk_sway_ratio"] = (sh_mid[:, 0] - hip_mid[:, 0]) / pelvis_w
    out["pelvis_width_px"] = pelvis_w
    out["torso_len_px"] = torso

    # body-fixed forward axis from the 3-D hips: cross(left - right, up)
    wHL, wHR = world[:, LM["HP"][0]], world[:, LM["HP"][1]]
    fwd = np.cross(wHL - wHR, UP)
    with np.errstate(invalid="ignore", divide="ignore"):
        fwd = fwd / np.linalg.norm(fwd, axis=1, keepdims=True)
    out["facing_z"] = fwd[:, 2]                          # < 0: facing the camera
    out["facing_x"] = fwd[:, 0]                          # > 0: walking towards image right
    w_hip_mid = (wHL + wHR) / 2

    for side, s in zip(SIDES, (0, 1)):
        SH, HP, KN, AN, HL, FT = (px[:, LM[k][s]] for k in ("SH", "HP", "KN", "AN", "HL", "FT"))
        ok = np.isfinite(np.stack([SH, HP, KN, AN, HL, FT], axis=1)).all(axis=(1, 2))
        out[f"{side}_ok"] = ok
        out[f"{side}_hip_2d"] = joint_angle(SH, HP, KN)
        out[f"{side}_knee_2d"] = joint_angle(HP, KN, AN)
        out[f"{side}_ankle_2d"] = joint_angle(KN, AN, FT)
        wSH, wHP, wKN, wAN, wFT, wHLm = (world[:, LM[k][s]] for k in ("SH", "HP", "KN", "AN", "FT", "HL"))
        out[f"{side}_hip_3d"] = joint_angle(wSH, wHP, wKN)
        out[f"{side}_knee_3d"] = joint_angle(wHP, wKN, wAN)
        out[f"{side}_ankle_3d"] = joint_angle(wKN, wAN, wFT)
        # legacy: raw normalised coordinates, exactly as the original script
        rx = track.px[:, :, 0] / track.width
        ry = track.px[:, :, 1] / track.height
        raw = np.stack([rx, ry], axis=-1)
        rok = (track.vis[:, [LM[k][s] for k in ("SH", "HP", "KN", "AN", "HL", "FT")]] >= min_vis).all(axis=1)
        for name, (a, b, c) in {"hip": ("SH", "HP", "KN"), "knee": ("HP", "KN", "AN"), "ankle": ("KN", "AN", "FT")}.items():
            v = joint_angle(raw[:, LM[a][s]], raw[:, LM[b][s]], raw[:, LM[c][s]])
            v[~rok] = np.nan
            out[f"{side}_{name}_2dnorm"] = v
        # event signals: anterior displacement in the body frame (m) and image height (2-D)
        out[f"{side}_heel_fwd"] = np.sum((wHLm - w_hip_mid) * fwd, axis=1)
        out[f"{side}_toe_fwd"] = np.sum((wFT - w_hip_mid) * fwd, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[f"{side}_heel_y_rel"] = (HL[:, 1] - hip_mid[:, 1]) / torso
            out[f"{side}_toe_y_rel"] = (FT[:, 1] - hip_mid[:, 1]) / torso
            out[f"{side}_heel_x_rel"] = (HL[:, 0] - hip_mid[:, 0]) / torso
            out[f"{side}_toe_x_rel"] = (FT[:, 0] - hip_mid[:, 0]) / torso
    kin = pd.DataFrame(out)

    hm = np.where(track.detected[:, None], track.px[:, list(LM["HP"])].mean(axis=1), np.nan)
    jump = np.linalg.norm(np.diff(hm, axis=0), axis=1) > 0.2 * track.height
    kin["tracker_jump"] = np.r_[False, np.nan_to_num(jump, nan=False).astype(bool)]
    return kin


def view_label(kin):
    r = np.nanmedian(kin["view_ratio"])
    if not np.isfinite(r):
        return "unknown"
    return "frontal" if r > 0.6 else ("sagittal" if r < 0.3 else "oblique")


# --------------------------------------------------------------------------
# Gait events and strides
# --------------------------------------------------------------------------
def _interp_nan(x):
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    if ok.sum() < 2:
        return np.zeros_like(x)
    idx = np.arange(len(x))
    return np.interp(idx, idx[ok], x[ok])


def dominant_period(x, fps, lo_s=0.6, hi_s=12.0, min_r=0.15):
    """Stride period (s) from the first strong autocorrelation peak, or None.

    The RESTORES walks are slow (2-2.5 s per step for P4), so fixed minimum
    spacings tuned for healthy cadence split every stride into several
    events; everything downstream is scaled by this estimate instead."""
    x = _interp_nan(x)
    ok = np.isfinite(np.asarray(x, float))
    if len(x) < int(2 * lo_s * fps) or ok.sum() < 10:
        return None
    x = x - np.mean(x)
    sd = np.std(x)
    if sd == 0:
        return None
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    ac = ac / ac[0]
    lo, hi = int(lo_s * fps), min(int(hi_s * fps), len(ac) - 2)
    if hi <= lo:
        return None
    pk, props = signal.find_peaks(ac[lo:hi], height=min_r)
    if not len(pk):
        return None
    return float((lo + pk[np.argmax(props["peak_heights"])]) / fps)


def walking_mask(x, fps, period_s, frac=0.4):
    """True where the signal still oscillates: rolling range over one period
    above frac x the global range.  Removes the standing start / stop."""
    s = pd.Series(_interp_nan(x))
    w = max(3, int(round(period_s * fps)))
    rng = (s.rolling(w, center=True, min_periods=1).max() - s.rolling(w, center=True, min_periods=1).min()).to_numpy()
    lo, hi = np.nanpercentile(x, [5, 95])
    return rng > frac * (hi - lo)


def _end_of_rise(x, fps, peak, vel_frac=0.3, max_shift_s=1.5):
    """Heel strike from a raw maximum of the heel signal.

    A landing is a fast rise that then plateaus while the foot is loaded, so
    the raw maximum can sit anywhere on the plateau.  The event is placed
    where the rise that produced the peak has slowed to vel_frac of its
    fastest rate."""
    v = np.gradient(_interp_nan(x)) * fps
    a = max(0, peak - int(max_shift_s * fps))
    seg = v[a:peak + 1]
    if not len(seg) or np.nanmax(seg) <= 0:
        return peak
    k = a + int(np.nanargmax(seg))
    thr = vel_frac * v[k]
    j = k
    while j + 1 <= peak and v[j + 1] > thr:
        j += 1
    return j


def _onset_of_fall(x, fps, trough, vel_frac=0.3, max_shift_s=1.5):
    """Toe off from a raw minimum of the toe image-height signal (frontal video).

    The toe recedes slowly through stance and drops fast once it lifts; the
    minimum is reached mid-swing.  The event is where the fast fall that
    leads to the minimum began (velocity reaches vel_frac of its most
    negative value)."""
    v = np.gradient(_interp_nan(x)) * fps
    a = max(0, trough - int(max_shift_s * fps))
    seg = v[a:trough + 1]
    if not len(seg) or np.nanmin(seg) >= 0:
        return trough
    k = a + int(np.nanargmin(seg))
    thr = vel_frac * v[k]
    j = k
    while j - 1 >= a and v[j - 1] < thr:
        j -= 1
    return j


def event_signals(kin, side, method):
    """(heel signal with HS at maxima, toe signal with TO at minima) for one side."""
    if method == "3d":
        return kin[f"{side}_heel_fwd"].to_numpy(), kin[f"{side}_toe_fwd"].to_numpy()
    if method == "2d-frontal":
        # walking towards the camera: the forward foot sits lower in the image
        return kin[f"{side}_heel_y_rel"].to_numpy(), kin[f"{side}_toe_y_rel"].to_numpy()
    if method == "2d-sagittal":
        d = np.sign(np.nanmedian(kin["facing_x"]))     # image-x direction of travel
        return d * kin[f"{side}_heel_x_rel"].to_numpy(), d * kin[f"{side}_toe_x_rel"].to_numpy()
    raise ValueError(method)


def choose_method(kin, method="auto"):
    if method != "auto":
        return method
    v = view_label(kin)
    return {"frontal": "2d-frontal", "sagittal": "2d-sagittal"}.get(v, "3d")


def detect_events(kin, fps, method="auto", prom_frac=0.3, period_s=None):
    """Heel strikes and toe offs per side, adaptive to the patient's cadence.

    methods
      '3d'          heel / toe anterior displacement in the body frame
                    (Zeni 2008); any camera direction, but relies on the
                    model's depth estimate
      '2d-frontal'  heel / toe image height relative to the hips; for the
                    RESTORES videos (patient walks towards the camera)
      '2d-sagittal' heel / toe horizontal displacement along the direction
                    of travel; side-on video
      'auto'        picked from the view ratio (frontal -> 2d-frontal,
                    sagittal -> 2d-sagittal, otherwise 3d)

    Steps: stride period from autocorrelation -> mask out standing ->
    peaks at least half a period apart -> heel strike placed at the end of
    the landing motion (30 % of peak velocity).  Toe off is the minimum of
    the toe's anterior displacement (Zeni) for '3d' / '2d-sagittal', and the
    onset of the toe's lift for '2d-frontal'.  Returns (events, period_s).
    """
    method = choose_method(kin, method)
    sig = {side: event_signals(kin, side, method) for side in SIDES}
    if period_s is None:
        ests = [dominant_period(s, fps) for pair in sig.values() for s in pair]
        ests = [e for e in ests if e]
        period_s = float(np.median(ests)) if ests else 1.2
    rows = []
    for side in SIDES:
        heel, toe = sig[side]
        mask = walking_mask(heel, fps, period_s)
        for f in _peaks(heel, fps, 0.5 * period_s, prom_frac):
            if mask[f]:
                rows.append((side, "HS", int(_end_of_rise(heel, fps, f))))
        for f in _peaks(-toe, fps, 0.5 * period_s, prom_frac):
            if mask[f]:
                rows.append((side, "TO", int(_onset_of_fall(toe, fps, f) if method == "2d-frontal" else f)))
    ev = pd.DataFrame(rows, columns=["side", "event", "frame"]).drop_duplicates()
    ev = ev.sort_values(["frame", "side"]).reset_index(drop=True)
    ev["t"] = kin["t"].to_numpy()[ev["frame"]] if len(ev) else []
    ev["method"] = method
    ev["period_s"] = round(period_s, 3)
    return ev, period_s


def stride_table(kin, ev, fps, period_s=None, min_valid=0.9):
    """One row per heel-strike-to-heel-strike stride, with validity checks.
    Accepted durations: 0.5-2 x the estimated stride period (else 0.4-15 s)."""
    rows = []
    t = kin["t"].to_numpy()
    if period_s:
        min_stride_s, max_stride_s = 0.5 * period_s, 2.0 * period_s
    else:
        min_stride_s, max_stride_s = 0.4, 15.0
    for side in SIDES:
        hs = ev[(ev.side == side) & (ev.event == "HS")]["frame"].to_numpy()
        to = ev[(ev.side == side) & (ev.event == "TO")]["frame"].to_numpy()
        other = ev[(ev.side != side) & (ev.event == "HS")]["frame"].to_numpy()
        ok = kin[f"{side}_ok"].to_numpy()
        knee3 = kin[f"{side}_knee_3d"].to_numpy()
        for a, b in zip(hs[:-1], hs[1:]):
            dur = (b - a) / fps
            tos = to[(to > a) & (to < b)]
            c_hs = other[(other > a) & (other < b)]
            seg_ok = ok[a:b].mean() if b > a else 0.0
            why = []
            if not (min_stride_s <= dur <= max_stride_s):
                why.append("duration")
            if len(tos) != 1:
                why.append(f"{len(tos)} toe-offs")
            if seg_ok < min_valid:
                why.append("visibility")
            if kin["tracker_jump"].to_numpy()[a:b].any():
                why.append("tracker jump")
            if len(c_hs) != 1:
                why.append(f"{len(c_hs)} contralateral HS")
            k = knee3[a:b]
            rows.append({
                "side": side, "hs_frame": int(a), "hs_next_frame": int(b), "t_hs": t[a], "t_hs_next": t[b],
                "stride_time_s": dur,
                "t_to": t[tos[0]] if len(tos) == 1 else np.nan,
                "stance_frac": (tos[0] - a) / (b - a) if len(tos) == 1 else np.nan,
                "step_time_s": (c_hs[0] - a) / fps if len(c_hs) == 1 else np.nan,
                "knee_3d_min": np.nanmin(k) if np.isfinite(k).any() else np.nan,
                "knee_3d_rom": (np.nanmax(k) - np.nanmin(k)) if np.isfinite(k).any() else np.nan,
                "hip_3d_rom": _rom(kin[f"{side}_hip_3d"].to_numpy()[a:b]),
                "ankle_3d_rom": _rom(kin[f"{side}_ankle_3d"].to_numpy()[a:b]),
                "knee_2d_rom": _rom(kin[f"{side}_knee_2d"].to_numpy()[a:b]),
                "peak_knee_flex_pct": (100.0 * (np.nanargmin(k) / (b - a))) if np.isfinite(k).any() else np.nan,
                "step_width_ratio_hs": kin["step_width_ratio"].to_numpy()[a],
                "valid_frac": seg_ok, "valid": not why, "reject_reason": "; ".join(why),
            })
    return pd.DataFrame(rows, columns=STRIDE_COLUMNS)


STRIDE_COLUMNS = ["side", "hs_frame", "hs_next_frame", "t_hs", "t_hs_next", "stride_time_s", "t_to", "stance_frac",
                  "step_time_s", "knee_3d_min", "knee_3d_rom", "hip_3d_rom", "ankle_3d_rom", "knee_2d_rom",
                  "peak_knee_flex_pct", "step_width_ratio_hs", "valid_frac", "valid", "reject_reason"]


def _rom(x):
    return (np.nanmax(x) - np.nanmin(x)) if np.isfinite(x).any() else np.nan


def normalize_cycles(kin, strides, columns, n_points=101):
    """-> {side: (n_valid_strides, n_points, len(columns))}, each stride resampled to 0-100 %."""
    out = {}
    grid = np.linspace(0, 1, n_points)
    for side in SIDES:
        st = strides[(strides.side == side) & strides.valid]
        cyc = []
        for _, r in st.iterrows():
            a, b = int(r.hs_frame), int(r.hs_next_frame)
            x = np.linspace(0, 1, b - a + 1)
            block = []
            for c in columns:
                y = kin[c].to_numpy()[a:b + 1]
                if not np.isfinite(y).all():
                    block = None
                    break
                block.append(np.interp(grid, x, y))
            if block is not None:
                cyc.append(np.stack(block, axis=1))
        out[side] = np.stack(cyc) if cyc else np.empty((0, n_points, len(columns)))
    return out


def _consistency(cycles):
    """Mean Pearson r between each stride's knee curve and the ensemble mean."""
    if len(cycles) < 2:
        return np.nan
    m = cycles.mean(axis=0)
    rs = [np.corrcoef(c, m)[0, 1] for c in cycles]
    return float(np.nanmean(rs))


def _si(l, r):
    """Symmetry index (%): 0 = identical sides."""
    if not (np.isfinite(l) and np.isfinite(r)) or (l + r) == 0:
        return np.nan
    return float(abs(l - r) / (0.5 * (l + r)) * 100)


def summarize(track: Track, kin, ev, strides, cycles, ev_alt=None, period_s=None):
    s = {"video": os.path.basename(track.path), "model": track.model, "fps": round(track.fps, 3),
         "size": f"{track.width}x{track.height}", "n_frames": track.n_frames,
         "duration_s": round(float(track.t[-1]), 2) if track.n_frames else 0.0,
         "detected_frac": round(float(track.detected.mean()), 3),
         "L_ok_frac": round(float(kin["L_ok"].mean()), 3), "R_ok_frac": round(float(kin["R_ok"].mean()), 3),
         "view": view_label(kin), "view_ratio_median": round(float(np.nanmedian(kin["view_ratio"])), 2),
         "facing_camera_frac": round(float(np.nanmean(kin["facing_z"] < 0)), 3),
         "multi_person_frac": round(float((kin["n_poses"] > 1).mean()), 3),
         "tracker_jumps": int(kin["tracker_jump"].sum()),
         "event_method": ev["method"].iloc[0] if len(ev) else None,
         "stride_period_est_s": round(period_s, 3) if period_s else np.nan}
    all_hs = ev[ev.event == "HS"].sort_values("frame")
    step = np.diff(all_hs["frame"].to_numpy()) / track.fps if len(all_hs) > 1 else np.array([])
    s["cadence_steps_per_min"] = round(float(60 / np.median(step)), 1) if len(step) else np.nan
    for side in SIDES:
        st = strides[(strides.side == side) & strides.valid]
        allst = strides[strides.side == side]
        knee = kin[f"{side}_knee_3d"].to_numpy()
        s[f"{side}_hs_count"] = int(((ev.side == side) & (ev.event == "HS")).sum())
        s[f"{side}_strides_valid"] = int(len(st))
        s[f"{side}_strides_rejected"] = int(len(allst) - len(st))
        s[f"{side}_stride_time_mean_s"] = round(float(st.stride_time_s.mean()), 3) if len(st) else np.nan
        s[f"{side}_stride_time_cv"] = round(float(st.stride_time_s.std() / st.stride_time_s.mean()), 3) if len(st) > 1 else np.nan
        s[f"{side}_stance_frac_mean"] = round(float(st.stance_frac.mean()), 3) if len(st) else np.nan
        s[f"{side}_knee_3d_rom_mean"] = round(float(st.knee_3d_rom.mean()), 1) if len(st) else np.nan
        s[f"{side}_hip_3d_rom_mean"] = round(float(st.hip_3d_rom.mean()), 1) if len(st) else np.nan
        s[f"{side}_ankle_3d_rom_mean"] = round(float(st.ankle_3d_rom.mean()), 1) if len(st) else np.nan
        s[f"{side}_peak_knee_flex_pct_mean"] = round(float(st.peak_knee_flex_pct.mean()), 1) if len(st) else np.nan
        s[f"{side}_cycle_consistency_r"] = round(_consistency(cycles[side][:, :, 0]), 3) if len(cycles[side]) else np.nan
        s[f"{side}_knee_flexion_peaks"] = int(len(_peaks(-knee, track.fps, 0.5 * (period_s or 1.2), 0.2)))
        if ev_alt is not None and len(ev_alt):
            s[f"{side}_hs_count_{ev_alt['method'].iloc[0]}"] = int(((ev_alt.side == side) & (ev_alt.event == "HS")).sum())
    s["stride_time_symmetry_pct"] = _si(s["L_stride_time_mean_s"], s["R_stride_time_mean_s"])
    s["knee_rom_symmetry_pct"] = _si(s["L_knee_3d_rom_mean"], s["R_knee_3d_rom_mean"])
    s["stance_symmetry_pct"] = _si(s["L_stance_frac_mean"], s["R_stance_frac_mean"])
    s["step_width_ratio_mean"] = round(float(strides[strides.valid].step_width_ratio_hs.mean()), 3) if strides.valid.any() else np.nan
    s["trunk_sway_ratio_sd"] = round(float(np.nanstd(kin["trunk_sway_ratio"])), 3)
    return s


# --------------------------------------------------------------------------
# Plots and annotated video
# --------------------------------------------------------------------------
def plot_traces(kin, ev, out_png, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = kin["t"].to_numpy()
    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
    panels = [("knee_3d", "Knee (3-D) deg"), ("hip_3d", "Hip (3-D) deg"), ("ankle_3d", "Ankle (3-D) deg")]
    colors = {"L": "tab:blue", "R": "tab:orange"}
    for ax, (key, lab) in zip(axes[:3], panels):
        for side in SIDES:
            ax.plot(t, kin[f"{side}_{key}"], color=colors[side], lw=0.9, label=f"{side}")
        ax.set_ylabel(lab)
        ax.legend(loc="upper right", fontsize=8)
    method = ev["method"].iloc[0] if len(ev) else "3d"
    for side in SIDES:
        heel, toe = event_signals(kin, side, method)
        axes[3].plot(t, heel, color=colors[side], lw=0.9, label=f"{side} heel")
        axes[3].plot(t, toe, color=colors[side], lw=0.6, ls="--", label=f"{side} toe")
        hs = ev[(ev.side == side) & (ev.event == "HS")]
        to = ev[(ev.side == side) & (ev.event == "TO")]
        axes[3].plot(hs["t"], heel[hs["frame"]], "v", color=colors[side], ms=7, label=f"{side} HS")
        axes[3].plot(to["t"], toe[to["frame"]], "^", color=colors[side], ms=6, mfc="none", label=f"{side} TO")
        for _, r in hs.iterrows():
            for ax in axes[:3]:
                ax.axvline(r.t, color=colors[side], lw=0.5, alpha=0.35)
    axes[3].set_ylabel({"3d": "anterior displacement (m)", "2d-frontal": "image height / torso",
                        "2d-sagittal": "horizontal displacement / torso"}[method])
    axes[3].legend(loc="upper right", fontsize=7, ncol=3)
    axes[3].set_xlabel("time (s)")
    fig.suptitle(title or "joint angles with detected heel strikes (v) and toe offs (^)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def plot_cycles(cycles, columns, out_png, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    grid = np.linspace(0, 100, cycles["L"].shape[1] if len(cycles["L"]) else 101)
    fig, axes = plt.subplots(1, len(columns), figsize=(5 * len(columns), 4))
    axes = np.atleast_1d(axes)
    colors = {"L": "tab:blue", "R": "tab:orange"}
    for j, (ax, c) in enumerate(zip(axes, columns)):
        for side in SIDES:
            cyc = cycles[side]
            if not len(cyc):
                continue
            m, sd = cyc[:, :, j].mean(axis=0), cyc[:, :, j].std(axis=0)
            ax.plot(grid, m, color=colors[side], label=f"{side} (n={len(cyc)})")
            ax.fill_between(grid, m - sd, m + sd, color=colors[side], alpha=0.2)
        ax.set_xlabel("% gait cycle (heel strike to heel strike)")
        ax.set_ylabel(c.replace("_", " ") + " (deg)")
        ax.legend(fontsize=8)
    fig.suptitle(title or "ensemble-averaged cycles, mean +/- sd")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


# On-screen text exactly as in Kezia's gait_analysis_mediapipe.py: plain white,
# one value per line, no outline.
FONT_SCALE = 0.7
FONT_THICKNESS = 2


def _overlay_text(img, text, org):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (255, 255, 255), FONT_THICKNESS, cv2.LINE_AA)


def write_annotated(track: Track, kin, ev, out_path, strides=None):
    """Second decode pass: full skeleton (MediaPipe's default pose style, as
    in Kezia's video), the 3-D hip/knee/ankle angles in her text layout, and
    a flash on the heel / toe at each detected heel strike / toe off."""
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.components.containers import landmark as lm_mod
    conns = vision.PoseLandmarksConnections.POSE_LANDMARKS
    style = vision.drawing_styles.get_default_pose_landmarks_style()
    cap = cv2.VideoCapture(track.path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, track.fps, (track.width, track.height))
    flash = {}
    for _, r in ev.iterrows():
        for f in range(int(r.frame) - 1, int(r.frame) + 4):
            flash.setdefault(f, []).append((r.side, r.event))
    colors = {"L": (255, 160, 0), "R": (0, 140, 255)}
    i = 0
    while i < track.n_frames:
        ok, frame = cap.read()
        if not ok:
            break
        p, v = track.px[i], track.vis[i]
        if np.isfinite(p[LM["HP"][0], 0]):
            # the drawing util hides landmarks below its own 0.5 visibility /
            # presence thresholds, which is what keeps occluded points off screen
            pts = [lm_mod.NormalizedLandmark(
                x=float(p[k, 0] / track.width) if np.isfinite(p[k, 0]) else 0.0,
                y=float(p[k, 1] / track.height) if np.isfinite(p[k, 1]) else 0.0,
                z=0.0,
                visibility=float(v[k]) if np.isfinite(v[k]) else 0.0,
                presence=1.0 if np.isfinite(p[k, 0]) else 0.0) for k in range(33)]
            vision.drawing_utils.draw_landmarks(frame, pts, conns, landmark_drawing_spec=style)
        row = kin.iloc[i]
        y0, dy = 30, 28
        if row.L_ok:
            _overlay_text(frame, f"L Hip:  {row.L_hip_3d:.1f}", (20, y0))
            _overlay_text(frame, f"L Knee: {row.L_knee_3d:.1f}", (20, y0 + dy))
            _overlay_text(frame, f"L Ankle:{row.L_ankle_3d:.1f}", (20, y0 + 2 * dy))
        else:
            _overlay_text(frame, "L angles: NA", (20, y0))
        if row.R_ok:
            _overlay_text(frame, f"R Hip:  {row.R_hip_3d:.1f}", (20, y0 + 3 * dy))
            _overlay_text(frame, f"R Knee: {row.R_knee_3d:.1f}", (20, y0 + 4 * dy))
            _overlay_text(frame, f"R Ankle:{row.R_ankle_3d:.1f}", (20, y0 + 5 * dy))
        else:
            _overlay_text(frame, "R angles: NA", (20, y0 + 3 * dy))
        for side, event in flash.get(i, []):
            k = LM["HL" if event == "HS" else "FT"][0 if side == "L" else 1]
            if np.isfinite(p[k]).all():
                cv2.circle(frame, tuple(p[k].astype(int)), 18, colors[side], 3, cv2.LINE_AA)
                cv2.putText(frame, f"{event} {side}", (int(p[k, 0]) - 30, int(p[k, 1]) - 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[side], 2, cv2.LINE_AA)
        writer.write(frame)
        i += 1
    cap.release()
    writer.release()


# --------------------------------------------------------------------------
# Whole-video driver
# --------------------------------------------------------------------------
CYCLE_COLUMNS = ["knee_3d", "hip_3d", "ankle_3d"]


def analyze_video(path, out_dir, model="heavy", min_vis=MIN_VIS, cutoff=SMOOTH_HZ, events="auto",
                  annotate=True, max_frames=None, reuse_landmarks=True, num_poses=1, progress=True):
    """Track, compute, detect, summarise and write everything for one video.
    -> summary dict.  Files land in out_dir/<video stem>/."""
    base = os.path.splitext(os.path.basename(path))[0]
    d = os.path.join(out_dir, base)
    os.makedirs(d, exist_ok=True)
    f = lambda suffix: os.path.join(d, f"{base}_{suffix}")

    cap = cv2.VideoCapture(path)
    fps, w, h = cap.get(cv2.CAP_PROP_FPS) or 30.0, int(cap.get(3)), int(cap.get(4))
    cap.release()
    lm_csv = f("landmarks.csv")
    cached = reuse_landmarks and os.path.exists(lm_csv) and not max_frames
    if cached and not cache_is_current(lm_csv):
        cached = False
        if progress:
            print(f"  {os.path.basename(lm_csv)} is from an older version (12 landmarks); re-tracking", flush=True)
    if cached:
        track = load_track(lm_csv, path, fps, w, h, model)
        if progress:
            print(f"  reusing {lm_csv} ({track.n_frames} frames)", flush=True)
    else:
        t0 = time.time()
        track = track_video(path, model=model, num_poses=num_poses, max_frames=max_frames, progress=progress)
        if progress:
            print(f"  tracked {track.n_frames} frames in {time.time()-t0:.0f}s", flush=True)
        save_track(track, lm_csv)

    kin = compute_kinematics(track, min_vis=min_vis, cutoff=cutoff)
    method = choose_method(kin, events)
    ev, period_s = detect_events(kin, track.fps, method=method)
    alt = "3d" if method != "3d" else "2d-frontal"          # second opinion on the heel-strike count
    ev_alt, _ = detect_events(kin, track.fps, method=alt, period_s=period_s)
    strides = stride_table(kin, ev, track.fps, period_s=period_s)
    cycles = {side: normalize_cycles(kin, strides, [f"{side}_{c}" for c in CYCLE_COLUMNS])[side] for side in SIDES}
    summary = summarize(track, kin, ev, strides, cycles, ev_alt, period_s=period_s)

    kin.to_csv(f("angles.csv"), index=False, float_format="%.4f")
    ev.to_csv(f("events.csv"), index=False)
    strides.to_csv(f("strides.csv"), index=False, float_format="%.4f")
    with open(f("summary.json"), "w") as fh:
        json.dump({k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in summary.items()}, fh, indent=2)
    grid = np.linspace(0, 100, 101)
    rows = []
    for side in SIDES:
        cyc = cycles[side]
        for j, c in enumerate(CYCLE_COLUMNS):
            if len(cyc):
                rows.append(pd.DataFrame({"side": side, "angle": c, "pct": grid,
                                          "mean": cyc[:, :, j].mean(axis=0), "sd": cyc[:, :, j].std(axis=0),
                                          "n": len(cyc)}))
    (pd.concat(rows) if rows else pd.DataFrame(columns=["side", "angle", "pct", "mean", "sd", "n"])).to_csv(
        f("cycles.csv"), index=False, float_format="%.3f")
    plot_traces(kin, ev, f("angles.png"), title=f"{base}: view={summary['view']}, events={method}, "
                                                 f"stride period ~{period_s:.1f}s")
    plot_cycles(cycles, CYCLE_COLUMNS, f("cycles.png"), title=f"{base}: ensemble cycles (valid strides only)")
    if annotate:
        t0 = time.time()
        write_annotated(track, kin, ev, f("annotated.mp4"))
        if progress:
            print(f"  annotated video written in {time.time()-t0:.0f}s", flush=True)
    return summary
