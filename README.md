# RESTORES EMG / gait analysis

Analysis code for the RESTORES study (epidural spinal cord stimulation in
spinal-cord-injury patients): surface EMG during physiotherapy exercises,
and walking videos.

## Setup

    python -m venv venv
    venv\Scripts\python -m pip install -r requirements.txt

Create `restores_config.py` (untracked) pointing at the data share:

    ROOT = r"\\neuroserver\RESTORES"

## EMG (P4-P6 CSV export)

* `emg_pipeline.py` - shared processing: stimulation-rate detection, artifact
  removal, filtering, activation detection, per-trial QC.
* `restores_csv.py` - loader for the CSV export (P4-P6); resolves baselines and
  stim condition by file name, maps exercise time tabs to samples, recognises
  walking files (`gaits`, `harness`).
* `run_csv_qc.py` - one QC row per recording:

      venv\Scripts\python run_csv_qc.py P4 --dates 2026-01-28 --max-seconds 60

Legacy P1-P3 (.mat) scripts from the original hand-over remain alongside.

## Walking videos (gait kinematics)

* `gait_video.py` - MediaPipe pose tracking, joint angles (2-D pixel, 3-D
  world, legacy normalised), gait events, stride table, ensemble cycles,
  summary metrics.  Read its module docstring before interpreting numbers:
  the RESTORES videos are frontal, so flexion angles come from the 3-D
  landmarks and the strong measures are timing, symmetry, step width and
  trunk sway.
* `run_gait_video.py` - batch driver:

      venv\Scripts\python run_gait_video.py                 # all videos in <share>\Kezia_Data\RESTORES_Walk
      venv\Scripts\python run_gait_video.py some.mp4 --no-annotate
      venv\Scripts\python run_gait_video.py --events 3d     # after a first run: reuses cached landmarks

  Outputs go to `outputs/gait/<video>/` (landmarks, angles, events, strides,
  cycles, plots, annotated video) plus `outputs/gait/gait_summary.csv`.
  The pose model (~30 MB) is downloaded to `models/` on first use.

The original video script from Kezia Susanto is on the share
(`Kezia_Data\RESTORES_Walk\gait_analysis_mediapipe.py`); it needs the legacy
`mp.solutions` API that mediapipe >= 1.0 no longer ships, which is why it was
ported rather than copied.
