# -*- coding: utf-8 -*-
"""
Gait kinematics from the RESTORES walking videos: one folder of outputs per video.

    python run_gait_video.py                       # every .mp4 in <share>\\Kezia_Data\\RESTORES_Walk
    python run_gait_video.py path\\to\\video.mp4 [more.mp4 | a_folder ...]
    python run_gait_video.py --model full --no-annotate --events 2d --out outputs\\gait

Per video, in outputs/gait/<stem>/:
    <stem>_landmarks.csv   raw per-frame landmarks (pixels, visibility, world metres)
    <stem>_angles.csv      per-frame joint angles (2-D pixel, 3-D world, legacy normalised),
                           view ratio, step width, trunk sway, event signals
    <stem>_events.csv      heel strikes / toe offs
    <stem>_strides.csv     one row per stride with validity and reject reason
    <stem>_cycles.csv/.png ensemble-averaged 0-100 % cycles (valid strides only)
    <stem>_angles.png      traces with events marked
    <stem>_summary.json    the numbers that go in a table
    <stem>_annotated.mp4   skeleton + angles + event flashes (skip with --no-annotate)
and one combined outputs/gait/gait_summary.csv.

Tracking is the slow part (~8 fps for the heavy model on CPU).  Landmarks
are cached in <stem>_landmarks.csv and reused on the next run unless
--retrack is given, so analysis parameters can be changed cheaply.
"""
import argparse
import os
import sys

import pandas as pd

import gait_video as gv


def find_videos(items):
    out = []
    for it in items:
        if os.path.isdir(it):
            out += sorted(os.path.join(it, f) for f in os.listdir(it) if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv")))
        elif os.path.isfile(it):
            out.append(it)
        else:
            print(f"!! not found: {it}", file=sys.stderr)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="*", help="video files or folders (default: the RESTORES_Walk share folder)")
    ap.add_argument("--out", default=os.path.join("outputs", "gait"))
    ap.add_argument("--model", default="heavy", choices=list(gv.MODEL_URLS))
    ap.add_argument("--events", default="auto", choices=["auto", "3d", "2d-frontal", "2d-sagittal"],
                    help="gait-event method (see gait_video.detect_events)")
    ap.add_argument("--min-vis", type=float, default=gv.MIN_VIS, help="landmark visibility needed for a frame to count")
    ap.add_argument("--smooth-hz", type=float, default=gv.SMOOTH_HZ, help="low-pass cut-off for landmark trajectories")
    ap.add_argument("--num-poses", type=int, default=1, help="poses the detector may return; the patient is chosen by continuity")
    ap.add_argument("--max-frames", type=int, default=None, help="only the first N frames (testing)")
    ap.add_argument("--no-annotate", action="store_true", help="skip the annotated video (fast)")
    ap.add_argument("--retrack", action="store_true", help="ignore cached landmarks and run the pose model again")
    args = ap.parse_args(argv)

    items = args.videos
    if not items:
        import emg_pipeline as ep
        if not ep.ROOT:
            sys.exit("no videos given and RESTORES_ROOT / restores_config.py is not set")
        items = [os.path.join(ep.ROOT, "Kezia_Data", "RESTORES_Walk")]
    videos = find_videos(items)
    if not videos:
        sys.exit("no videos found")

    rows = []
    for v in videos:
        print(f"--- {v}", flush=True)
        try:
            s = gv.analyze_video(v, args.out, model=args.model, min_vis=args.min_vis, cutoff=args.smooth_hz,
                                 events=args.events, annotate=not args.no_annotate, max_frames=args.max_frames,
                                 reuse_landmarks=not args.retrack, num_poses=args.num_poses)
        except Exception as e:                       # report, never silently skip
            print(f"    !! {type(e).__name__}: {e}", flush=True)
            rows.append({"video": os.path.basename(v), "error": f"{type(e).__name__}: {e}"})
            continue
        print(f"    view={s['view']} detected={s['detected_frac']:.2f} "
              f"HS L/R={s['L_hs_count']}/{s['R_hs_count']} valid strides L/R={s['L_strides_valid']}/{s['R_strides_valid']} "
              f"stride {s['L_stride_time_mean_s']}/{s['R_stride_time_mean_s']} s  cadence={s['cadence_steps_per_min']} steps/min "
              f"knee ROM L/R={s['L_knee_3d_rom_mean']}/{s['R_knee_3d_rom_mean']} deg", flush=True)
        rows.append(s)
    os.makedirs(args.out, exist_ok=True)
    out_csv = os.path.join(args.out, "gait_summary.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nwrote {len(rows)} rows -> {out_csv}")


if __name__ == "__main__":
    main()
