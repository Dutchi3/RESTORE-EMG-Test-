# Graph Report - Thaddeus RESTORES EMG Test  (2026-09-15)

## Corpus Check
- 30 files · ~50,807 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 317 nodes · 454 edges · 25 communities (19 shown, 6 thin omitted)
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `2b375de6`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- restores_csv.py
- emg_pipeline.py
- preprocessing_functions.py
- What You Must Do When Invoked
- Core Capabilities
- Core Capabilities
- graphify reference: extra exports and benchmark
- graphify reference: query, path, explain
- graphify reference: add a URL and watch a folder
- graphify reference: commit hook and native CLAUDE.md integration
- graphify reference: incremental update and cluster-only
- graphify reference: GitHub clone and cross-repo merge
- graphify reference: transcribe video and audio
- Clinical Scores code.py
- CLAUDE.md
- .claude/CLAUDE.md
- extraction-spec.md
- RESTORES EMG / gait analysis
- process_trial
- gait_video.py
- process_arrays
- measure_stim
- clean
- artifact_prominence
- run_gait_video.py

## God Nodes (most connected - your core abstractions)
1. `analyze_video()` - 15 edges
2. `Core Capabilities` - 12 edges
3. `What You Must Do When Invoked` - 12 edges
4. `Core Capabilities` - 12 edges
5. `qc_session()` - 11 edges
6. `detect_events()` - 10 edges
7. `/graphify` - 10 edges
8. `process_arrays()` - 9 edges
9. `Track` - 9 edges
10. `apply_filters_P3()` - 9 edges

## Surprising Connections (you probably didn't know these)
- `apply_filters_P3()` --calls--> `clean()`  [EXTRACTED]
  preprocessing_functions.py → emg_pipeline.py
- `analyze_file()` --calls--> `moving_rms()`  [EXTRACTED]
  run_walking_emg.py → emg_pipeline.py
- `qc_session()` --calls--> `pulse_train_consensus()`  [EXTRACTED]
  run_csv_qc.py → emg_pipeline.py
- `stim_rate()` --calls--> `pulse_train_consensus()`  [EXTRACTED]
  run_walking_emg.py → emg_pipeline.py
- `qc_session()` --calls--> `remove_artifact()`  [EXTRACTED]
  run_csv_qc.py → emg_pipeline.py

## Import Cycles
- None detected.

## Communities (25 total, 6 thin omitted)

### Community 0 - "restores_csv.py"
Cohesion: 0.08
Nodes (33): is_environmental(), True for a spectral comb sitting on the ENV_COMB_HZ ladder. A spike train…, agonists(), baseline_for(), _classify(), _clock(), list_session(), load() (+25 more)

### Community 1 - "emg_pipeline.py"
Cohesion: 0.17
Nodes (9): detect_stim_frequency(), Patient, Per-patient recording conventions., Find the stimulation line in a recording, or None if there isn't one. STIM_HZ…, Single-path EMG processing for the RESTORES study (P1-P3). Replaces the six…, Windowed RMS. window_s=0.05 matches the original's 500-sample window (its…, Finding 17: max() is a single sample and one artifact spike rescales a whole…, rms_envelope() (+1 more)

### Community 2 - "preprocessing_functions.py"
Cohesion: 0.13
Nodes (29): Created on Mon Mar 9 11:58:36 2026 @author: David Teo, RMS(), RMS_envelope(), Created on Wed Apr 1 16:44:56 2026 @author: David Teo, RMS(), RMS_envelope(), Created on Wed Mar 11 17:32:02 2026 @author: David Teo, RMS() (+21 more)

### Community 3 - "What You Must Do When Invoked"
Cohesion: 0.08
Nodes (24): For /graphify add and --watch, For /graphify query, For the commit hook and native CLAUDE.md integration, For --update and --cluster-only, /graphify, Honesty Rules, Interpreter guard for subcommands, Part A - Structural extraction for code files (+16 more)

### Community 4 - "Core Capabilities"
Cohesion: 0.08
Nodes (23): 10. Event-Related Analysis, 11. Multi-Signal Integration, 1. Cardiac Signal Processing (ECG/PPG), 2. Heart Rate Variability Analysis, 3. Brain Signal Analysis (EEG), 4. Electrodermal Activity (EDA), 5. Respiratory Signal Processing (RSP), 6. Electromyography (EMG) (+15 more)

### Community 5 - "Core Capabilities"
Cohesion: 0.08
Nodes (23): 10. Event-Related Analysis, 11. Multi-Signal Integration, 1. Cardiac Signal Processing (ECG/PPG), 2. Heart Rate Variability Analysis, 3. Brain Signal Analysis (EEG), 4. Electrodermal Activity (EDA), 5. Respiratory Signal Processing (RSP), 6. Electromyography (EMG) (+15 more)

### Community 6 - "graphify reference: extra exports and benchmark"
Cohesion: 0.22
Nodes (8): graphify reference: extra exports and benchmark, Step 6b - Wiki (only if --wiki flag), Step 7 - Neo4j export (only if --neo4j or --neo4j-push flag), Step 7a - FalkorDB export (only if --falkordb or --falkordb-push flag), Step 7b - SVG export (only if --svg flag), Step 7c - GraphML export (only if --graphml flag), Step 7d - MCP server (only if --mcp flag), Step 8 - Token reduction benchmark (only if total_words > 5000)

### Community 7 - "graphify reference: query, path, explain"
Cohesion: 0.33
Nodes (5): For /graphify explain, For /graphify path, graphify reference: query, path, explain, Step 0 — Constrained query expansion (REQUIRED before traversal), Step 1 — Traversal

### Community 8 - "graphify reference: add a URL and watch a folder"
Cohesion: 0.50
Nodes (3): For /graphify add, For --watch, graphify reference: add a URL and watch a folder

### Community 9 - "graphify reference: commit hook and native CLAUDE.md integration"
Cohesion: 0.50
Nodes (3): For git commit hook, For native CLAUDE.md integration, graphify reference: commit hook and native CLAUDE.md integration

### Community 10 - "graphify reference: incremental update and cluster-only"
Cohesion: 0.50
Nodes (3): For --cluster-only, For --update (incremental re-extraction), graphify reference: incremental update and cluster-only

### Community 17 - "RESTORES EMG / gait analysis"
Cohesion: 0.40
Nodes (4): EMG (P4-P6 CSV export), RESTORES EMG / gait analysis, Setup, Walking videos (gait kinematics)

### Community 18 - "process_trial"
Cohesion: 0.18
Nodes (14): check_duplicate_recordings(), find_baseline(), find_trials(), load_concatenated(), load_recording(), parse_name(), process_trial(), -> (exercise_number, subtrial or None), or (None, None) if unnumbered. (+6 more)

### Community 19 - "gait_video.py"
Cohesion: 0.06
Nodes (53): analyze_video(), choose_method(), compute_kinematics(), _consistency(), detect_events(), dominant_period(), _end_of_rise(), event_signals() (+45 more)

### Community 20 - "process_arrays"
Cohesion: 0.18
Nodes (9): detect_activation(), gain_scale(), moving_rms(), process_arrays(), Same-length RMS envelope., Samples where the muscle is active. The threshold is derived from the *rest*…, Divide out the amplifier range so amplitudes are comparable across conditions…, Process one exercise given its (n_ch, N) array and its rest recording. Format-… (+1 more)

### Community 21 - "measure_stim"
Cohesion: 0.20
Nodes (10): detect_pulse_train(), detect_stim_comb(), measure_stim(), pulse_train_consensus(), Sample indices of stimulation spikes. -> (indices, |dx|, settled-slope level) A…, -> (pulse rate in Hz or None, consistency). Measured from the spacing of…, Fundamental of a harmonic comb in the spectrum, or None. -> (f0, mean dB)…, -> (rate or None, method) where method is 'pulses', 'comb' or None. (+2 more)

### Community 22 - "clean"
Cohesion: 0.22
Nodes (9): blank_pulses(), build_filter(), clean(), _notch_sections(), Bandpass plus stimulation and mains harmonics, as one second-order-section…, Filter one channel. remove_stim=False reproduces the original behaviour.…, Cut each stimulation spike out of the trace and bridge the gap. -> (blanked…, Blank the pulses (if a rate was found), then bandpass + notches. stim_hz=None… (+1 more)

### Community 23 - "artifact_prominence"
Cohesion: 0.50
Nodes (4): artifact_prominence(), qc_row(), Height of the stimulation line above local spectral background, in dB., Per-trial quality metrics. The original emitted nothing like this, so a 26x…

### Community 24 - "run_gait_video.py"
Cohesion: 0.67
Nodes (3): find_videos(), main(), Gait kinematics from the RESTORES walking videos: one folder of outputs per…

## Knowledge Gaps
- **86 isolated node(s):** `Overview`, `When to Use This Skill`, `1. Cardiac Signal Processing (ECG/PPG)`, `2. Heart Rate Variability Analysis`, `3. Brain Signal Analysis (EEG)` (+81 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 185 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **6 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `clean()` connect `clean` to `emg_pipeline.py`, `preprocessing_functions.py`?**
  _High betweenness centrality (0.078) - this node is a cross-community bridge._
- **Why does `apply_filters_P3()` connect `preprocessing_functions.py` to `clean`?**
  _High betweenness centrality (0.030) - this node is a cross-community bridge._
- **Why does `qc_session()` connect `restores_csv.py` to `process_arrays`, `measure_stim`, `clean`?**
  _High betweenness centrality (0.022) - this node is a cross-community bridge._
- **What connects `Overview`, `When to Use This Skill`, `1. Cardiac Signal Processing (ECG/PPG)` to the rest of the system?**
  _86 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `restores_csv.py` be split into smaller, more focused modules?**
  _Cohesion score 0.07948717948717948 - nodes in this community are weakly interconnected._
- **Should `preprocessing_functions.py` be split into smaller, more focused modules?**
  _Cohesion score 0.1251778093883357 - nodes in this community are weakly interconnected._
- **Should `What You Must Do When Invoked` be split into smaller, more focused modules?**
  _Cohesion score 0.08 - nodes in this community are weakly interconnected._