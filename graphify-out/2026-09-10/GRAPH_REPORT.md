# Graph Report - Thaddeus RESTORES EMG Test  (2026-09-10)

## Corpus Check
- 27 files · ~44,171 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 246 nodes · 327 edges · 19 communities (12 shown, 7 thin omitted)
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `921fab14`
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
- README.md
- process_trial

## God Nodes (most connected - your core abstractions)
1. `Core Capabilities` - 12 edges
2. `What You Must Do When Invoked` - 12 edges
3. `Core Capabilities` - 12 edges
4. `qc_session()` - 11 edges
5. `/graphify` - 10 edges
6. `process_arrays()` - 9 edges
7. `apply_filters_P3()` - 9 edges
8. `NeuroKit2` - 9 edges
9. `NeuroKit2` - 9 edges
10. `filter_baseline()` - 8 edges

## Surprising Connections (you probably didn't know these)
- `apply_filters_P3()` --calls--> `clean()`  [EXTRACTED]
  preprocessing_functions.py → emg_pipeline.py
- `qc_session()` --calls--> `is_environmental()`  [EXTRACTED]
  run_csv_qc.py → emg_pipeline.py
- `qc_session()` --calls--> `pulse_train_consensus()`  [EXTRACTED]
  run_csv_qc.py → emg_pipeline.py
- `qc_session()` --calls--> `remove_artifact()`  [EXTRACTED]
  run_csv_qc.py → emg_pipeline.py
- `qc_session()` --calls--> `detect_activation()`  [EXTRACTED]
  run_csv_qc.py → emg_pipeline.py

## Import Cycles
- None detected.

## Communities (19 total, 7 thin omitted)

### Community 0 - "restores_csv.py"
Cohesion: 0.10
Nodes (24): agonists(), baseline_for(), _classify(), _clock(), list_session(), load(), parse_preamble(), -> (meta dict, line index of the column header). (+16 more)

### Community 1 - "emg_pipeline.py"
Cohesion: 0.06
Nodes (43): artifact_prominence(), blank_pulses(), build_filter(), clean(), detect_activation(), detect_pulse_train(), detect_stim_comb(), detect_stim_frequency() (+35 more)

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

### Community 18 - "process_trial"
Cohesion: 0.18
Nodes (14): check_duplicate_recordings(), find_baseline(), find_trials(), load_concatenated(), load_recording(), parse_name(), process_trial(), -> (exercise_number, subtrial or None), or (None, None) if unnumbered. (+6 more)

## Knowledge Gaps
- **84 isolated node(s):** `Overview`, `When to Use This Skill`, `1. Cardiac Signal Processing (ECG/PPG)`, `2. Heart Rate Variability Analysis`, `3. Brain Signal Analysis (EEG)` (+79 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 155 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **7 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `clean()` connect `emg_pipeline.py` to `preprocessing_functions.py`?**
  _High betweenness centrality (0.119) - this node is a cross-community bridge._
- **Why does `apply_filters_P3()` connect `preprocessing_functions.py` to `emg_pipeline.py`?**
  _High betweenness centrality (0.046) - this node is a cross-community bridge._
- **Why does `qc_session()` connect `restores_csv.py` to `emg_pipeline.py`?**
  _High betweenness centrality (0.042) - this node is a cross-community bridge._
- **What connects `Overview`, `When to Use This Skill`, `1. Cardiac Signal Processing (ECG/PPG)` to the rest of the system?**
  _84 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `restores_csv.py` be split into smaller, more focused modules?**
  _Cohesion score 0.09885057471264368 - nodes in this community are weakly interconnected._
- **Should `emg_pipeline.py` be split into smaller, more focused modules?**
  _Cohesion score 0.05585106382978723 - nodes in this community are weakly interconnected._
- **Should `preprocessing_functions.py` be split into smaller, more focused modules?**
  _Cohesion score 0.1251778093883357 - nodes in this community are weakly interconnected._