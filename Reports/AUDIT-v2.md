# AUDIT v2 — End-to-End Pipeline & Dashboard Audit

Date: 2026-08-30
Branch: `adi-drafts` @ `3c1b171`
Scope: grid engine module (`grid.py`, `aggregate.py`, `handoff.py`,
`radial_filter.py`, `grid_state.py`, `profiler.py`), synthetic data/test
harness, both Streamlit dashboards (`dashboard_pro.py`,
`dashboard_driving.py`), and repo hygiene.

This is an **independent re-verification from the current working tree**,
not a continuation of `Reports/AUDIT.md`. Every finding below was
reproduced by reading current source, running the test scripts, running
targeted throwaway checks, or driving the live dashboards — nothing here is
carried over from AUDIT.md's prose without being re-checked against the
code on disk today.

---

## 0. Headline finding: AUDIT.md Round 2's "fixes" are not in the tree

`Reports/AUDIT.md` §10 ("Round 2 — Fixes Implemented & Re-Test Results",
dated same day) claims eight fixes landed and were re-tested with specific
numbers (e.g. "32-58ms/frame" after a Numba pass). **None of them are
present in the current source on `adi-drafts`.** Confirmed by reading every
core module in full and grepping the whole repo:

| Round 2 claim | Current reality | Evidence |
|---|---|---|
| `validate_labels()` added to `aggregate.py` | Absent | `grep -r validate_labels` → zero hits repo-wide |
| `validate_inputs()` added to `handoff.py` | Absent | `grep -r validate_inputs` → zero hits repo-wide; `generate_2_5d_grid()` does no validation at all |
| Numba `@njit`-compiled sub-cell indexing in `grid.py` (packed-int64 keys) | Absent — `grid.py` is pure NumPy, no `numba`/`njit` import | `grep -r "njit\|import numba"` → zero hits in any `.py` file |
| Frame time down to 32-58ms via the Numba pass | Not reproducible — `test_integration.py` measured 69-140ms/frame today, 5/10 frames still over the 100ms budget | see §4 |
| Majority-class tie-break reversed to favor higher class ID | Still plain `np.argmax` (lowest class ID wins ties) | `aggregate.py:80`; reproduced live, see §3.3 |
| `radial_filter.py` importing `INNER_RADIUS`/`OUTER_RADIUS` from `grid.py` | Still hardcodes `10.0`/`100.0` as literals | `radial_filter.py:26-27,37` |

**Forensic corroboration that this isn't a misread of AUDIT.md**: the
tracked `__pycache__/` directory (itself a hygiene problem, see §7) still
contains `grid._sub_index_loop-17.py314.1.nbc` / `.nbi` — Numba's on-disk
compilation cache for a function named `_sub_index_loop`. That function
does not exist anywhere in current `grid.py`. This is direct physical
evidence the Numba-JIT version of `grid.py` existed and ran at some point,
and the source has since reverted or was never committed on this branch —
this is a **lost-work regression**, not a documentation error.

Likely mechanism: `.gitignore` excludes every `*.md` except `/README.md`
(see §7.4), so `Reports/AUDIT.md` itself is untracked. A session that
implemented and validated these fixes could have had its report survive
(local file, never wiped) while the corresponding code changes were on a
branch/stash/worktree that got reset, never committed, or committed then
reverted by a later merge (`3c1b171` merges `origin/main` into
`adi-drafts` — a plausible point where these changes could have been
dropped if they only existed in one parent).

**Action needed**: before trusting AUDIT.md's Round 2 or Round 3 status for
planning purposes, treat every "resolved" claim there as unverified until
re-confirmed against current source, the way this report did.

---

## 1. Interface contract compliance (CLAUDE.md §3)

| Requirement | Status | Notes |
|---|---|---|
| `points` (N,4) float32 X,Y,Z,Intensity | Compliant | `synthetic_lidar_data.py` generators all cast to `float32`; `radial_filter`/`grid.assign_cells` index `[:, 0:2]`/`[:, 2]` correctly |
| `labels` (N,) int64/uint8, fixed 0/1/2 mapping | Compliant in generators; **not enforced** at ingestion | No range check anywhere — see §3.2 for the crash this causes |
| `spikes` = count not flag, event logic must check `>0` | Correctly implemented as `>0` throughout (`grid_state.py:62`) | But dtype is **not enforced** — `handoff.generate_2_5d_grid()` silently accepts a `float32` spikes array with non-integer values (e.g. `2.7`) with no error, no cast, no warning (verified live, §3.4). Real SNN output (`spk_rec.sum()` over timesteps) is exactly this shape/dtype, so this isn't a hypothetical input. |
| Ego-centric frame, UGV always at origin | Compliant | `grid.py`, `radial_filter.py` never reference a world frame; `driving_sequence.py` explicitly transforms world→ego before returning points |
| Class ID mapping fixed (0/1/2) | Compliant in code that respects it | Nothing stops an out-of-range label from reaching `aggregate_cells` and crashing (§3.2) |
| Intensity dropped after ingestion, Z not carried per-point downstream | Compliant | `aggregate_cells` only reads `points[:,2]` for elevation stats, never re-emits raw points |

---

## 2. Core module summary (read in full, current source)

- **`grid.py`** — Base 50cm grid over ±100m; per-parent-cell subdivision
  into a 10×10 array of 5cm sub-cells, decided once per unique parent cell
  via `np.unique` (never per-point, never per-point-distance) — matches
  the CLAUDE.md design exactly. Fully vectorized. No Numba present (see §0).
- **`aggregate.py`** — `elevation_max` via `np.maximum.at`; `elevation_var`
  via bincount sum/sum-of-squares (population variance); majority class via
  `np.bincount` + `np.argmax` (ties → lowest class ID); `spike_sum` via
  weighted `bincount`. Fully vectorized, no Python point loop, per CLAUDE.md
  §4. `NUM_CLASSES=3` hardcoded — this is the direct cause of the
  out-of-range-label crash (§3.2).
- **`handoff.py`** — `generate_2_5d_grid(points, labels, spikes)` delegates
  to a module-level `GridState` singleton; `memory_metrics()` compares
  sparse-cell bytes to a naive dense-voxel estimate. No input validation of
  any kind.
- **`radial_filter.py`** — `prefilter_mask()` (the actual gate,
  `r <= 100.0`) and `radial_filter()` (diagnostic-only inner/outer masks).
  Both hardcode `10.0`/`100.0` rather than importing `grid.INNER_RADIUS`/
  `OUTER_RADIUS` — two independent sources of truth for the same physical
  constants, a latent maintenance hazard even though the values currently
  agree.
- **`grid_state.py`** — Dict-keyed cache (`(parent_ix,parent_iy,sub_ix,
  sub_iy) -> CellRecord`), commits a cell when `spike_sum > 0` OR it's
  never been seen (cold-start baseline). **No eviction policy** — the cache
  grows monotonically for the life of the process (confirmed live: grid
  cell count climbed from 14,689 → 27,371 → 33,247+ across 5 frames of
  `dashboard_driving.py`, see §5).
- **`profiler.py`** — `EdgeProfiler.evaluate_efficiency()` computes FPS,
  sparsity, AC/MAC op counts and energy-saved estimate from fixed
  4.6pJ/0.9pJ constants; unrelated to this module's own runtime, only used
  by the SNN/dashboard side.
- **`synthetic_lidar_data.py`** — RNG-sharing bug from earlier audit rounds
  is fixed at the repo root: every generator takes `rng=None` and creates
  its own local `np.random.default_rng(42)`. Confirmed this fix is real
  and present (unlike the Round 2 claims in §0).

---

## 3. Correctness findings (reproduced this session)

### 3.1 Boundary stress ring — PASS, no seam

Ran `test_grid.py` directly: `generate_boundary_stress_ring()` (400 points
jittered ±2cm around r=10m) — every subdivided point's sub-cell falls
exactly within its parent cell's bounds (tolerance 1e-6), 250/400 points
correctly routed fine, 150/400 correctly routed coarse, using the
per-parent-cell-center rule (never per-point). **This is the single most
important check in the module and it holds.**

### 3.2 Out-of-range class ID crashes `aggregate_cells` — CONFIRMED, still live

```
labels = [99]  (single point)
-> ValueError: cannot reshape array of size 100 into shape (1,3)
```
`class_flat_counts = np.bincount(group_id*NUM_CLASSES + lbl, minlength=n_cells*NUM_CLASSES)`
silently produces an oversized array when `lbl >= NUM_CLASSES` (bincount
sizes itself to `max(index)+1`, ignoring `minlength` when it's smaller),
and the subsequent `.reshape(n_cells, NUM_CLASSES)` throws an opaque
`ValueError` with no indication the real cause was a bad label. This was
first found in AUDIT.md Round 1 and is **still exactly reproducible today**
— it was never fixed, despite Round 2 claiming otherwise (§0).

### 3.3 Tie-break still favors lowest class ID — CONFIRMED, contradicts AUDIT.md Round 2

Constructed a cell with one point labeled class 0 and one labeled class 2
(equal counts): `np.argmax([1,0,1])` returns index 0. **Class 0 (drivable)
wins over class 2 (dynamic object) on a tie.** For a perception system
whose whole point is not missing dynamic obstacles, silently defaulting
ambiguous cells to "drivable terrain" is the least safe direction to break
ties in, and Round 2's claim to have reversed this is not reflected in the
tree.

### 3.4 NaN Z silently corrupts cell stats — CONFIRMED, still live

A single point with `Z=NaN` in a cell containing an otherwise-normal point
produces `elevation_max=nan`, `elevation_var=nan` for that cell, with no
error and no filtering — `np.maximum.at`/bincount both propagate NaN
uncontained. Per CLAUDE.md, Member 1 owns the "Z must never be NaN"
guarantee, but this module has zero defense if that contract is ever
violated for one frame, and a single corrupted cell wouldn't be visually
obvious in a dashboard.

### 3.5 Non-integer/float `spikes` accepted silently — new finding this round

`generate_2_5d_grid(points, labels, spikes=np.array([2.7], dtype=float32))`
runs with no error and no cast/round/validation. The interface contract
says `spikes` should be `uint8`/`int32` (a count), but the real producer
(`spk_rec.sum(dim=0)` from `SpikingPointNet`, as wired in both dashboards)
naturally emits float tensors converted with `.numpy()`, not integers. This
isn't a stress-test edge case — it's the actual shape of data coming from
the SNN side in production usage, and the module accepts it without
complaint. Low severity in practice (`>0` comparison still works correctly
for any positive float), but it's a silent contract violation, not a
handled one.

### 3.6 Event-driven update gating — PASS

`test_grid_state.py`: across a 10-frame moving-cluster sequence, ~496-576
cells change per frame out of ~21,000 touched, consistently comprising the
~50-60 moving-cluster cells plus the fixed pole's static cells captured
once — everything else in the ~20,000-cell ground plane stays cached
without re-triggering, exactly matching the "cells outside a moving
cluster shouldn't refresh" acceptance criterion in CLAUDE.md §6.

---

## 4. Test suite & performance

None of the `test_*.py` files are pytest-style (`def test_...`); they are
runnable scripts with top-level asserts, meant to be invoked directly
(`python test_grid.py`), which is why `pytest` collects 0 items against
them — this is a naming-convention trap for anyone assuming pytest
discovery works here, worth a docstring note if pytest is ever adopted.

Running each directly (`py -3 test_X.py`): **all 6 pass, zero assertion
failures, zero exceptions.**

Timing (`test_integration.py`, grid-engine-only, no SNN inference in the
loop):

| Frame | ms | vs 100ms budget |
|---|---|---|
| 0 | 109.3 | over |
| 1 | 139.8 | over |
| 2 | 134.9 | over |
| 3 | 77.5 | under |
| 4 | 77.3 | under |
| 5 | 71.3 | under |
| 6 | 70.4 | under |
| 7 | 80.5 | under |
| 8 | 69.1 | under |
| 9 | 107.4 | over |

5/10 frames breach the 100ms (10fps) budget mentioned in AUDIT.md's
earlier rounds — this is the grid engine alone, with `numba` installed
(v0.65.0) but unused. Since Round 2's Numba pass isn't in the tree (§0),
this is expected, not a new regression — but it means the performance
story currently told to Member 5 (MAC-vs-AC benchmarking) is not backed by
what's actually running.

Dependencies actually importable in this environment: `numba 0.65.0`,
`streamlit 1.56.0`, `torch 2.13.0+cu126`, `snntorch 1.0.0`, `sklearn
1.8.0`, `scipy 1.17.0`, `plotly 6.6.0`, `pandas 3.0.0` — all present, so
the missing Numba usage is a code gap, not a missing-dependency problem.

---

## 5. Dashboard audit (live, via Chrome automation)

Both dashboards were launched locally (`streamlit run dashboard_pro.py
--server.port 8501`, `streamlit run dashboard_driving.py --server.port
8502`) and driven through the browser this session.

### 5.1 `dashboard_pro.py` (single-frame snapshot)

- Loads cleanly, ~8-10s cold start, checkpoint loads onto `cuda`, no
  console errors observed (empty console log both before and after
  interaction).
- **Run-to-run instability confirmed live**: two consecutive "Generate
  Scene" clicks (same deterministic-by-default synthetic scene) produced
  materially different results — active cells 7,699 → 7,713; Static
  Obstacle count 6 → 7; Dynamic Threat count 437 → 444; detected-object
  distances changed entirely; **SNN inference latency swung 497ms → 66ms**
  and **grid-engine latency swung 53ms → 194ms** between the two identical
  runs. The sidebar itself displays the caveat "Untrained weights —
  predictions will collapse toward one class," which is honest framing,
  but the latency swing (not just class-prediction swing) is a separate,
  unexplained instability — the grid engine's own 53ms→194ms range on
  supposedly-identical inputs suggests either JIT/cache warm-up effects or
  non-deterministic input sampling not visible from the UI.
- Static Obstacle count (6-7 out of ~8,192 sampled points, against a scene
  built from 2 poles) is implausibly low — consistent with the untrained
  model collapsing predictions toward one class, as the UI itself warns.
- Confirmed dead `sys.path.append('AVRLM')` at line 11 has no effect
  (Streamlit runs from repo root already) — cosmetic, not a functional bug.

### 5.2 `dashboard_driving.py` (animated multi-frame sequence)

- Loads cleanly; a debug info box directly on the page reads `Loaded:
  C:\Users\Aditya\Downloads\code\AVRLM\snn_weights.pth` — this leaks the
  full local filesystem path into the UI itself (not just server logs),
  which is a minor but real information-hygiene issue for anything meant
  to look like a demo/product surface rather than a debug console.
- **Grid cell count grows unbounded and fast**: 14,689 → 27,371 → 33,247+
  cells across the first 5 animated frames, consistent with `GridState`
  having no eviction policy (§2) — for a "40 frame" sequence (the default
  slider value) this will keep growing for the whole run with no cap,
  which will matter for any long demo session or the memory-savings story
  told to Member 5.
- **Object over-detection**: by frame 5/40, "Objects: 254" is reported
  against a scripted scenario containing exactly 2 dynamic actors (one
  pedestrian, one overtaking car) plus 5 static poles. Even with "Filter
  noisy detections" checked, this is two orders of magnitude off the
  ground truth, driven by the untrained SNN's spike/class output being
  fragmented across many small DBSCAN clusters rather than 2 coherent
  blobs — a correctness issue in the demo narrative even though it's not
  strictly this module's grid-engine code.
- **Frame-counter inconsistency observed live**: the top-right HUD read
  "Frame 5/40" while the bottom-left canvas caption simultaneously read
  "Frame 3 | 27371 cells" — two different frame counters on the same
  screen disagreeing by 2, a real UI bug independent of the WebSocket
  issue below.
- **WebSocket reconnect resets the whole run — reproduced live, matches
  AUDIT.md Round 3's finding exactly**: mid-sequence (~frame 5), the
  browser tab showed "CONNECTING" in the Streamlit status area; on
  reconnect the entire page reverted to the initial idle "Start Driving
  Sequence" state — frame counter, accumulated grid, and detected-object
  list all lost. `dashboard_driving.py` does not use `st.session_state` to
  persist sequence position/results, so any transient reconnect (network
  blip, tab backgrounding, Streamlit's own periodic reconnects) during a
  demo throws away all progress with no warning to the operator. This is
  the single most disruptive dashboard bug for a live demo context and was
  directly observed, not inferred.

---

## 6. What's genuinely solid

To avoid the report reading as all-negative: the parts of CLAUDE.md's
"design decisions already made" section that were explicitly called out as
load-bearing are the parts holding up under direct testing —
per-parent-cell-center subdivision decision (never per-point, never
half-subdivided cells), the shared-origin alignment formula (§3.1's zero
seam error at r=10m), fully vectorized aggregation (no Python point loops
anywhere in `grid.py`/`aggregate.py`), and event-driven cache gating (§3.6)
all work exactly as specified. The core algorithmic contribution of this
module is correct; the problems found are in validation, tie-break
direction, performance delivery, and the demo/dashboard layer around it.

---

## 7. Repo hygiene

1. **`__pycache__/` is git-tracked** (`.pyc` files, plus Numba's `.nbc`/
   `.nbi` cache — see §0's forensic evidence), and currently shows as
   locally modified in `git status`. Should be gitignored and untracked;
   as-is it's binary noise in diffs and, per §0, an accidental record of
   code that no longer exists in source.
2. **`snn_weights.pth` is git-tracked** and shows modified — a binary model
   checkpoint in version control will bloat the repo on every retrain; a
   `.gitignore` entry (or Git LFS) would be more appropriate for a
   hackathon repo people are actively retraining.
3. **No `requirements.txt`/`pyproject.toml`/`setup.py` anywhere.**
   `QUICKSTART.md` gives a manual `pip install streamlit pandas numpy
   plotly torch snntorch numba scipy scikit-learn` line; `README.md`
   separately lists a broader, partly aspirational stack (Open3D, PyVista,
   torchprofile/fvcore, psutil, laspy/pypcd, torch-geometric) that doesn't
   match what's actually imported anywhere in the codebase. Worth
   reconciling into one real, minimal `requirements.txt` — especially
   since `numba` is installed and listed but never imported (§0/§4).
4. **`.gitignore` excludes every `*.md` except `/README.md`.** This means
   `CLAUDE.md`, `DESIGN.md`, `DESIGN-dash_pro.md`, `Reports/AUDIT.md` (and
   now this file) are all untracked. That's plausibly how Round 2's code
   changes and this audit report ended up out of sync (§0) — an audit
   report and design docs that never enter git history can silently drift
   from the code they describe, with no diff to catch it. Worth
   deliberately deciding whether these should be tracked, even if
   `README.md` alone stays the public-facing one.
5. **`Desktop/sih_snn_engine/` is a stale, git-tracked backup**, not a
   second live module: `dashboard_pro.py` there is a strict prefix-subset
   of the root file (missing the later Nocturne CSS redesign), and
   `dashboard_pro_v1.py` only exists there. Its copy of
   `synthetic_lidar_data.py` still has the old shared-module-level-RNG bug
   that was fixed at the repo root, confirming it predates that fix and
   isn't being kept in sync. Recommend flagging for removal/archival
   rather than leaving it in the working tree where it could be mistaken
   for a second target.

---

## 8. Recommended next actions, in priority order

1. **Root-cause the Round 2 regression** (§0) before doing anything else —
   check `git reflog`/stash list/other local branches for the missing
   Numba/validation/tie-break code before re-implementing it from scratch,
   in case it's recoverable rather than lost.
2. Re-implement (or recover) `validate_labels()`/label-range checking in
   `aggregate.py` — the out-of-range-class-ID crash (§3.2) is a hard crash
   on malformed input, not a degraded-quality issue, and is the single
   highest-severity open item.
3. Fix the majority-class tie-break direction (§3.3) — ties should not
   silently prefer "drivable terrain" over an obstacle/dynamic class in a
   safety-relevant grid.
4. Add `st.session_state`-backed persistence to `dashboard_driving.py` so a
   WebSocket reconnect doesn't discard an in-progress demo run (§5.2) —
   this is the most visible failure mode for a live demo audience.
5. Decide on and implement a real Numba/vectorization performance pass for
   `grid.py` (or accept and document the current 70-140ms/frame range) so
   the latency story given to Member 5 matches what's actually running.
6. Cap or evict `GridState`'s cache (§2, §5.2) so cell count doesn't grow
   unbounded across a long-running dashboard session.
7. Add a NaN-Z guard in `aggregate_cells` (or confirm with Member 1 that
   their upstream guarantee is enforced and this is accepted risk) —
   currently a silent single-cell corruption with no visible signal.
8. Repo hygiene cleanup (§7): untrack `__pycache__`/`snn_weights.pth`, add
   a real `requirements.txt`, decide on the `*.md` gitignore policy,
   archive/remove `Desktop/sih_snn_engine/`.

---

## §9 — Fixes Implemented & Re-Test Results — 2026-08-30

Implemented per `implement_audit_v2_fixes_guide.md` Phases 0-9 (source-code
and dashboard fixes only — Phase 10, repo hygiene/git archaeology, is
explicitly out of scope pending separate user approval). Every item below
was watched working against current source this session, not assumed from
reading the diff — commands and live dashboard runs, with actual output.

### Phase 0 — Baseline recon

Re-ran all 6 `test_*.py` and the §3.2/§3.3/§3.4/§3.5 repro script before
any changes: all matched this report's original numbers exactly (out-of-
range label crash, tie-break to class 0, NaN propagation, silent float-
spikes acceptance all reproduced as documented).

### Phase 1 — Out-of-range label → `ValueError`

Added `validate_labels()` to `aggregate.py`, called from
`handoff.generate_2_5d_grid()` before any grid work runs.

```
validate_labels(99) correctly raised: labels contains value 99, outside valid range [0, 3)
validate_labels(NUM_CLASSES-1=2): accepted, as expected
validate_labels(NUM_CLASSES=3) correctly raised: labels contains value 3, outside valid range [0, 3)
```
Also confirmed at the true entry point (`handoff.generate_2_5d_grid`, not
just `aggregate_cells`): `ValueError raised at entry point: labels
contains value 99, outside valid range [0, 3)`.

### Phase 2 — NaN-Z guard

`aggregate_cells` now raises before computing elevation stats:
```
NaN-Z correctly raised: points[:, 2] (Z) contains NaN at point indices [0]
```
`CLAUDE.md`'s contract section updated to describe the guard instead of
"accepted risk owned by Member 1." Existing ground-plane/pole assertions
in `test_aggregate.py` (which never feed NaN Z) still pass unaffected.

### Phase 3 — Non-integer `spikes` coercion

`coerce_spikes()` rounds and warns once (not raises) at the entry point,
matching real SNN output shape (`spk_rec.sum(dim=0)` is float):
```
float spikes [2.7] accepted with warning, cell committed (point_count=1)
```
Confirmed the warning actually fires via `warnings.catch_warnings` in the
test, not just that no exception was raised.

### Phase 4 — Shape/dtype validation at the entry point

`validate_inputs()` added to `handoff.py`, run before Phases 1-3's checks:
```
wrong-shaped labels (N,2) correctly raised: labels must have shape (2,) to match points, got (2, 2)
length-mismatched labels correctly raised: labels must have shape (2,) to match points, got (3,)
length-mismatched spikes correctly raised: spikes must have shape (2,) to match points, got (5,)
```

### Phase 5 — `radial_filter.py` shared constants

Replaced hardcoded `10.0`/`100.0` with `grid.INNER_RADIUS`/`OUTER_RADIUS`.
`test_grid.py`'s boundary-stress-ring check re-run after the change: zero
seam error still holds (`containment check: all 250 subdivided points'
sub-cells fall exactly within their parent cell bounds`).

### Phase 6 — Tie-break direction

Ties now favor the highest class ID:
```
tie-break (class 0 vs class 2, equal counts): resolved to class_id=2
```
(previously resolved to class_id=0, per §3.3). Non-tied pole/ground-plane
assertions in the same test file are unaffected.

### Phase 7 — Performance pass

**Profiled first, per the guide's instruction not to assume the hot path.**
cProfile over a 10-frame moving sequence found the actual bottleneck was
**not** `_sub_index_loop` (grid.py's already-fully-vectorized sub-cell
math, the lost Round-2 target) — it was `GridState.update()`'s own
Python-level per-cell loop (0.936s of 1.638s total, 57%), with
`np.unique(axis=0)`'s row-wise argsort a distant second (0.499s, 30%) in
both `grid.py` and `aggregate.py`'s cell-key dedup.

The guide's Phase 7 explicitly scoped out JIT-ing the dict-based
`GridState` cache (Numba doesn't handle Python dicts of tuples well). This
was flagged to the user mid-implementation; **the user chose to fix only
the Numba-safe np.unique(axis=0) cost** and leave the per-cell Python loop
untouched, accepting that this alone might not close the full gap.

Implemented: replaced the 2-column (`grid.py`) and 4-column
(`aggregate.py`) `np.unique(axis=0)` cell-key dedup with a packed 1D
int64 key + 1D `np.unique` (matches the "packed-int64 keys" forensic clue
from the `.nbc` cache filename in §0).

Correctness re-verified first: all 6 tests still pass after the change.

Profiled effect: **1.638s → 1.086s total (34% reduction)**; np.unique's
argsort cost **0.499s → 0.047s (91% reduction)**, exactly as targeted. The
untouched `GridState.update()` loop is now ~82% of what remains.

`test_integration.py` before/after (10-frame moving sequence):

| Frame | Before (§4 baseline) | After Phase 7 | After (this regression run) |
|---|---|---|---|
| 0 | 109.3ms (over) | 80.8ms | 90.7ms |
| 1 | 139.8ms (over) | 68.9ms | 93.9ms |
| 2 | 134.9ms (over) | 67.9ms | 99.7ms |
| 3 | 77.5ms | 52.9ms | 82.8ms |
| 4 | 77.3ms | 69.1ms | 67.2ms |
| 5 | 71.3ms | 52.8ms | 71.5ms |
| 6 | 70.4ms | 77.3ms | 83.0ms |
| 7 | 80.5ms | 65.8ms | 69.6ms |
| 8 | 69.1ms | 51.3ms | 86.9ms |
| 9 | 107.4ms (over) | 80.4ms | 106.0ms (over) |

Baseline: 5/10 frames over the 100ms budget. Immediately after the Phase 7
change: 10/10 under. This final regression-pass run (after Phase 8's LRU
bookkeeping was also added, plus normal machine-load variance): 9/10
under, 1/10 marginally over (106.0ms vs 100ms). Net: a real, substantial
improvement (roughly 20-40% faster per frame) achieved without touching
the dict-cache loop, but the budget is not yet met with full
reliability — the untouched `GridState.update()` loop remains the
dominant cost and is the next place to look if 10/10-under-budget is a
hard requirement.

### Phase 8 — `GridState` bounded cache (LRU eviction)

Added `max_cells` (default 200,000) to `GridState`, with per-`CellRecord`
`last_touched_frame` tracking and least-recently-touched eviction.
Existing tests (max ~114,910 cells accumulated) stay under the default cap
and are unaffected. New test with `max_cells=30000`:
```
LRU eviction: cache size after 20 frames = 30000 (max_cells=30000), plateaued at cap: True
```
No cell touched in a given frame was evicted during that frame (asserted
directly in the test).

### Phase 9 — Dashboard fixes

All four items implemented in `dashboard_driving.py` (and the dead
`sys.path.append` also removed from `dashboard_pro.py`):

- **WebSocket reconnect data loss (highest-severity dashboard item) —
  fixed and directly reproduced live, twice.** Rearchitected from one
  blocking Python loop per button-click into `st.session_state`-driven
  playback advancing one frame per script run via `st.rerun()`. Launched
  the dashboard, started a sequence, let it advance to frame 6, then
  navigated the tab away to a different site and back to the same URL
  (same browser session/cookie) — a genuine `tornado.websocket
  .WebSocketClosedError` was raised server-side by this. Server-side
  frame counter (confirmed via temporary instrumentation, removed after
  verification) continued **6 → 7**, not reset to 0; the browser UI
  showed the running state (progress + "Stop" control), not the idle
  "Start Driving Sequence" screen. Repeated a second time from frame 20:
  continued **20 → 21**. Both reconnects resumed cleanly.
- **Frame-counter inconsistency — fixed.** Both the HUD metric and the
  canvas caption now read a single `frame_display` variable; confirmed
  live both showed "Frame 1/40" / "Frame 1 | ... cells" in agreement
  immediately after starting a run (previously HUD used `frame_idx+1`
  and the caption used raw `frame_idx`, the source of the "5/40" vs "3"
  mismatch in §5.2).
- **Local filesystem path leak — fixed.** Sidebar now shows `Loaded:
  snn_weights.pth` instead of the full `C:\Users\...` path; confirmed
  live via screenshot.
- **Dead `sys.path.append('AVRLM')` — removed** from both
  `dashboard_pro.py` and `dashboard_driving.py`, along with the
  now-unused `import sys` in both files (confirmed via grep: zero
  remaining `sys.` references in either file).

**Observed but not a regression from this work**: driving the rearchitected
dashboard via browser automation, screenshot/page-text calls repeatedly
timed out with "the page is busy" while a sequence was running — the
full-page `st.rerun()` per frame is heavier on the frontend than the old
per-placeholder-update loop was, since Streamlit reconciles the whole
widget tree each frame instead of patching individual `st.empty()` slots
within one continuous run. The animation itself kept advancing correctly
server-side throughout (confirmed via server-side logging), so this is a
UI-responsiveness cost of the fix, not a functional break, but worth
noting: the dashboard may feel less smooth per-frame than before, trading
smoothness for actually surviving a reconnect.

### Overall regression pass

- All 6 `test_*.py` scripts: **exit 0**, no failures.
- All AUDIT-v2 §3 repros re-run and now behave as fixed (§3.2 raises
  cleanly, §3.3 resolves to class 2, §3.4 raises cleanly, §3.5 accepted
  with a warning); §3.1 (boundary ring) and §3.6 (event-driven gating)
  re-confirmed still passing.
- `test_integration.py` timing: improved from a 5/10-over-budget baseline
  to 9-10/10 under budget depending on run (see Phase 7 table above);
  `GridState.update()`'s Python loop remains the largest unaddressed cost
  by design (user's explicit choice, see Phase 7).
- Both dashboards re-driven live; reconnect and frame-counter fixes
  directly confirmed as described above.

### Phase 10 — not started (needs separate approval)

Per the implementation guide, the following remain undone and require
explicit user sign-off before any of them proceed, since they're git/repo-
state changes rather than source fixes:

1. Investigate `git reflog`/stash/other local branches for the lost
   Round-2 work (in case any of it — beyond what was independently
   re-implemented here — is recoverable rather than needing a redo).
2. Untrack `__pycache__/` (including the stale Numba `.nbc`/`.nbi` cache)
   and add it to `.gitignore`.
3. Untrack `snn_weights.pth`; add to `.gitignore` or set up Git LFS.
4. Decide the `.gitignore` `*.md` policy (currently everything except
   `/README.md` is excluded — the plausible mechanism behind §0's
   regression going undetected).
5. Add a real `requirements.txt` reconciling `QUICKSTART.md`'s manual
   install line against actual imports (noting `numba` is installed but,
   after Phase 7's targeted fix, still not the tool used — the fix used
   plain NumPy, not `@njit`).
6. Archive or remove `Desktop/sih_snn_engine/` (stale tracked backup with
   the pre-fix shared-RNG bug still present).

---

## §10 — Repo Hygiene & Git Archaeology — 2026-08-30

Implemented with explicit user approval, as its own set of changes on top
of Phases 0-9 (not committed by this session - left for the user to
review and commit).

1. **Git archaeology for the lost Round-2 work.** `git reflog`, `git stash
   list`, `git branch -a`, and `git fsck --unreachable` found no dangling
   commits or stashes containing it - the loss isn't a matter of an
   orphaned commit sitting in history waiting to be cherry-picked.
   However, `git grep` across every commit on every branch found the
   string `validate_labels` inside the **tracked** `.pyc` files at the
   very first commit of this branch (`57bbedf`, "intial drafts") -
   meaning the compiled bytecode of the fixed source was already stale
   and committed by accident before this audit even began. Loading that
   bytecode directly (`marshal.loads` - Python 3.14's bytecode magic
   number matches the current interpreter exactly, so it's directly
   loadable, not just inspectable as bytes) confirms:
   - `aggregate.cpython-314.pyc` contained `validate_labels`,
     `_pack_cell_key`, `_unpack_cell_key`, and `aggregate_cells`. Its
     `validate_labels` docstring literally says *"Blocker 2 (AUDIT.md
     round 1): raise a clear error at the input boundary instead of
     letting an out-of-range class ID crash the bincount/reshape math
     below"* - proving this fix was written specifically in response to
     the Round 1 finding, then lost before ever being committed as source.
     Its logic (`np.unique` + a `ValueError` naming the bad values and
     `NUM_CLASSES`) matches Phase 1's independent reimplementation almost
     exactly.
   - `grid.cpython-314.pyc` contained `njit`, `numba`, and
     `_sub_index_loop` - confirming the Numba pass really existed.
   - `handoff.cpython-314.pyc` contained `validate_inputs`.
   - The lost `_pack_cell_key`/`_unpack_cell_key` used bit-shift packing
     (`_PIX_SHIFT`/`_PIY_SHIFT`/`_PIY_MASK`) rather than Phase 7's
     multiplication-based packing - a different technique for the same
     packed-int64-key idea, both valid.

   Full source reconstruction was not attempted (would need a Python 3.14
   bytecode decompiler, tooling that's immature for such a new version)
   and was judged unnecessary: Phases 1, 4, and 7 already reimplement
   equivalent functionality, independently written and independently
   tested this session rather than trusted from either the old bytecode
   or AUDIT.md's prose.
2. **Untracked `__pycache__/`** (`git rm -r --cached`, including the
   `.nbc`/`.nbi` Numba cache files that were this report's original §0
   evidence) and added `__pycache__/`, `*.pyc`, `*.nbc`, `*.nbi` to
   `.gitignore`. Files remain on disk, only removed from git tracking.
3. **Untracked `snn_weights.pth`** (`git rm --cached`) and added it to
   `.gitignore`. No Git LFS setup introduced - plain `.gitignore` matches
   this repo's current scale, and the checkpoint is regenerable via
   `synthetic_train_loop_v5.py`.
4. **`.gitignore` `*.md` policy decided**: keep the blanket `*.md` ignore
   (most `.md` files here are one-off planning/task documents), but add
   explicit exceptions for `CLAUDE.md`, `DESIGN.md`, `DESIGN-dash_pro.md`,
   `QUICKSTART.md`, and everything under `Reports/` - these are the docs
   that describe the current contract/design/audit state and are exactly
   the ones whose untracked drift from source caused §0's regression to
   go undetected. `Plans_Agent/*.md` and the large standalone spec doc
   stay ignored (transient planning artifacts, not living contracts).
5. **Added `requirements.txt`**, reconciled against actual `import`
   statements across the repo (excluding the now-removed `Desktop/`
   copy) rather than README.md's broader aspirational list: `streamlit`,
   `pandas`, `numpy`, `plotly`, `torch`, `snntorch`, `scikit-learn`.
   `numba` is listed but commented out - installed, but not imported
   anywhere after Phase 7 (which used plain NumPy). `scipy` is not
   listed - confirmed genuinely unused, matching CLAUDE.md §5's own
   prediction ("SciPy... likely not [needed]"). Verified with `pip
   install --dry-run -r requirements.txt`: resolves cleanly against the
   current environment, no conflicts.
6. **Removed `Desktop/sih_snn_engine/`** (`git rm -r`) - confirmed stale
   duplicate of `dashboard_pro.py` (a strict prefix, pre-dating the
   Nocturne CSS redesign) plus `dashboard_pro_v1.py` and copies of
   `synthetic_lidar_data.py` etc. that still carried the already-fixed
   shared-RNG bug. Recoverable from git history if ever needed.

All 6 `test_*.py` re-run after these changes: **exit 0**, no failures
(git/doc/dependency changes don't touch runtime code paths).

Nothing in this section has been committed - `git status` shows these as
staged/unstaged changes for the user to review before committing, per the
guide's instruction to treat each Phase 10 item as its own reviewable
change rather than one bundled commit.

---

## §11 — Test Suite Rebuild — 2026-08-30

Implemented per `rebuild_test_suite_guide.md`, bringing the suite up to
date with Phases 1-10 above. Every existing assertion got an explicit
disposition (kept as-is, fixed, or given a new companion) rather than a
blanket rewrite; three new files cover behavior that had no coverage at
all. Suite convention (script-style vs. pytest) was intentionally left
untouched - that's the gated Phase 7 decision, presented separately.

### Coverage map (Phase 0)

| Change | Existing coverage before this rebuild | Disposition |
|---|---|---|
| `validate_labels` out-of-range | `test_aggregate.py` - tested 99/`NUM_CLASSES-1`/`NUM_CLASSES` but never asserted the error message content | fixed |
| NaN-Z guard | `test_aggregate.py` - one-NaN+one-normal point, asserted `ValueError` | kept + companions added |
| float spikes coercion | `test_handoff.py` - already matched current behavior | kept + companions added |
| `validate_inputs` shape/length | `test_handoff.py` - each field wrong alone | kept + companions added |
| `radial_filter.py` importing `grid.py`'s radius constants | `test_radial_filter.py` | kept, unchanged (behavior-invariant) |
| tie-break direction reversal | `test_aggregate.py` - already asserted class 2 wins the 1-vs-1 case | kept + 3-class companion added |
| packed-int64 keys (`grid.py`, `aggregate.py`) | none | added (`test_packed_keys.py`) |
| LRU eviction / `max_cells` | `test_grid_state.py` - cap + no-mid-frame-eviction + plateau over 20 frames | kept + added (exact fill+1, LRU-vs-FIFO, pressure, 800-frame stress) |
| dashboard session persistence | none | added (`test_dashboard_driving.py`) |
| frame-counter single source | none | added |
| checkpoint path-leak fix | none | added |
| `test_integration.py` timing | zero `assert` statements existed | first assertion added (was not a "fix", the file had never asserted anything) |

Two premises in the original guide didn't match reality once checked
against current source (per the guide's own "doubt rule"): the tie-break
test already asserted the post-fix winner, and no test anywhere hardcoded
"114,910 cells" or asserted unbounded growth - that figure only ever
appeared as an explanatory comment.

### Tests added

- **`test_aggregate.py`**: message-content asserts on the existing
  `validate_labels` boundary test; empty-array/large-valid-array/`-1`
  cases for `validate_labels`; all-NaN-Z-cell and isolated-NaN-Z-with-
  otherwise-valid-fields cases; a 3-class tie-break companion (count
  beats tie-break rule).
- **`test_handoff.py`**: `validate_inputs` with only `points` malformed,
  and all three fields wrong at once (confirms the points-first
  short-circuit order); `coerce_spikes` rounding-direction edge cases
  including the round-half-to-even (`2.5 -> 2`, not 3) case and the
  measured-not-assumed `1e-6` threshold boundary; once-per-call (not
  once-ever) warning confirmation.
- **`test_integration.py`**: a real assertion (>=8/10 frames under the
  100ms budget) where none existed before.
- **`test_packed_keys.py`** (new): differential tests proving the packed-
  1D-int64 dedup in `grid.py`/`aggregate.py` produces identical cell sets
  to an independently-reimplemented `np.unique(axis=0)`, on both
  `build_scene()` and `generate_boundary_stress_ring()`; injectivity
  checks for both packing schemes at their documented domain boundaries.
- **`test_grid_state.py`**: exact fill-to-`max_cells`-plus-one eviction;
  a touch-before-evict case that specifically distinguishes true LRU from
  FIFO; eviction-under-pressure (a same-call batch pushing a pre-filled
  cache over the cap); an explicit same-call-batch-exceeds-cap-outright
  case (documented as an inherent mathematical limit, not a bug); an
  800-frame plateau stress test.
- **`test_dashboard_driving.py`** (new): a small pure `advance_frame_index`
  helper was factored out of `dashboard_driving.py`'s playback block into
  a new dependency-free module, `dashboard_state.py`, specifically so it's
  testable without importing the full Streamlit/torch script. Tests cover
  reconnect resumption (not reset), accumulated-grid survival across
  simulated reruns, record-list recomputation idempotency, a source-text
  structural check that the frame counter has one single source read by
  both the HUD and both caption sites, and the path-leak fix.
- **`test_stress.py`** (new): 230 validation-fuzzing cases (shape/length
  mismatches, out-of-range labels, NaN/Inf-Z, non-integer/negative
  spikes, empty frames, valid randoms) each tagged with its expected
  outcome by construction; a 10,000-frame adversarial-churn eviction test
  (one brand-new cell every frame, cap=1,000); a 500-frame long-horizon
  growth test on genuinely novel terrain (`build_moving_sequence`,
  cap=30,000, cap is reached and held); a head-to-head timing comparison
  of the packed-key dedup against an inline old-style `axis=0` dedup on a
  ~123,500-point dense scene (confirmed 7-8x faster across two runs) as a
  hardware-independent regression guard.

### Findings from the new tests

- **Timing-assertion flakiness caught and fixed during this rebuild**: the
  eviction-churn test's first draft asserted last-100-frames median time
  stays within 2x of first-100-frames median. Running it through `pytest
  --collect-only` (which executes these script-style files at import/
  collection time) surfaced a real 2.7x jump from system-load variance
  alone, no code change involved. Loosened to a 5x tolerance - still tight
  enough to catch a genuine O(n)-in-cache-size regression (which would
  show as an order-of-magnitude jump over 10,000 frames), loose enough to
  absorb normal jitter. Re-verified stable across multiple direct and
  pytest-collection runs afterward.
- **Inf-Z is legitimately unguarded, confirmed by the fuzz suite**: per
  CLAUDE.md's documented scope, only NaN-Z is checked, not Inf-Z. The
  fuzz test in `test_stress.py` confirms Inf-Z points are accepted (not
  rejected) and surfaces a `RuntimeWarning: invalid value encountered in
  subtract` from `aggregate.py:124` (`inf - inf = nan` inside the
  elevation-variance calc) as an expected, harmless side effect of that
  documented gap - not something this test-rebuild pass fixes, since
  fixing production validation behavior is out of this task's scope.

### Gaps intentionally left uncovered

- **Live multi-tab WebSocket reconnect** for `dashboard_driving.py`
  remains a documented manual regression step (in
  `test_dashboard_driving.py`'s module docstring) rather than an
  automated Chrome-driven test - disproportionate to automate for a
  regression check versus the original one-time Phase 9 verification.
- **Path-leak "without a flag" variant**: no debug/dev flag exists
  anywhere in `dashboard_driving.py` (confirmed by source-text grep in
  the test itself) - only the single unconditional code path is tested.

### Phase 7 — pytest migration (approved and implemented)

Presented to the user as a choice between leaving the suite as runnable
scripts (with a docs note) or migrating to pytest-discoverable naming;
**migration was chosen**. All 9 files were rewritten from top-level-assert
scripts into `def test_...()` functions:

- Each file's existing "---"-delimited sections became its own named test
  function (e.g. `test_tie_break`, `test_lru_eviction_exact_fill_plus_one`),
  giving genuine per-test pass/fail reporting instead of one pass/fail per
  file. Shared helpers (`run_pipeline`, `cluster_cell_keys`, the mesh-index
  generators) stay as plain module-level functions, not prefixed `test_`,
  so pytest doesn't try to collect them.
- `test_handoff.py`'s tests that call `generate_2_5d_grid()` now explicitly
  reset the module-level `handoff._state` singleton at the start of each
  test (`handoff._state = GridState()`) - a correctness fix required by
  the new shared-process execution model: the original scripts each ran in
  their own fresh Python process, so the singleton started empty every
  time; under pytest, multiple test functions share one process, so
  without the reset a later test would silently inherit an earlier test's
  accumulated grid state.
- `dashboard_driving.py`'s `advance_frame_index` helper (extracted in
  Phase 5) was moved out to its own dependency-free module,
  `dashboard_state.py`, specifically so `test_dashboard_driving.py` can
  `import` it directly under pytest without triggering
  `dashboard_driving.py`'s Streamlit/torch top-level code at collection
  time.
- `test_stress.py`'s fuzz-case construction and all four stress scenarios
  were moved from module-level code into their respective test functions
  (previously they ran unconditionally at import time) - this was
  necessary, not just style: script-style files executed their entire
  body on every `python test_X.py` run *and* on every pytest
  collection/import, but proper pytest tests should do their work when
  *run*, not when *collected*.
- **A real flakiness bug was caught and fixed during the migration
  itself**: running the suite through `pytest --collect-only` (which
  executes these files' import-time code) surfaced the eviction-churn
  timing assertion failing under system-load variance that a bare
  `python test_stress.py` run hadn't hit. Fixed by loosening the
  tolerance from 2x to 5x (documented at the assertion site) - see
  "Findings from the new tests" above, which predates this section but
  covers the same fix.
- A backward-compatible `if __name__ == "__main__":` block was added to
  every file, calling all its test functions in the original order and
  printing the original "all assertions passed" line - `python test_X.py`
  still works exactly as before, it's just no longer the *only* way to
  run these tests.

**Verification after migration:**
- `py -3 -m pytest -v`: **collected 41 items, 41 passed**, 47.7s total
  (2 warnings - the same documented Inf-Z `RuntimeWarning` and the
  expected `coerce_spikes` non-integer `UserWarning`, both from
  `test_stress.py::test_validation_fuzzing`, not failures).
- `py -3 -m pytest --collect-only`: now takes **~1s** (was ~85s before the
  migration, since collection no longer executes the 10,000-frame churn
  loop and other heavy work at import time) and correctly reports
  **"collected 41 items"** - §4's "collects 0 items" trap is resolved.
- `py -3 -m pytest -k tie_break`: correctly selects and runs just the 2
  matching tests (`test_tie_break`, `test_tie_break_three_class_companion`),
  confirming `-k` filtering works as intended.
- Direct execution (`python test_X.py` for all 9 files) re-verified
  unchanged and still passing after the rewrite.

### Full regression pass (Phase 8)

All 9 files (`test_aggregate.py`, `test_grid.py`, `test_grid_state.py`,
`test_handoff.py`, `test_integration.py`, `test_radial_filter.py`,
`test_packed_keys.py`, `test_dashboard_driving.py`, `test_stress.py`) run
directly via `py -3 test_X.py`: **exit 0, all assertions pass**, and via
`py -3 -m pytest`: **41 passed**, both reproduced across multiple runs.
