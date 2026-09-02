# AUDIT v3 — End-to-End Pipeline, Dashboard & Desktop App Audit

Date: 2026-09-02
Branch: `main` @ `f9b2e57` (working tree has uncommitted changes to
`avrlm_radar_app.py`, `engine_adapter.py`, `radar_view_2d.py`, and an
untracked new file `terrain_relief.py` — these are the desktop-app passes
this audit re-verifies)
Scope: grid engine re-verification, both Streamlit dashboards
(`dashboard_pro.py`, `dashboard_driving.py`, `dashboard_state.py`), the
PyQt6 desktop app (`avrlm_radar_app.py`, `engine_adapter.py`,
`radar_view_2d.py`, `terrain_relief.py`), and repo hygiene.

This is an independent re-verification against current source, the same
standard as `Reports/AUDIT-v2.md` §0 set: every finding below is backed by
a file:line citation or a reproduced diagnostic script's actual printed
output — nothing is carried forward from an earlier report's prose without
being re-checked against the code on disk today.

Since AUDIT-v2 (2026-08-30), five passes landed on top of the grid-engine
work it covered: live-mode wiring of `engine_adapter.py` to the real
SpikingPointNet + grid engine, an ego-vs-world coordinate-frame fix,
`GridState` periodic reset for unbounded memory growth, a `QThread` worker
so real inference doesn't block the Qt UI thread, track-match-radius
widening (4.0m → 7.0m) to cut ID churn, a two-stage (class-then-hits)
render/density split plus velocity clamping+EMA to declutter the radar,
and a new near-field terrain relief panel. This report checks all of it.

---

## 1. Grid engine — re-verification of AUDIT-v2's findings

AUDIT-v2's §0 headline finding (Round-2 fixes claimed but absent from the
tree) is now resolved — every one of those fixes is present and correct in
current source:

| AUDIT-v2 finding | Current status | Evidence |
|---|---|---|
| `validate_labels()` missing from `aggregate.py` | **Fixed** | `aggregate.py:20-30`, raises `ValueError` naming the offending value |
| `validate_inputs()` missing from `handoff.py` | **Fixed** | `handoff.py:17-38`, shape/length checks run before label/spike validation |
| `radial_filter.py` hardcoding `10.0`/`100.0` instead of importing `grid.py`'s constants | **Fixed** | `radial_filter.py:9` imports `INNER_RADIUS, OUTER_RADIUS` from `grid.py` |
| `GridState` unbounded growth, no eviction | **Fixed** | `grid_state.py:32,52-94` — `OrderedDict` LRU keyed by `DEFAULT_MAX_CELLS=200_000`, `move_to_end` per touch, `popitem(last=False)` eviction |
| Z NaN not guarded | **Fixed** (CLAUDE.md-mandated addition, not an AUDIT-v2 item) | `aggregate.py:77-83` raises `ValueError` naming offending point indices before any elevation math runs |
| Numba JIT pass | Still absent — plain NumPy | `grid.py`/`aggregate.py`, no `numba`/`njit` import anywhere in the repo |

**One residual gap, not previously flagged**: `aggregate_cells()`
(`aggregate.py:75`) has no internal label-range guard of its own — it
trusts `lbl = labels[mask].astype(np.int64)` blindly and uses it directly
as a bincount index (`aggregate.py:131`,
`group_id * NUM_CLASSES + lbl`). The *only* thing preventing an
out-of-range label from reaching this line is that `handoff.py:54-55`
happens to call `validate_labels(labels)` before `_state.update()`. Any
future caller that reaches `aggregate_cells()` directly — bypassing
`handoff.generate_2_5d_grid()` — gets no protection at all. Low severity
today (no such caller exists), but it's a guard living at the wrong layer:
the function that actually needs the invariant doesn't enforce it itself.

**Recommendation**: move (or duplicate) the `validate_labels()` call inside
`aggregate_cells()`, or accept the current contract explicitly with a
one-line comment noting the invariant is caller-enforced.

The Numba absence and unpinned dependencies are unchanged from AUDIT-v2
and are not re-litigated here — see AUDIT-v2 §2/§7 for that discussion.

---

## 2. Dashboard audit (`dashboard_pro.py`, `dashboard_driving.py`, `dashboard_state.py`)

Both dashboards independently copy the same scene→tensor→model→cluster
pipeline (a consequence of not sharing a module with `engine_adapter.py`),
and the copies have drifted from each other in several concrete ways:

| # | Finding | Severity | Evidence |
|---|---|---|---|
| 2.1 | `dashboard_pro.py`'s checkpoint load omits `weights_only=True`; `dashboard_driving.py`'s includes it. `torch.load` without `weights_only=True` uses the full pickle deserializer — a real (if low-probability, local-file) deserialization-safety gap that the sibling dashboard already closed. | Medium | `dashboard_pro.py:131` vs `dashboard_driving.py:112` |
| 2.2 | `CHECKPOINT_PATH` is a bare relative string in `dashboard_pro.py` (`"snn_weights.pth"`, resolved against the process's CWD) vs. an `__file__`-relative absolute path in `dashboard_driving.py` and `engine_adapter.py`. Running `dashboard_pro.py` from any directory other than the repo root silently fails to find a checkpoint that exists. | Medium | `dashboard_pro.py:95` vs `dashboard_driving.py:26` |
| 2.3 | Neither dashboard's `load_model()` wraps `torch.load` in a try/except. Both call it unconditionally at module import time (`ai_model, device = load_model()` / `ai_model, device, checkpoint_loaded = load_model()`, outside any button/callback), so a present-but-corrupt or wrong-format checkpoint file crashes the entire Streamlit app on load rather than falling back gracefully — unlike `engine_adapter.py`'s `Engine.__init__`, which wraps the equivalent code in try/except and sets `self.live_init_error` (see engine_adapter.py's live-mode init). | Medium | `dashboard_pro.py:126-133`, `dashboard_driving.py:106-115` |
| 2.4 | `cluster_objects(df, eps, min_samples)` exists in both dashboards with the same name and near-identical DBSCAN body, but different signatures: `dashboard_pro.py` returns `objects` only; `dashboard_driving.py` returns `(objects, clustered_points)` (it additionally filters out DBSCAN noise label `-1` and hands back the clustered subset). A third, independent reimplementation of the same concept lives in `engine_adapter.py._cluster_cells_to_detections(active_map, ugv_world_pos)`, which operates on aggregated grid cells (not raw per-point predictions), uses a different class-label scheme (`LIVE_CLASS_LABELS = {1: "obstacle", 2: "vehicle"}` vs. the dashboards' hardcoded `"Static Obstacle"`/`"Dynamic Threat"` strings), and adds a min-cluster-cell filter plus a nearby-cluster merge pass that neither dashboard has. Three independently-maintained copies of "DBSCAN classified data into discrete objects," each behaviorally different. | Low-Medium (maintenance hazard, not a live bug) | `dashboard_pro.py:295-321`, `dashboard_driving.py:137-174`, `engine_adapter.py`'s `_cluster_cells_to_detections`/`_merge_nearby_clusters` |
| 2.5 | Dead variable: `checkpoint_loaded` is computed inside `load_model()`, returned, and assigned at module scope (`ai_model, device, checkpoint_loaded = load_model()`) but never read again anywhere in the file. | Low | `dashboard_driving.py:110-118` (only 4 occurrences of the name in the whole file, all in the definition/assignment) |
| 2.6 | `_scene_to_tensor`/`scene_to_tensor`'s unseeded `np.random.choice(n, num_points, replace=False)` per-frame point resample is present verbatim in both dashboards and is the root cause discussed in detail in §3 below for the desktop app — confirmed it originates here, not in `engine_adapter.py`. | See §3 | `dashboard_driving.py:124`, mirrored in `dashboard_pro.py` and `engine_adapter.py._scene_to_tensor` |
| 2.7 | Extensive use of raw `st.markdown(..., unsafe_allow_html=True)` for custom-styled panels (16 call sites in `dashboard_pro.py` alone) — all current call sites build their HTML from fixed f-strings/constants with no user-controlled interpolation, so there is no live XSS vector today, but the pattern is fragile: any future change that interpolates a value derived from uploaded data, a filename, or a URL parameter into one of these strings would introduce one with no code-level barrier stopping it. | Low (latent pattern risk, not an active vulnerability) | `dashboard_pro.py:92,153,159,169,174,184,192,201,209,216,242,261,267,666,747,842,846` |

`dashboard_state.py` (14 lines) was read in full — a trivial
`st.session_state` accessor wrapper, no issues found.

---

## 3. `avrlm_radar_app.py` — why the radar still looks like a mess of obstacles

### 3.1 What Pass 6 actually changed, and why it wasn't enough

Pass 6 added a two-stage declutter in `Engine.step()`
(`engine_adapter.py:379-404`):

1. **Class split** — only `"vehicle"`-labeled detections go through
   `ObjectTracker.update()`; everything else becomes an untracked
   `density_point` immediately.
2. **Persistence split** — among confirmed vehicle tracks, only ones with
   `trk.threat == True` or `trk.hits >= RENDER_MIN_HITS` (10) get full
   icon treatment (`radar_view_2d.py`'s cube icon + comet trail + velocity
   arrow + elevation pin, drawn by `_draw_tracks`); the rest are demoted
   to density points.

This measurably reduced clutter from the pre-Pass-6 state (every
"vehicle"-labeled detection, however transient, got full tracked-object
treatment). But three compounding effects — all confirmed live against
the real `Engine()` in live mode this session — explain why a meaningful
population of full-icon objects is still on screen at any given moment.

### 3.2 Cause 1 — the rendered-track population grows across loop replays, not just within one

`driving_sequence.py` provides a fixed 40-frame sequence
(`LIVE_N_FRAMES=40`) that the desktop app loops indefinitely. Re-running
the app's `Engine()` for 200 frames (5 full loop replays) and logging
`len(frame.tracks)` every 20 frames:

```
$ py diag_audit1.py
live: True
frame 20:  frame.tracks=27  density_points=49  tracker.active total=93
frame 40:  frame.tracks=40  density_points=50  tracker.active total=99
frame 60:  frame.tracks=34  density_points=40  tracker.active total=87
frame 80:  frame.tracks=44  density_points=39  tracker.active total=91
frame 100: frame.tracks=34  density_points=43  tracker.active total=90
frame 120: frame.tracks=48  density_points=34  tracker.active total=90
frame 140: frame.tracks=36  density_points=38  tracker.active total=81
frame 160: frame.tracks=41  density_points=38  tracker.active total=88
frame 180: frame.tracks=40  density_points=44  tracker.active total=91
frame 200: frame.tracks=43  density_points=36  tracker.active total=90

frame.tracks per frame: frames 1-40=25.5  frames 161-200=40.2 (first vs. last replay cycle)
```

`tracker.active` (the total known-track population, confirmed or not)
plateaus quickly around 87-99 — that part is stable, `TRACK_MAX_AGE_FRAMES`
is doing its job. But `frame.tracks` (the subset that clears
`RENDER_MIN_HITS`) climbs from an average of 25.5 in the first replay
cycle to 40.2 by the fifth — a ~58% increase with no sign of having
plateaued at 200 frames. **The hit-count gate is a threshold on
accumulated real elapsed time, not on "is this real," and the desktop app
runs far longer than a one-off dashboard click-and-observe session** — so
the longer it runs, the more of the ~90-track noise population eventually
crosses the bar and starts rendering as a full icon.

### 3.3 Cause 2 — the threat-override safety net is itself noise-contaminated

Pass 6's `trk.threat or trk.hits >= RENDER_MIN_HITS` rule was written so a
genuine closing threat is never hidden by the hit-count gate. Measuring
how rendered tracks actually get onto screen over 100 frames:

```
$ py diag_audit2.py
n_frames=100
total rendered track-instances (summed across frames): 3354
  via threat override ONLY (hits < 10 but threat=True): 697  (20.8%)
  via hits>=threshold only (not flagged threat):        2175 (64.8%)
  via both hits AND threat:                             482  (14.4%)

fraction of ALL tracker.active tracks with is_dynamic=True: mean=0.794
fraction of ALL tracker.active tracks with threat=True at any given frame: mean=0.165
```

**20.8% of every rendered track-instance is on screen purely because
`trk.threat` fired, with fewer than 10 real matches behind it** — i.e. a
fifth of the "obstacles" the user sees bypassed the declutter entirely.
This is because the inputs to `is_dynamic`/`threat` are unreliable at the
population level, not just occasionally: **79.4% of every active track at
any moment** (not just rendered ones) is flagged `is_dynamic=True`, and
**16.5%** is flagged `threat=True`. A scene with 3-4 real moving actors
(per `driving_sequence.py`) cannot plausibly have ~80% of its detections
be genuinely dynamic — this confirms the Pass-6 plan's own finding that
smoothed velocity doesn't discriminate real motion from re-clustering
noise, and shows the consequence flowing straight through to the
supposedly-safe override path.

### 3.4 Cause 3 — the shared root cause: unseeded per-frame point resampling

Both causes above trace to one thing: `_scene_to_tensor`'s
`np.random.choice(n, NUM_POINTS_LIVE, replace=False)`
(`engine_adapter.py`, same pattern as `dashboard_pro.py`/
`dashboard_driving.py`, see §2.6) is **unseeded and re-rolled every
frame**. `driving_sequence.py`'s sequence is looped, so the exact same
physical scene (same points, same true labels) is replayed every
`LIVE_N_FRAMES`, but a *different random 8192-point subsample* is fed to
the model each time — producing real classification and cluster-centroid
instability for objects that have not moved at all between visits. This
is not something introduced this session (confirmed present verbatim in
both dashboards' code and documented in `DESIGN.md`/`DESIGN-dash_pro.md`);
the desktop app's continuous multi-minute runtime just exposes it far more
starkly than a dashboard's single-click-and-observe usage pattern ever
would, because the population of noisy detections gets more real time to
accumulate hits (§3.2) and to spike `is_dynamic`/`threat` (§3.3).

### 3.5 A fourth, compounding factor: full-glyph treatment is concentrated exactly where the scene is busiest

`radar_view_2d.py`'s `_draw_tracks` (`radar_view_2d.py:295-310`) only
draws the full cube-icon + comet-trail + velocity-arrow treatment for
tracks within `DETAIL_RANGE_M = 25.0` meters; beyond that it's a plain
dot. `driving_sequence.py`'s actual actors (the UGV's own path, an
oncoming/overtaking vehicle, a pedestrian) are concentrated close to the
UGV by construction — the same near-field zone where Cause 1-3's noise
population is also densest (closer objects produce more, tighter DBSCAN
clusters per frame). So the 25-48 tracks/frame measured in §3.2 aren't
spread evenly across the 100m sensor range — a disproportionate share of
them sit inside the 25m detail radius, each getting the heaviest visual
treatment simultaneously. This is why the display reads as "a mess of
obstacles" rather than "some noisy far-field dots": the noise is loudest
exactly where the renderer tries hardest to make things legible.

### 3.6 What's fixed vs. still open

| Item | Status |
|---|---|
| Every "vehicle" detection getting full tracked treatment, however transient | Fixed (Pass 6 class split) |
| Track-ID churn from a too-tight match radius | Fixed (Pass 5, 4.0m→7.0m) |
| Raw velocity-arrow noise (median 48m, max 352m pre-Pass-6) | Substantially reduced (clamp+EMA), not eliminated — see AUDIT context in the Pass-6 plan for the measured before/after |
| UI freezing during real inference | Fixed (QThread worker, Pass 4) |
| GridState unbounded memory/latency growth | Fixed (periodic reset, Pass 4/5) |
| **The population of rendered "obstacles" growing over long runtime, and the threat-override bypassing the declutter for ~21% of them** | **Open** — both are downstream symptoms of §3.4's root cause, not yet addressed by anything shipped so far. `RENDER_MIN_HITS` and the threat override are mitigations layered on top of noisy input, not a fix to the noise itself. |

**Recommendation** (highest leverage, not implemented by this audit):
stabilize `_scene_to_tensor`'s resample — e.g. seed it deterministically
per `driving_sequence.py` index (`np.random.default_rng(idx)` instead of
the global unseeded `np.random.choice`), or cache the sampled indices
across the `LIVE_RESET_EVERY_N_FRAMES` window — so that revisiting the
same physical scene produces the same detections instead of re-rolling
classification noise every single frame. This addresses §3.2's and
§3.3's root cause directly rather than adding another downstream
threshold.

---

## 4. Terrain relief mapping (`terrain_relief.py`)

New since AUDIT-v2 (untracked file, added this session). Read in full for
this audit.

### 4.1 The near-field panel disagrees with the radar panel about what's nearby

`TerrainReliefView._draw_markers` (`terrain_relief.py:265-295`) iterates
**only `frame.tracks`** — the same hits/threat-gated, post-declutter set
`radar_view_2d.py` uses for full-icon treatment. It never looks at
`frame.density_points`. Net effect: an object inside the panel's own 10m
radius — precisely the zone where per-object detail is supposed to matter
most for a near-field terrain view — needs the same `RENDER_MIN_HITS=10`
or a threat flag to get a marker at all, while the main radar panel shows
the identical object immediately as a density-field dot (§3, `radar_view_2d.py:280-293`).
The two panels can legitimately disagree, at the same instant, about
whether there's something at a given nearby location.

**Recommendation**: either draw `frame.density_points` within the panel's
`radius_m` as small dots (mirroring `_draw_density_field`'s treatment on
the main radar), or explicitly document that the terrain panel is
intentionally "confirmed objects only" — as it stands this looks like an
oversight rather than a decision.

### 4.2 Resolved: the "Mean of empty slice" `RuntimeWarning` leak

This was an open thread from earlier in this session. Root cause found:
`fill_gaps()` (`terrain_relief.py:116-139`) suppresses the warning
correctly inside its main iterative-fill loop —

```python
with np.errstate(invalid="ignore"), warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)   # terrain_relief.py:131-133
    neighbor_mean = np.nanmean(np.stack(neighbors, axis=0), axis=0)
```

— but its **fallback path does not**:

```python
remaining = np.isnan(filled)
if np.any(remaining):
    fallback = np.nanmean(filled)   # terrain_relief.py:137 -- NOT wrapped
```

This fallback only runs when NaNs survive all 4 fill passes, which happens
precisely when the input `height_array` is **entirely** NaN — i.e. when
`bin_cells_to_heightfield` was called with zero cells inside the 10m
radius (`terrain_relief.py:88-99` returns an all-NaN array immediately in
that case). That situation is not rare: `TerrainReliefView.update_frame`
(`terrain_relief.py:205-227`) feeds it `frame.live_cells` directly, and
`engine_adapter.py`'s live path resets `handoff._state = GridState()`
every `LIVE_RESET_EVERY_N_FRAMES=10` frames (`engine_adapter.py:429-445`)
— a fix built for the *tracking/latency* side of the system. For a few
frames right after every such reset, the near-field zone has genuinely
accumulated zero cells yet, `fill_gaps` hits its unwrapped fallback, and
`np.nanmean` on an all-NaN array emits the warning. **This is a one-line
fix** (wrap line 137 in the same `catch_warnings`/`errstate` pattern as
lines 131-133) — confirmed but not applied, since this audit is read-only.

### 4.3 Placeholder colorization range for real data, unverified

`update_frame` colorizes real live cells with `colorize(height, vmin=0.0,
vmax=3.0)` (`terrain_relief.py:219`) — the comment above it (lines 216-218)
already flags this as a placeholder range, not measured against
`driving_sequence.py`'s actual elevation distribution. Confirmed this is
still true: no code anywhere computes or asserts the real scene's
elevation range against these bounds. Low severity (visual-only, doesn't
affect correctness of the underlying heightfield data), but worth
resolving the same way AUDIT-v2 flagged similar placeholder ranges
elsewhere.

### 4.4 Reset-induced flicker (a side effect of §4.2/§3's shared reset mechanism)

A consequence of §4.2: every `LIVE_RESET_EVERY_N_FRAMES` frames, the
terrain panel's real-data path (`frame.live_cells is not None`) briefly
sees a sparse-to-empty near-field cell set as `GridState` rebuilds from
scratch, so the heightfield visibly flattens toward its fallback color for
a few frames before repopulating. This is not a bug in `terrain_relief.py`
itself — it's a visible cost, on a panel that didn't exist when the reset
mechanism was designed, of a fix built to solve a different subsystem's
memory/latency problem (`engine_adapter.py:429-445`'s own comment
documents the original tradeoff in terms of step-time growth, with no
mention of the terrain panel).

### 4.5 What's solid

`sample_heightfield` (synthetic placeholder) and `bin_cells_to_heightfield`
(real path) both correctly apply the heading-up rotation convention
matching `radar_view_2d.py`'s `_draw_tracks` (bearing minus heading; here,
the inverse, adding heading to local offsets — `terrain_relief.py:38-44`
documents this explicitly and the math checks out). `_draw_ugv_marker`
correctly keeps the UGV glyph fixed at panel-center (matching
`RadarView2D._draw_ugv`'s convention) rather than rotating it. The
~6Hz throttled recompute (`terrain_relief.py:208-210`) vs. full-30Hz
marker redraw is a reasonable, intentional performance/freshness tradeoff,
not an issue.

---

## 5. Dependency / repo hygiene

| Finding | Evidence |
|---|---|
| `requirements.txt` lists `streamlit`, `pandas`, `numpy`, `plotly`, `torch`, `snntorch`, `scikit-learn` — but the PyQt6 desktop app's hard imports, `PyQt6` and `pyqtgraph`, are absent entirely. A fresh checkout following `requirements.txt` literally cannot run `avrlm_radar_app.py`. | `requirements.txt` (17 lines, full file read); `avrlm_radar_app.py:45-52` imports `PyQt6.QtWidgets`, `PyQt6.QtCore`, `PyQt6.QtGui`, `pyqtgraph` |
| Untracked `reference_images/` directory and untracked `terrain_relief.py` in working tree | `git status --short` |

---

## 6. Summary of findings by severity

| Severity | Count | Items |
|---|---|---|
| High | 0 | — |
| Medium | 4 | 2.1 (checkpoint deserialization safety), 2.2 (CWD-relative checkpoint path), 2.3 (unhandled checkpoint-load crash), §3 root cause (unseeded resample — the shared cause behind the desktop app's core complaint) |
| Low-Medium | 2 | 1's residual `aggregate_cells` guard gap, 2.4 (three-way `cluster_objects` drift) |
| Low | 4 | 2.5 (dead variable), 2.7 (latent HTML-injection pattern), 4.1 (terrain/radar marker disagreement), 4.3 (placeholder colorization range) |
| Cosmetic, resolved this audit | 1 | 4.2 (`fill_gaps` warning leak — root cause found, one-line fix identified, not applied) |

**Recommended next actions, in priority order** (none implemented by this
audit — it is a diagnosis, not a fix pass):

1. Stabilize `_scene_to_tensor`'s point resample (seed per sequence index,
   or cache across the reset window) — the single highest-leverage fix,
   since §3.2 and §3.3's symptoms and §4.4's flicker all trace back to it.
2. Wrap `fill_gaps`'s fallback `np.nanmean` (`terrain_relief.py:137`) the
   same way its main loop already does.
3. Decide and document whether `TerrainReliefView` should surface
   `density_points`, and reconcile `dashboard_pro.py`'s checkpoint loading
   with `dashboard_driving.py`'s (`weights_only=True`, `__file__`-relative
   path, wrapped in try/except).
4. Consolidate the three independent `cluster_objects`/
   `_cluster_cells_to_detections` implementations, or at minimum document
   why they must differ.
5. Add `PyQt6`/`pyqtgraph` to `requirements.txt`.

---

## 7. Addendum (2026-09-02, post-audit fix pass) — §3.4's root-cause attribution was wrong

§3.4 named `_scene_to_tensor`'s unseeded per-frame resample as the shared
root cause behind §3.2 and §3.3, and §6 ranked fixing it as the #1 next
action. That attribution was tested directly and does not hold. Measured
ablation, 160 live frames, `Engine()` in live mode:

| Variant | tracks/frame | icons ≤25m | distinct rendered IDs |
|---|---|---|---|
| As-audited | 36.6 | 22.8 | 228 |
| Seeded resample only (§6's recommendation #1) | 35.6 | 22.1 | 202 |
| Cluster only cells touched this frame | **7.6** | **5.4** | **55** |

Seeding the resample is within noise of doing nothing. The actual cause is
a **coordinate-frame error in how `engine_adapter.py` consumes the grid
engine's output**, which this audit did not identify:

`handoff.generate_2_5d_grid()` returns `_state.snapshot()` — every cell
cached since the last reset (`grid_state.py:99-102`), correct for the
event-driven persistence contract. But `_live_step()` handed that entire
snapshot to `_cluster_cells_to_detections()`, which converts cells to
world frame using only the **current** frame's `ugv_world_pos`
(`engine_adapter.py:212-234`). A `CellRecord.center_x/y` is ego-centric to
the frame that wrote it, and `driving_sequence.py`'s UGV moves 1.5 m/s, so
a cell cached 9 frames ago describes a position relative to a UGV 13.5m
behind. Every static object smeared into a ~13.5m streak of stale ghost
cells; DBSCAN shattered each streak into several clusters, each becoming a
separate track whose centroid drifts — which is also why §3.3 measured
79.4% of tracks flagged `is_dynamic` in a scene with 3-4 real movers.

The signature is a sawtooth locked to `LIVE_RESET_EVERY_N_FRAMES=10`,
which §3.2's every-20-frames sampling interval could not resolve
(detections per frame, world conversion held fixed):

```
frame:      1    2    3    4    5    6    7    8    9   10  | 11   12
full snap:  5   11   20   31   32   35   45   49   52   60  |  4   12
fresh only: 5    4    4    6    6    4    5    3    5    4  |  4    7
```

The `GridState` reset added in Pass 4/5 for step-time reasons had been
acting as the only brake on this, which is why §3.2 read the growth as
unbounded accumulation over long runtime rather than a per-window cycle.

**Fix applied**: `_live_step()` now filters `active_map` to cells whose
`last_touched_frame` equals the current frame before clustering, and
publishes that same filtered list as `Frame.live_cells`. Entirely within
`engine_adapter.py` — no grid-engine change, no change to
`generate_2_5d_grid`'s signature.

Consequent status changes to this report:

- **§3.6's open item** ("population of rendered obstacles growing over
  long runtime") — **Closed**. Measured across five 40-frame replays:
  6.4, 7.7, 8.4, 8.0, 8.5 tracks/frame (was 25.5 → 40.2, still climbing).
- **§4.4 (terrain-panel reset flicker)** — **Closed**. `Frame.live_cells`
  no longer carries an accumulation cycle, so there is nothing left to
  flicker. `terrain_relief.py` was not modified.
- **§4.2 (`fill_gaps` unwrapped `np.nanmean` fallback)** — no longer
  reachable on the live path; minimum 607 fresh cells within the panel's
  10m radius across 60 frames, never zero. Confirmed by re-running the app
  under `-W error::RuntimeWarning`: the warning fires on the pre-fix build
  and does not fire post-fix. The unwrapped call at `terrain_relief.py:137`
  is still latent and worth fixing on its own merits.
- **§3.3's threat-override concern** — deliberately **not** acted on.
  Tested: dropping the `trk.threat` render bypass or raising
  `RENDER_MIN_HITS` on top of this fix cuts ghosts only 42%→31% while
  collapsing real detection of the overtaking car from 61% to 9%. The
  existing gates are a reasonable operating point once the input is clean.
- **Incidental**: `test_integration.py::test_timing_budget`, failing at
  the time of this audit, now passes — clustering runs on ~7.6k cells
  instead of up to 58k.

**Still open, unchanged**: ~42% of rendered tracks do not correspond to a
real object, and `is_dynamic` remains ~98% true. Both trace to SNN class-2
over-prediction (measured 1,000-3,600 class-2 cells/frame vs. 27-112
class-1) reaching a jitter-limited tracker. A net-displacement
discriminator was tested and does not cleanly separate (static poles still
50% flagged moving). This needs its own pass, and per §3.6's own reasoning
it should not be met with another downstream threshold.
