# Dashboard Design — `dashboard_pro.py` & `dashboard_driving.py`

This document describes the internal architecture of the two Streamlit
dashboards in this repo, in detail. For how to run them, see
`QUICKSTART.md`. For a list of verified bugs/quirks/open decisions, see
`Reports/AUDIT.md` §11 — this document describes the two dashboards as they
are, and cross-references AUDIT.md rather than re-litigating findings.

---

## 1. Overview

Both dashboards are Streamlit front-ends for the same end-to-end perception
demo: **synthetic LiDAR scene → spiking neural net (SNN) → this module's
variable-resolution 2.5D grid engine → 3D visualization + object
detection.** They exist to make the pipeline described in the project's
`CLAUDE.md` (`Raw LiDAR points -> Spiking PointNet++ -> semantic labels +
spike counts -> variable-resolution 2.5D grid engine -> dashboard`) visible
and interactive, using synthetic data in place of a real sensor/dataset.

Both dashboards integrate with this module through exactly the two
functions CLAUDE.md declares as the fixed handoff contract:

- `handoff.generate_2_5d_grid(points, labels, spikes)` — runs one frame
  through the grid engine's event-driven cache and returns the full sparse
  grid as `(cell_key, CellRecord)` pairs.
- `handoff.memory_metrics()` — active cell count and a sparse-vs-dense
  memory comparison, for the "memory savings" story.

They additionally use `profiler.EdgeProfiler.evaluate_efficiency()` for the
FPS/sparsity/energy-savings numbers (Member 5's MAC-vs-AC benchmarking
story), even though neither dashboard is "Member 5's" own code.

The two files are ~90% structurally identical — same imports, same model
loading, same tensor pipeline, same rendering approach — differing mainly in
**what scene they drive** (one static scene vs. a scripted multi-frame
sequence) and a handful of consequently-different details (checkpoint path
resolution, extra UI controls, an extra display cap for point count). Section
2 covers the shared architecture; Sections 3 and 4 cover what's specific to
each file.

---

## 2. Shared architecture

### 2.1 Imports and startup

Both files start with:

```python
import streamlit as st, pandas as pd, numpy as np, plotly.graph_objects as go
import torch, time, os, math, sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'AVRLM'))
from spiking_model import SpikingPointNet
from handoff import generate_2_5d_grid, memory_metrics
from grid_state import GridState
from profiler import EdgeProfiler
```

then a scene-source-specific import (`from synthetic_lidar_data import
build_scene` in `dashboard_pro.py`; `from driving_sequence import
build_driving_sequence` in `dashboard_driving.py`), then an optional
`sklearn.cluster.DBSCAN` import behind a `try/except ImportError` that sets
`HAS_SKLEARN`.

The `sys.path.append(..., 'AVRLM')` line is a no-op in both files: it
appends a `.../AVRLM/AVRLM` path that doesn't exist. It's harmless because
the sibling-module imports on the next lines already resolve via the
script's own directory being implicitly on `sys.path` when Streamlit runs
it. This pattern is copied from `integration_pipeline.py`, where it *does*
matter (that script expects an `AVRLM` subfolder). See `Reports/AUDIT.md`
§11.2.2.

`st.set_page_config(page_title="DRDO Tactical UGV Perception", layout="wide")`
is called once at the top — this is what makes the page use the full
browser width instead of Streamlit's default centered column.

Module-level constants shared by both files:

```python
NUM_POINTS = 8192     # points sampled per scene/frame before feeding the SNN
MAX_RANGE  = 100.0    # matches grid.py's OUTER_RADIUS — used for coord normalization
CLASS_NAMES = {0: "Drivable", 1: "Static Obstacle", 2: "Dynamic Threat"}
CLASS_COLOR = {"Drivable": "rgb(120,120,130)", "Static Obstacle": "rgb(0,170,255)", "Dynamic Threat": "rgb(255,30,60)"}
BOX_COLOR   = {"Static Obstacle": "rgb(0,200,255)", "Dynamic Threat": "rgb(255,60,80)"}
```

`profiler = EdgeProfiler()` is instantiated once at module scope (module
scope in a Streamlit script re-runs on every interaction, so this is
recreated every rerun — cheap, since `EdgeProfiler.__init__` just sets two
float constants).

### 2.2 Sidebar controls (common to both)

Built with `st.sidebar.*` calls before the main layout:

| Control | Widget | Range/default | Effect |
|---|---|---|---|
| Show 10m foveation boundary | checkbox | default True | toggles the dashed lime ring at r=10m in the 3D plot |
| Show 3D bounding boxes | checkbox | default True | toggles DBSCAN-derived bounding-box wireframes |
| Cluster distance (m) | slider | 0.3–3.0, default 1.0 | DBSCAN `eps` passed to `cluster_objects()` |
| Min points per object | slider | see per-file table below | DBSCAN `min_samples` |

`dashboard_driving.py` adds a "Filter noisy detections (display only)"
checkbox and three more sliders — see §4.

Below the controls, both files render a checkpoint-status line:

```python
if os.path.exists(CHECKPOINT_PATH):
    st.sidebar.success(f"Loaded: {CHECKPOINT_PATH}")
else:
    st.sidebar.error("No checkpoint found -- run synthetic_train_loop_v5.py")
```

Note this only checks **existence** of the file at the time this line runs
(every script rerun), not that `load_model()` actually succeeded in loading
it — in practice these are the same event since `load_model()`'s own
`os.path.exists` check runs on the same path a moment earlier, but they are
two separate checks, not one shared result.

### 2.3 Model loading — `@st.cache_resource load_model()`

```python
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ai_model = SpikingPointNet(num_steps=10).to(device)
    if os.path.exists(CHECKPOINT_PATH):
        ai_model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    ai_model.eval()
    return ai_model, device
```

`@st.cache_resource` means this function's body runs **once per Streamlit
server process** (not once per script rerun, not once per browser tab) —
Streamlit caches the returned `(model, device)` tuple keyed on the function
having no changing arguments, and reuses it across every rerun triggered by
button clicks, slider drags, etc. This is why the "10-15s cold start" in
`QUICKSTART.md` only happens once per `streamlit run` launch, not on every
interaction.

**`CHECKPOINT_PATH` differs between the two files — a real behavioral
divergence, not just a naming difference:**

- `dashboard_pro.py`: `CHECKPOINT_PATH = "snn_weights.pth"` — a **relative**
  path, resolved against the process's current working directory at the
  moment `os.path.exists`/`torch.load` run. This only finds the checkpoint
  if `streamlit run dashboard_pro.py` was launched from the repo root (as
  `QUICKSTART.md` instructs); launching from elsewhere silently fails to
  find an existing checkpoint, with no different error message — it just
  looks like no checkpoint exists.
- `dashboard_driving.py`: `CHECKPOINT_PATH =
  os.path.join(os.path.dirname(os.path.abspath(__file__)), "snn_weights.pth")`
  — an **absolute** path derived from the script file's own location,
  independent of the current working directory.

`dashboard_driving.py`'s `load_model()` also returns a third value,
`checkpoint_loaded: bool`, and calls `torch.load(..., weights_only=True)`;
`dashboard_pro.py`'s does not track that boolean and omits `weights_only`
(a security-hardening flag that restricts unpickling to tensor data only —
harmless here since both dashboards only ever load their own locally-trained
checkpoint, but worth knowing the two files aren't identical on this point).

**No-checkpoint fallback**: if `CHECKPOINT_PATH` doesn't exist, `ai_model`
is left with PyTorch's default random initialization and no
`torch.manual_seed()` is called anywhere in either file or in
`spiking_model.py`. Since `SpikingPointNet` is a from-scratch `Conv1d` +
`snntorch.Leaky` LIF stack with no pretrained prior, an untrained forward
pass tends to collapse to one dominant output class rather than a plausible
mix — and because the init is unseeded, *which* class dominates differs
between separate `streamlit run` process launches. This was directly
observed during the Round 3 audit: one process run classified ~98% of cells
"Static Obstacle", a separate run on identical code classified the majority
"Dynamic Threat" instead. See `Reports/AUDIT.md` §11.2.1, §11.3.2, §11.3.3.

### 2.4 `scene_to_tensor()` — point sampling and normalization

Identical in both files:

```python
def scene_to_tensor(points, num_points=NUM_POINTS, max_range=MAX_RANGE):
    n = points.shape[0]
    if n >= num_points:
        idx = np.random.choice(n, num_points, replace=False)
    else:
        idx = np.pad(np.arange(n), (0, num_points - n), mode="wrap")
    pts_sampled = points[idx]

    coords = pts_sampled[:, :3]
    norm_coords = np.clip((coords + max_range) / (2.0 * max_range), 0.0, 1.0)
    norm_tensor = np.zeros((4, num_points), dtype=np.float32)
    norm_tensor[:3, :] = norm_coords.T
    norm_tensor[3, :] = pts_sampled[:, 3]
    return norm_tensor, pts_sampled
```

- **Sampling**: the raw scene (tens of thousands of points, see §3/§4) is
  reduced to exactly `NUM_POINTS=8192` points before feeding the SNN
  (`Conv1d` needs a fixed input length). If the scene has more points than
  8192, a random subset is drawn without replacement
  (`np.random.choice(..., replace=False)`) — note this uses NumPy's global
  RNG, unseeded, so which points are sampled differs every call. If the
  scene has *fewer* than 8192 points (possible for a sparse `driving_sequence`
  frame with few objects in range), the index array is padded by **wrapping**
  (`np.pad(..., mode="wrap")`) — i.e. cycling back through the existing
  points to fill the length, which duplicates points rather than zero-
  padding. Both `pts_sampled` (the raw, unnormalized points used later for
  the grid engine) and the normalized tensor are derived from the exact same
  sampled index set, so they stay aligned per-point.
- **Coordinate normalization**: `(coords + max_range) / (2*max_range)`
  linearly maps ego-centric X/Y/Z from `[-100, 100]` (the full sensor range,
  `MAX_RANGE`) to `[0, 1]`, then clips to that range (defends against any
  point technically outside ±100m, though `prefilter_mask`/`radial_filter`
  elsewhere in the pipeline already exclude those). Because real Z values
  span only a few meters (not ±100m), this normalization compresses height
  variation into a very narrow band near 0.5 — a modeling choice inherited
  as-is from the training pipeline, not something the dashboards control.
- **Intensity**: `norm_tensor[3, :] = pts_sampled[:, 3]` passes intensity
  through **unnormalized**, unlike X/Y/Z. This was checked during the Round
  3 audit and found to be harmless: every intensity generator in
  `synthetic_lidar_data.py` (`generate_ground_plane`, `generate_pole`,
  `generate_dynamic_cluster`, `generate_boundary_stress_ring`) already draws
  from a sub-range of `[0.1, 0.9]`, so intensity and the normalized
  coordinate channels are already on a compatible `[0,1]`-ish scale — no
  separate normalization step was needed. See `Reports/AUDIT.md` §11.2.4.

### 2.5 Inference → grid pipeline

Both files, inside their respective button handler, run:

```python
import handoff
handoff._state = GridState()          # reset the event-driven cache for this run

points, labels_gt, spikes_gt = build_scene()          # or one frame of build_driving_sequence()
norm_tensor, raw_sampled = scene_to_tensor(points)
inputs = torch.tensor(norm_tensor, dtype=torch.float32).unsqueeze(0).to(device)

with torch.no_grad():
    spk_rec = ai_model(inputs)
total_spikes = spk_rec.sum(dim=0)
preds_np  = torch.argmax(total_spikes, dim=1).squeeze().cpu().numpy()
spikes_np = total_spikes.sum(dim=1).squeeze().cpu().numpy()

active_map = generate_2_5d_grid(raw_sampled, preds_np, spikes_np)
mem_stats  = memory_metrics()
perf = profiler.evaluate_efficiency(spk_rec, time.perf_counter() - start_time, 1)
```

Points worth calling out in detail:

- **`handoff._state = GridState()`**: this reassigns the *module-level*
  `_state` singleton inside `handoff.py` (see `handoff.py`'s own docstring:
  "event-driven persistence is kept in a module-level `GridState`
  singleton"). Because `import handoff; handoff._state = ...` mutates the
  module's own global namespace, and `generate_2_5d_grid()`/`memory_metrics()`
  (imported directly via `from handoff import ...`) look up `_state` in that
  same namespace at call time, this reset is visible to those functions even
  though they were imported by name rather than accessed via `handoff.`.
  **Effect**: every "Generate Scene" click (`dashboard_pro.py`) or every
  "Start Driving Sequence" click (`dashboard_driving.py`) starts the grid
  cache completely empty. Within a single driving-sequence run, the cache
  then accumulates normally across that run's frames (this is what
  `dashboard_driving.py`'s Memory metric and growing cell count demonstrate)
  — but a fresh click always starts over, it never continues accumulating
  across separate clicks.
- **Tensor shape trace**: `SpikingPointNet.forward()` (in `spiking_model.py`)
  runs a `num_steps`-iteration loop of `Conv1d → Leaky` three times per
  step, appending each step's `spk_out` (shape `(batch, 3, num_points)`,
  since `Conv1d` keeps `(batch, channels, length)`) to a list, then
  `torch.stack(spk_rec)` gives the full record shape
  **`(num_steps, batch, 3, num_points)`**. `total_spikes =
  spk_rec.sum(dim=0)` sums over time, leaving `(batch, 3, num_points)`.
  `torch.argmax(total_spikes, dim=1)` reduces over `dim=1`, which is the
  **3-class channel dimension** — correctly picking the winning class per
  point, not per batch or per point-index. `total_spikes.sum(dim=1)` sums
  the same channel dimension instead of reducing it, giving a per-point
  "total activity across all 3 output channels" scalar used as the `spikes`
  count. Both are `.squeeze()`d to drop the size-1 batch dimension before
  `.cpu().numpy()`. This was traced end-to-end during the Round 3 audit and
  confirmed correct (`Reports/AUDIT.md` §11.2.5).
- **`spikes_np` dtype**: this is a `float32`-derived NumPy array, not
  `uint8`/`int32` as CLAUDE.md's interface contract nominally specifies for
  `spikes`. `handoff.validate_inputs()` (called inside
  `generate_2_5d_grid()`) checks `points` shape/dtype and `labels`/`spikes`
  *length*, but not `spikes` dtype — so this passes validation. Functionally
  harmless (every consumer just does `spike_sum > 0` or bincount-weighted
  sums, which work identically on floats), but it's a soft, currently-
  undecided gap between the documented contract and what actually flows
  through it. See `Reports/AUDIT.md` §11.2.3.
- **`raw_sampled` as `points`**: the *raw* (unnormalized) sampled points
  from `scene_to_tensor()` — shape `(8192, 4)`, `float32`, matching
  `generate_2_5d_grid`'s contract exactly — are what's actually passed to
  the grid engine, not the normalized tensor used for inference. The model
  only ever sees normalized coordinates; the grid engine only ever sees raw
  ego-centric meters, which is what it needs for its 10m/50cm radial logic.
- **`profiler.evaluate_efficiency(spk_rec, elapsed, 1)`**: `elapsed` is
  `time.perf_counter() - start_time`, where `start_time` was captured
  *before* `scene_to_tensor()` and stops being measured right after
  `generate_2_5d_grid()`/`memory_metrics()` return — i.e. this timer covers
  sampling + normalization + SNN inference + the grid engine update, but
  **not** the `cluster_objects()` DBSCAN call that happens afterward. The
  dashboard's "Speed" metric is therefore a measure of model+grid latency,
  not full click-to-render latency — worth knowing if the displayed FPS
  number seems inconsistent with how long the UI visibly takes to update.
  See `Reports/AUDIT.md` §11.3.2.

### 2.6 `cluster_objects()` — DBSCAN grouping for the object panel

```python
def cluster_objects(df, eps, min_samples):
    objects = []
    if not HAS_SKLEARN or df.empty:
        return objects
    for class_label in ["Static Obstacle", "Dynamic Threat"]:
        sub = df[df["Class"] == class_label]
        if len(sub) < min_samples:
            continue
        coords = sub[["x", "y"]].to_numpy()
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
        ...
```

Runs **separately per class** (Static Obstacle, Dynamic Threat — never
Drivable) over each active cell's `(x, y)` center, using the sidebar's
cluster-distance (`eps`) and min-points sliders. For each non-noise DBSCAN
cluster (`cid != -1`), it computes an axis-aligned bounding box padded by
0.15m in X/Y, a Z range from 0 to `max(0.3, max cell elevation in cluster)`,
the cluster's centroid distance from the UGV, and its point count. The
resulting `objects` list is sorted by distance (closest first) and drives
both the bounding-box wireframes (`add_bbox_trace()`) and the "Detected
Objects" side panel text.

`dashboard_driving.py`'s version additionally returns the concatenated
subset of *clustered* (non-noise) cells as a second value, used to
optionally render only "real" detections when the noise-filter toggle is
on — see §4.

`add_bbox_trace()` (identical in both files) draws a wireframe box as 12
separate `go.Scatter3d` line traces (one per cube edge), each with
`showlegend=False, hoverinfo="skip"` so they don't clutter the legend or
tooltips.

### 2.7 Rendering

Both files build a single `plotly.graph_objects.Figure` per render:

1. One `go.Scatter3d` marker trace per class present in the active-cell
   dataframe (`CLASS_COLOR` for color, per-file marker size logic — see §3/
   §4), each named with its point count for the legend (e.g. "Static
   Obstacle (7598)").
2. A single-point yellow diamond trace for the UGV at the origin.
3. If "Show 10m foveation boundary" is checked: a dashed lime circle
   (`theta = linspace(0, 2π, 100)`, radius 10, at `z=0.02` so it sits just
   above the ground plane) — a purely visual reference for where the
   grid engine's fine/coarse boundary is, not derived from the actual grid
   data.
4. If "Show 3D bounding boxes" is checked: one `add_bbox_trace()` call per
   detected object.

Layout: dark theme (`rgb(6,6,10)` background/scene panes), `aspectmode="data"`
(so X/Y/Z scale proportionally rather than being stretched to fill the plot
box — important for a spatial scene), and **`uirevision="constant"`** — this
is what prevents Plotly from resetting the camera's pan/zoom/rotation every
time a new figure object replaces the old one in the same `st.empty()`
placeholder; without it, every click/frame would snap the camera back to
its default angle.

The figure is pushed into a pre-created `st.empty()` placeholder
(`map_placeholder.plotly_chart(fig, use_container_width=True)`) rather than
called inline, so repeated updates replace content in place instead of
appending new chart elements to the page. `use_container_width=True`
currently prints a Streamlit deprecation warning on every call (the
installed version wants `width="stretch"` instead) — cosmetic today, flagged
in `Reports/AUDIT.md` §11.3.5.

---

## 3. `dashboard_pro.py` specifics

- **Scene source**: `synthetic_lidar_data.build_scene()` — one static
  synthetic scene (ground plane + poles + a dynamic cluster), ~21,000-25,000
  raw points before sampling.
- **Flow**: entirely gated behind `if st.button("Generate Scene", ...)`.
  Nothing runs until clicked; between clicks the page shows a static
  `st.info("Click 'Generate Scene' to run the perception pipeline.")`
  message in the `else` branch.
- **Metrics row**: 5 columns — Speed (FPS), Active Cells, Sparsity,
  Energy Saved (µJ, converted from the profiler's picojoules), Objects.
- **Layout**: `main_col, side_col = st.columns([3, 1])` — a wide 3D-plot
  column and a narrower "Detected Objects" column, both declared once at
  module scope (so their `st.empty()` placeholders persist across reruns).
- **Marker sizing**: fine (`is_fine=True`, i.e. inside 10m) cells render at
  marker size 2.5, coarse cells at 4.5 — coarse cells are drawn slightly
  larger to stay visible despite representing more physical area per cell.
- **Debug caption**: below the plot, a small caption reports
  `"{N} cells | fine (≤10m): {n1} | coarse (>10m): {n2}"`, computed directly
  from the rendered dataframe's `radius` column (not from a separate
  source), as a quick sanity check against the sidebar's boundary-ring
  toggle.
- **No `st.session_state` use, no accumulation across clicks** — each click
  is an independent snapshot (see §2.5's note on `handoff._state` reset).

## 4. `dashboard_driving.py` specifics

- **Scene source**: `driving_sequence.build_driving_sequence(n_frames=...)`
  — see below for how the scenario itself is constructed. Unlike
  `dashboard_pro.py`, this is a *list of frames*, generated once up front
  when the button is clicked, then iterated.
- **Extra sidebar controls**:
  - "Filter noisy detections (display only)" checkbox (default True) — when
    on, only cells that DBSCAN placed inside a real cluster (not "noise",
    label `-1`) are drawn for Static Obstacle/Dynamic Threat classes; the
    underlying grid-cell count and model predictions are unaffected, only
    what's *drawn* changes (the code comment in-file is explicit about
    this).
  - Sequence length (frames): 10–60, default 40.
  - Frame delay (s): 0.0–1.0, default 0.2 — `time.sleep(frame_delay)` at the
    end of each loop iteration.
  - Min points per object slider range is 1–10 here (vs. 1–15 in
    `dashboard_pro.py`), default 2 (vs. 5) — tuned lower since a single
    driving-sequence frame typically has far fewer points per object than
    the static demo scene.
- **`DRIVABLE_DISPLAY_CAP = 6000`**: Drivable-class cells (ground) dominate
  the cell count every frame but carry little visual information beyond
  showing terrain extent. Re-sending tens of thousands of ground points to
  Plotly/the browser every `frame_delay` seconds is real per-frame cost and
  a likely contributor to visible stutter, so if the Drivable subset exceeds
  6000 points it's randomly downsampled (`sub.sample(DRIVABLE_DISPLAY_CAP,
  random_state=frame_idx)` — seeded by frame index so the *same* subsample
  is picked if a frame were somehow re-rendered, though in practice each
  frame index is only rendered once per run). Static/Dynamic classes are
  never capped.
- **Frame loop structure**: a single `with torch.no_grad(): for frame_idx,
  (...) in enumerate(sequence):` block runs the full per-frame pipeline
  (§2.5) plus clustering (§2.6) plus rendering (§2.7) sequentially, updating
  the same set of `st.empty()` placeholders each iteration — this is what
  produces the "live animation" effect within a single Streamlit script run
  (Streamlit does not rerun the script per frame here; the loop itself lives
  inside one button-triggered execution).
- **Zero-active-cell frames**: if a frame produces no active cells at all
  (`len(df) == 0` — possible early/late in the sequence when nothing is in
  range), the code deliberately does **not** touch `map_placeholder` at all,
  leaving the previous frame's chart on screen; only the debug caption and
  detection panel update to say "0 active cells (no change this frame)".
  The in-file comment explains this avoids a chart flash/disappear effect
  that swapping to a `.warning()` box and back would cause.
- **No explicit Plotly `key`**: the in-file comment notes an earlier version
  passed an explicit `key` to `plotly_chart()` and hit
  `StreamlitDuplicateElementKey` on frame 2, because Streamlit requires keys
  to be unique across an entire script run, not just per-iteration-of-a-loop.
  The fix was to rely on `map_placeholder` (the same `st.empty()` slot every
  iteration) for in-place replacement and `uirevision="constant"` for camera
  stability, and drop the key entirely.
- **Marker sizing**: fine cells at size 2.0, coarse at 4.0 (slightly smaller
  than `dashboard_pro.py`'s 2.5/4.5 — a minor cosmetic difference).
- **Metrics row**: 6 columns — Speed, Grid Cells, **Memory** (the sparse-
  vs-dense savings ratio from `memory_metrics()`, not shown at all in
  `dashboard_pro.py`), Sparsity, Energy Saved, Objects.
- **Header extras**: a frame counter (`"{frame_idx+1}/{n_frames}"`) and the
  UGV's world-frame position (`"x={:.1f}m, y={:.1f}m"`, from
  `driving_sequence.py`'s `get_ugv_position()`) are shown top-right, updated
  every frame.
- **Completion**: after the loop finishes normally, a
  `progress_placeholder.success("Sequence complete.")` message appears. (A
  WebSocket reconnect *during* the run can interrupt this — see §5.2.)

### 4.1 `driving_sequence.py` — the scripted scenario

Built on top of `synthetic_lidar_data.py`'s primitives
(`generate_ground_plane`, `generate_pole`, `generate_dynamic_cluster`), but
composed into a scene with actual narrative structure across frames, all in
world coordinates that get transformed into the UGV's ego-centric frame per
frame:

- **UGV**: moves along +X at a constant `UGV_SPEED = 1.5` m/s;
  `get_ugv_position(frame_idx) = (1.5 * frame_idx, 0.0)`.
- **Static poles**: 5 fixed world positions,
  `[(20,3), (35,-4), (55,2.5), (70,-3.5), (90,4)]` — each frame, a pole is
  only included if its ego-frame position (`pole_world - ugv_pos`) is within
  ±100m in both X and Y, matching the module's own sensor range.
- **Pedestrian** (dynamic cluster): starts at world `(40, -15)`, moves at
  `(0.0, 1.2)` m/s — a "crossing the road" trajectory that the UGV's own
  forward motion sweeps across its relative field of view.
- **Overtaking car** (a second, larger/faster dynamic cluster): starts
  behind the UGV at world `(-15, -2)`, moves at `(3.5, 0.0)` m/s — faster
  than the UGV, so it closes in and passes.
- Each frame's points/labels/spikes are the concatenation of the ground
  plane plus every in-range pole/pedestrian/car part for that frame,
  built with a single shared `np.random.default_rng(seed=2026)` instance
  threaded through every generator call (so a full `build_driving_sequence()`
  call is reproducible run-to-run, independent of the model's own unseeded
  randomness described in §2.3).
- **Deliberately no `__main__` self-test block.** The module's own
  docstring explains why: an earlier version had one, and its print
  statements were observed firing on every `streamlit run
  dashboard_driving.py` launch even before any button was clicked, because
  Streamlit's script-execution/reload model doesn't reliably respect a
  `if __name__ == "__main__":` guard the way plain `python file.py` does.
  The recommended way to test this module directly is a separate throwaway
  script, not a guarded block inside the module itself.

---

## 5. State & lifecycle

A clear model of what persists, what resets, and when, across the three
different "lifetimes" a Streamlit app has:

| Layer | Lifetime | Mechanism |
|---|---|---|
| `ai_model`, `device` (+ `checkpoint_loaded` in `dashboard_driving.py`) | Once per **server process** (i.e. once per `streamlit run` launch, shared across all browser tabs/reruns connected to it) | `@st.cache_resource` |
| `handoff._state` (the `GridState` cache) | Reset at the **start of every button click**; accumulates across frames *within* that one click's execution (meaningful only in `dashboard_driving.py`'s multi-frame loop) | Explicit `handoff._state = GridState()` reassignment |
| Sidebar widget values (sliders, checkboxes) | Persist across reruns via Streamlit's normal widget state, reset to their coded defaults on a fresh page load | Streamlit's built-in widget state (not `st.session_state` used explicitly anywhere in either file) |
| Rendered chart / metrics / detected-objects panel | Held in `st.empty()` placeholders, updated in place; **not** stored anywhere durable | Plain Python locals inside the button handler — nothing is written to `st.session_state` |

**Neither file uses `st.session_state` at all.** This matters for one
specific, verified failure mode:

### 5.1 WebSocket-reconnect state loss (verified during the Round 3 audit)

Because none of a running or just-completed sequence's state — frame index,
accumulated `GridState`, the rendered figure, the metrics — lives in
`st.session_state`, it all only exists as local variables inside one
in-progress script execution plus whatever's currently sitting in the
`st.empty()` placeholders. If the browser's WebSocket connection to the
Streamlit server drops and reconnects — which can happen from something as
mundane as a laptop sleeping, a tab being backgrounded long enough to be
throttled, or a network hiccup, not just from unusual conditions — Streamlit
starts a **fresh script execution** for the new connection. Since the button
condition (`if st.button(...)`) is only true during the run where the click
itself occurred, this fresh execution renders the initial pre-click state:
no frame counter, no metrics, no chart, and (if the reconnect happened
mid-sequence) no "Sequence complete." message either, since that line is
only reached by the interrupted run, not the new one.

This was directly observed while auditing `dashboard_driving.py`: mid-way
through an animated run, a reconnect (visible server-side as repeated
`tornado.websocket.WebSocketClosedError` entries) caused the page to revert
entirely to its "Start Driving Sequence" initial screen with no explanation
shown to the user. See `Reports/AUDIT.md` §11.3.3 for the full writeup and
the open decision (accept as a known `st.empty()`-based-live-update
limitation, or move key state into `st.session_state` so a reconnect
mid-run doesn't discard the run).

---

## 6. Known limitations / open items

This section intentionally stays short — `Reports/AUDIT.md` §11.4 is the
authoritative, up-to-date list. As of the Round 3 audit, the open
non-blocking items relevant to these two dashboards are:

1. Sequence state loss on WebSocket reconnect mid-animation (§5.1 above;
   AUDIT.md §11.3.3).
2. No trained checkpoint ships with the repo — predictions are unseeded and
   unstable run-to-run without one (§2.3 above; AUDIT.md §11.2.1, §11.3.2,
   §11.3.3).
3. `use_container_width` deprecation warning on every `plotly_chart`/button
   call (§2.7 above; AUDIT.md §11.3.5) — not yet migrated to `width="stretch"`.
4. Dead `sys.path.append(.../AVRLM)` in both files (§2.1 above; AUDIT.md
   §11.2.2) — harmless, not yet removed.
5. `spikes` arrives as `float`, not `uint8`/`int32` as CLAUDE.md's contract
   nominally specifies, and `handoff.validate_inputs()` doesn't check its
   dtype (§2.5 above; AUDIT.md §11.2.3) — undecided whether to enforce or
   formally loosen the documented contract.
6. The "Speed" metric's timer window excludes DBSCAN clustering cost
   (§2.5 above; AUDIT.md §11.3.2) — minor labeling-accuracy issue.

None of these are correctness bugs in the grid engine itself — see
`Reports/AUDIT.md` §§1-10 for that module's own (separately-tracked) audit
history.
