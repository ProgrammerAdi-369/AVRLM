# Design Document — `dashboard_pro.py`

This document describes **only** `dashboard_pro.py`, in full detail: every
constant, every widget, the exact page layout top to bottom, and the exact
behavior of every function. It is meant to be readable standalone, without
needing `DESIGN.md` (which covers both dashboards together) open at the
same time. Where a behavior was verified by actually running the dashboard
in a browser, or is tracked as an open issue, this doc points to
`Reports/AUDIT.md` §11 rather than re-deriving those findings from scratch.

---

## 1. Purpose

`dashboard_pro.py` is a **single-frame, click-to-generate** perception demo.
It builds exactly one synthetic LiDAR scene, runs it once through the
spiking neural net and the variable-resolution grid engine, and renders the
result as a static 3D view plus a metrics/detection summary. It does
**not** animate, does not accumulate state across multiple generations, and
has no time dimension — every click is an independent snapshot. (The
animated, multi-frame, accumulating-state version of this same demo is
`dashboard_driving.py`, documented separately.)

It demonstrates the full pipeline described in the repo's `CLAUDE.md`:
synthetic points → `SpikingPointNet` → `handoff.generate_2_5d_grid()` (this
module's fixed integration contract) → `EdgeProfiler` metrics → a 3D
visualization with DBSCAN-derived object detection layered on top.

---

## 2. Page configuration & module-level constants

At the top of the script, in source order:

```python
st.set_page_config(page_title="DRDO Tactical UGV Perception", layout="wide")

CHECKPOINT_PATH = "snn_weights.pth"
NUM_POINTS = 8192
MAX_RANGE = 100.0

CLASS_NAMES = {0: "Drivable", 1: "Static Obstacle", 2: "Dynamic Threat"}
CLASS_COLOR = {
    "Drivable": "rgb(120,120,130)",
    "Static Obstacle": "rgb(0,170,255)",
    "Dynamic Threat": "rgb(255,30,60)",
}
BOX_COLOR = {"Static Obstacle": "rgb(0,200,255)", "Dynamic Threat": "rgb(255,60,80)"}

profiler = EdgeProfiler()
```

- `st.set_page_config(..., layout="wide")` — must be the first Streamlit
  call in the script (Streamlit's requirement); makes the page use the full
  browser width instead of the default centered ~730px column. The browser
  tab title becomes "DRDO Tactical UGV Perception".
- **`CHECKPOINT_PATH = "snn_weights.pth"` is a relative path.** It resolves
  against the **current working directory of the process that ran
  `streamlit run dashboard_pro.py`**, not against the script file's own
  directory. If you launch the dashboard from anywhere other than the repo
  root, a checkpoint that genuinely exists at the repo root will not be
  found, and the sidebar will show "No checkpoint found" with no indication
  that the real cause is the launch directory rather than a missing file.
  (Contrast with `dashboard_driving.py`, whose `CHECKPOINT_PATH` is built
  from `os.path.dirname(os.path.abspath(__file__))` and is therefore CWD-
  independent — this is a genuine difference between the two files, not
  just a formatting choice.)
- `NUM_POINTS = 8192` — the fixed number of points the scene is
  sampled/padded down or up to before being fed to `SpikingPointNet` (which
  requires a fixed-length input for its `Conv1d` layers).
- `MAX_RANGE = 100.0` — matches `grid.py`'s `OUTER_RADIUS`; used purely for
  normalizing coordinates into `[0,1]` before inference (see §5).
- `CLASS_NAMES` / `CLASS_COLOR` — the fixed 0/1/2 class-ID mapping from
  CLAUDE.md's interface contract, plus the display color for each in the 3D
  plot (grey for drivable terrain, cyan for static obstacles, red for
  dynamic/moving objects — the palette is meant to read as "cool = safe,
  warm = threat").
- `BOX_COLOR` — separate (slightly different, brighter) colors used only
  for the DBSCAN bounding-box wireframes, so boxes are visually
  distinguishable from the point markers they enclose even though they
  represent the same two classes.
- `profiler = EdgeProfiler()` — instantiated once at module scope. Since a
  Streamlit script re-executes top-to-bottom on every rerun (button click,
  slider drag, etc.), this line re-runs every time too — cheap, since
  `EdgeProfiler.__init__` just sets `MAC_ENERGY_PJ = 4.6` and
  `AC_ENERGY_PJ = 0.9` as instance attributes; it holds no state that needs
  to survive across reruns.

---

## 3. Full page layout, top to bottom (exact source order)

This is the order elements actually appear in the script, which is also
render order:

### 3.1 Sidebar (`st.sidebar.*`)

| Order | Element | Type | Default / range | Variable |
|---|---|---|---|---|
| 1 | "Perception Controls" | `st.sidebar.title` | — | — |
| 2 | "Show 10m foveation boundary" | checkbox | `True` | `show_zone_ring` |
| 3 | "Show 3D bounding boxes" | checkbox | `True` | `show_bboxes` |
| 4 | "Cluster distance (m)" | slider | 0.3 – 3.0, default 1.0, step 0.1 | `cluster_eps` |
| 5 | "Min points per object" | slider | 1 – 15, default 5, step 1 (implicit) | `cluster_min_pts` |
| 6 | horizontal rule | `st.sidebar.markdown("---")` | — | — |
| 7 | checkpoint status | conditional `st.sidebar.success` / `st.sidebar.error` | — | — |

The checkpoint status line (item 7) is:

```python
if os.path.exists(CHECKPOINT_PATH):
    st.sidebar.success(f"Loaded: {CHECKPOINT_PATH}")
else:
    st.sidebar.error("No checkpoint found -- run synthetic_train_loop_v5.py")
```

Note this is a **separate** existence check from the one inside
`load_model()` (§4) — both check the same path at roughly the same moment,
so in practice they always agree, but they are two independent `os.path.exists`
calls, not one shared result passed between them.

### 3.2 Main title block

```python
st.title("Neuromorphic Variable Resolution Perception")
st.caption("Spiking Neural Network | UGV-centric LiDAR scene | Rotate and zoom the 3D view below")
```

Plain page header, always visible regardless of button state.

### 3.3 Metrics row (5 columns, created before the button)

```python
metrics_row = st.columns(5)
fps_metric = metrics_row[0].empty()
cells_metric = metrics_row[1].empty()
sparsity_metric = metrics_row[2].empty()
energy_metric = metrics_row[3].empty()
objects_metric = metrics_row[4].empty()
```

| Column index | Placeholder variable | Label (set on click) | Empty until first click? |
|---|---|---|---|
| 0 | `fps_metric` | "Speed" | Yes |
| 1 | `cells_metric` | "Active Cells" | Yes |
| 2 | `sparsity_metric` | "Sparsity" | Yes |
| 3 | `energy_metric` | "Energy Saved" | Yes |
| 4 | `objects_metric` | "Objects" | Yes |

All five are `st.empty()` placeholders created immediately, before any
scene has been generated — they render as blank space until the button
handler populates them with `.metric(...)` calls (§8). This is why, on
first page load or before any click, this row is present in the DOM but
visually empty (five equal-width blank column slots) rather than absent
entirely.

### 3.4 Two-column body: 3D view + detected-objects panel

```python
main_col, side_col = st.columns([3, 1])
with main_col:
    map_placeholder = st.empty()
    debug_placeholder = st.caption("")
with side_col:
    st.markdown("### Detected Objects")
    detection_panel = st.empty()
```

- `main_col` : `side_col` width ratio is **3:1** — the 3D plot column is
  three times as wide as the detected-objects column.
- `map_placeholder` (inside `main_col`) — will hold either the Plotly
  figure or a `.warning()` message (§9), depending on whether the generated
  scene has any active cells.
- `debug_placeholder` — created via `st.caption("")`, i.e. it starts as an
  **empty caption string**, not a fully empty placeholder in the
  `st.empty()` sense; it's later overwritten with the fine/coarse cell-count
  caption (§10) by calling `.caption(...)` on the same object again. It sits
  directly below the plot.
- `side_col` gets a static `"### Detected Objects"` markdown header (always
  visible, never changes) followed by the `detection_panel` placeholder that
  the button handler fills with either a `.markdown()` list or an
  `.info()` fallback (§11).

### 3.5 The button and its two branches

```python
if st.button("Generate Scene", use_container_width=True):
    ...   # the entire pipeline — §5 through §11
else:
    st.info("Click 'Generate Scene' to run the perception pipeline.")
```

This is the **last** top-level statement in the file. `st.button(...)`
returns `True` only during the single script rerun triggered by the actual
click; on every other rerun (including the very first page load) it's
`False`, so the `else` branch's info box is what's visible by default. The
button spans the full width of the content column
(`use_container_width=True` — see §14 for the associated deprecation
warning).

**Important**: because everything from `handoff._state = GridState()`
onward (§5+) lives inside this `if` block, none of it executes at all until
the button is actually clicked — the metrics row, plot, and detection panel
all stay at their initial empty/placeholder state until then.

---

## 4. `load_model()` — cached model construction

```python
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ai_model = SpikingPointNet(num_steps=10).to(device)
    if os.path.exists(CHECKPOINT_PATH):
        ai_model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    ai_model.eval()
    return ai_model, device

ai_model, device = load_model()
```

- **`@st.cache_resource`**: this function's body executes exactly **once
  per running Streamlit server process** (i.e. once per `streamlit run
  dashboard_pro.py` launch — not once per browser tab, not once per
  script rerun). Every subsequent rerun (every button click, slider move,
  checkbox toggle) reuses the cached `(ai_model, device)` tuple instead of
  reconstructing the model. This is what makes the ~10-15 second cold start
  (dominated by `torch`/`snntorch` imports plus model construction) a
  one-time cost per launch rather than a per-click cost.
- **Device selection**: CUDA if available, else CPU — standard PyTorch
  pattern, no explicit override control in the UI.
- **Model construction**: `SpikingPointNet(num_steps=10)` — the `num_steps`
  argument controls how many LIF timesteps the forward pass runs (see
  `spiking_model.py`); `10` is hardcoded here, not a UI-exposed setting.
- **Checkpoint loading**: `torch.load(CHECKPOINT_PATH, map_location=device)`
  is called **only if** `os.path.exists(CHECKPOINT_PATH)` — if the file
  doesn't exist, `ai_model` simply keeps PyTorch's default random weight
  initialization and the function silently proceeds (no exception, no
  warning raised from inside `load_model()` itself — the sidebar's separate
  `os.path.exists` check, §3.1, is what surfaces this to the user).
  **Note**: unlike `dashboard_driving.py`'s equivalent loader, this call
  does **not** pass `weights_only=True` to `torch.load`. `weights_only=True`
  is a security-hardening flag added in recent PyTorch versions that
  restricts unpickling to tensor data only (rather than arbitrary Python
  objects) — omitting it here is low-risk in practice since the only
  checkpoint ever loaded is one you trained yourself locally via
  `synthetic_train_loop_v5.py`, but it is a real, verified difference in
  loader code between the two dashboard files, not merely a stylistic one.
- **`ai_model.eval()`**: puts the model in evaluation mode (disables any
  training-only behavior — though `SpikingPointNet` as defined has no
  dropout/batchnorm layers where this would matter beyond convention).
- **Consequence of no checkpoint**: since neither this file nor
  `spiking_model.py` ever calls `torch.manual_seed(...)`, a missing
  checkpoint means `ai_model`'s weights are both **untrained** and
  **unseeded** — a completely fresh random initialization every time the
  Streamlit process is launched. An untrained `Conv1d`+LIF stack with no
  learned prior tends to collapse its output toward one dominant class
  rather than a plausible mix, and because the init is unseeded, *which*
  class dominates varies from one `streamlit run` launch to the next. This
  was directly observed during the Round 3 audit (see §12 below for the
  specific numbers seen, and `Reports/AUDIT.md` §11.2.1/§11.3.2 for the full
  finding).

---

## 5. `scene_to_tensor()` — sampling & normalization

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

Called once per click as `norm_tensor, raw_sampled = scene_to_tensor(points)`
where `points` is the full scene from `build_scene()` (typically ~21,000-
25,000 raw points before this function runs).

- **Sampling to exactly `NUM_POINTS=8192`**:
  - If the scene has **8192 or more** points (the normal case for
    `build_scene()`), `NUM_POINTS` of them are chosen **without replacement**
    via `np.random.choice(n, num_points, replace=False)`. This uses NumPy's
    *global* RNG (not a locally-seeded one), so the specific subset sampled
    differs on every call/click, even for what is nominally "the same"
    scene.
  - If the scene has **fewer** than 8192 points, the index array
    `np.arange(n)` is extended to length `num_points` by **wrapping**
    (`np.pad(..., mode="wrap")`) — i.e. cycling back through the existing
    point indices from the start to fill the remainder, which duplicates
    real points rather than introducing zero/padding points. For
    `dashboard_pro.py` specifically this branch is unlikely to trigger in
    practice, since `build_scene()`'s typical output (tens of thousands of
    points) already exceeds 8192, but the code handles it defensively.
  - Both the normalized tensor **and** `pts_sampled` (the raw points later
    handed to the grid engine) are indexed from this exact same `idx` array,
    so per-point correspondence between what the model sees and what the
    grid engine sees is preserved.
- **Coordinate normalization**: `norm_coords = clip((coords + 100) / 200, 0, 1)`
  — a linear map from the full ego-centric sensor range `[-100, 100]`
  (meters, in all of X/Y/Z) to `[0, 1]`, with a final `np.clip` as a
  defensive bound (points from `build_scene()` should already be within
  ±100m by construction, so this rarely if ever actually clips anything in
  practice). Because real Z values in these synthetic scenes only span a
  few meters (not the full ±100m range), this compresses almost all height
  variation into a narrow band close to 0.5 in the normalized space — a
  property of the model's training-time normalization scheme, not something
  this dashboard file has any control over.
- **Tensor layout**: `norm_tensor` is built as shape `(4, num_points)` —
  channel-first, matching `Conv1d`'s expected `(channels, length)` layout
  (a batch dimension is added later via `.unsqueeze(0)`, see §6). Channels
  0-2 are normalized X/Y/Z; channel 3 is intensity.
- **Intensity is passed through unnormalized**: `norm_tensor[3,:] =
  pts_sampled[:,3]` — raw intensity, not scaled by `max_range` or anything
  else. This was checked during the Round 3 audit against
  `synthetic_lidar_data.py`'s actual intensity-generating code and found
  harmless: every generator (`generate_ground_plane`, `generate_pole`,
  `generate_dynamic_cluster`, `generate_boundary_stress_ring`) draws
  intensity from a sub-range within `[0.1, 0.9]`, which already sits on a
  comparable scale to the normalized `[0,1]` coordinate channels — no scale
  mismatch results in practice, even though the two channels are treated
  differently in code.
- **Return value**: `(norm_tensor, pts_sampled)` — the normalized tensor
  feeds the model; `pts_sampled` (raw, unnormalized, shape `(8192, 4)`,
  `float32`) is what's later passed as the `points` argument to
  `generate_2_5d_grid()`, satisfying that function's fixed `(N,4)` `float32`
  contract exactly.

---

## 6. The "Generate Scene" button handler — full step-by-step

Everything below runs, in this exact order, only when `st.button("Generate
Scene", ...)` evaluates `True` for that rerun.

### 6.1 Reset the grid engine's cache

```python
import handoff
handoff._state = GridState()
```

`handoff.py` keeps its event-driven cell cache in a module-level singleton
named `_state`. This line reassigns that singleton to a **brand-new**
`GridState()` — because `import handoff; handoff._state = ...` mutates
`handoff`'s own module namespace, and `generate_2_5d_grid()`/
`memory_metrics()` (imported earlier via `from handoff import
generate_2_5d_grid, memory_metrics`) look up `_state` in that same
namespace at call time, this reset is fully visible to those functions even
though the button handler only ever calls them by their imported names, not
via `handoff.generate_2_5d_grid(...)`.

**Effect**: every single click of "Generate Scene" starts the grid engine's
cache completely empty. There is no accumulation across clicks in this
dashboard — each click is a fully independent one-frame run. (Contrast with
`dashboard_driving.py`, where this same reset happens once per **sequence**
click, and the cache then accumulates normally across that sequence's many
internal frames.)

### 6.2 Generate the scene and start the timer

```python
points, labels_gt, spikes_gt = build_scene()
start_time = time.perf_counter()
```

`build_scene()` (from `synthetic_lidar_data.py`) returns a full synthetic
point cloud plus its ground-truth labels/spikes — but note **`labels_gt`
and `spikes_gt` are never used anywhere else in this file**. Only `points`
(the raw coordinates+intensity) is used going forward; the model's own
predictions (§6.4) are what actually drive the visualization, not the
scene's ground truth. This is intentional — the dashboard is demonstrating
the *model's* perception, not replaying ground truth — but worth knowing
explicitly since the ground-truth arrays are computed and then discarded.

`start_time` is captured **after** scene generation but **before**
`scene_to_tensor()`/inference/grid-engine work — this is the timer window
the "Speed" metric will later be computed from (see §6.5 and §12 for what
this window does and doesn't include).

### 6.3 Sample and normalize

```python
norm_tensor, raw_sampled = scene_to_tensor(points)
inputs = torch.tensor(norm_tensor, dtype=torch.float32).unsqueeze(0).to(device)
```

`scene_to_tensor()` is described fully in §5. `.unsqueeze(0)` adds a
size-1 batch dimension (making the shape `(1, 4, 8192)`), and `.to(device)`
moves it to CPU or CUDA depending on `load_model()`'s device choice.

### 6.4 Model inference and prediction extraction

```python
with torch.no_grad():
    spk_rec = ai_model(inputs)
total_spikes = spk_rec.sum(dim=0)
preds_np = torch.argmax(total_spikes, dim=1).squeeze().cpu().numpy()
spikes_np = total_spikes.sum(dim=1).squeeze().cpu().numpy()
```

- `torch.no_grad()` disables gradient tracking — this is pure inference,
  never training.
- **Shape trace** (verified against `spiking_model.py`'s `forward()`):
  `SpikingPointNet` runs a Python loop of `num_steps=10` iterations, each
  doing `Conv1d → Leaky` three times (`conv1/lif1 → conv2/lif2 →
  conv3/lif3`), and appends that step's `spk_out` — shape `(batch, 3,
  num_points)`, since `Conv1d` keeps a `(batch, channels, length)` layout
  throughout, and the final `conv3` has `out_channels=3` (one per class) —
  to a list. `torch.stack(spk_rec)` over that list gives the full record
  shape **`(num_steps, batch, 3, num_points)` = `(10, 1, 3, 8192)`** for
  this dashboard's inputs.
  - `total_spikes = spk_rec.sum(dim=0)` sums over the time dimension,
    leaving shape `(1, 3, 8192)`.
  - `torch.argmax(total_spikes, dim=1)` reduces over `dim=1`, which is the
    **3-class channel dimension** — this correctly picks, per point, the
    class with the most accumulated spikes over the 10 timesteps. Shape
    after argmax: `(1, 8192)`; `.squeeze()` drops the batch dim to `(8192,)`;
    `.cpu().numpy()` gives `preds_np`, a plain NumPy int array of predicted
    class IDs (0/1/2), one per point.
  - `total_spikes.sum(dim=1)` instead **sums** the same channel dimension
    (rather than reducing via argmax), giving each point's total spike
    activity summed across all 3 output channels — used as a per-point
    "spike count" proxy. Same shape/squeeze/numpy treatment as above,
    producing `spikes_np`.
- **`spikes_np`'s dtype**: derived from a `torch.float32` tensor, so
  `spikes_np` is a NumPy float array — not `uint8`/`int32` as CLAUDE.md's
  interface contract nominally specifies for `spikes`. `handoff.
  validate_inputs()` (invoked inside `generate_2_5d_grid()`, §6.5) checks
  `points`' shape and dtype and `labels`/`spikes`' *length*, but never
  checks `spikes`' dtype, so this passes validation without complaint.
  Functionally this is harmless — every downstream consumer only ever does
  `spike_sum > 0` or a bincount-weighted sum, both of which work identically
  on floats — but it is a real, verified gap between the documented
  contract and what actually flows through it at runtime. See
  `Reports/AUDIT.md` §11.2.3 for the full discussion and the currently-open
  decision (enforce the dtype, or formally loosen the documented contract).

### 6.5 Grid engine, memory metrics, and profiler

```python
active_map = generate_2_5d_grid(raw_sampled, preds_np, spikes_np)
mem_stats = memory_metrics()
perf = profiler.evaluate_efficiency(spk_rec, time.perf_counter() - start_time, 1)
```

- `generate_2_5d_grid(raw_sampled, preds_np, spikes_np)` — the module's
  fixed handoff-contract call. `raw_sampled` is the **raw, unnormalized**
  8192-point subset from `scene_to_tensor()` (§5) — shape `(8192,4)`,
  `float32` — exactly matching the contract's expected `points` shape/dtype.
  Internally this calls `validate_inputs()` then updates the (just-reset,
  §6.1) `GridState` and returns the full sparse grid as a list of
  `(cell_key, CellRecord)` pairs, assigned to `active_map`.
- `memory_metrics()` — called with no arguments, so it defaults to the same
  module-level `_state` singleton `generate_2_5d_grid()` just updated;
  returns a dict with `active_cell_count`, `estimated_sparse_bytes`,
  `naive_dense_bytes`, and `savings_ratio`. Of these, only
  `active_cell_count` is actually used later in this file (populating the
  "Active Cells" metric, §8) — the byte-count and savings-ratio fields are
  computed but not displayed anywhere in `dashboard_pro.py` (contrast with
  `dashboard_driving.py`, which does surface `savings_ratio` as its
  "Memory" metric column).
- `profiler.evaluate_efficiency(spk_rec, elapsed, 1)` — `elapsed =
  time.perf_counter() - start_time`, measured from **just before**
  `scene_to_tensor()` (§6.3) to **just after** the two calls above finish.
  This window covers: point sampling/normalization, model inference, and
  the full grid-engine update — but it does **not** include the
  `cluster_objects()` DBSCAN call that happens afterward (§6.6), nor the
  subsequent dataframe construction, nor Plotly figure construction/render.
  The batch-size argument is hardcoded to `1` (matches the actual batch
  size used throughout, since this dashboard only ever processes one scene
  at a time). `perf` ends up holding `fps`, `latency_sec`, `sparsity_pct`,
  `ac_ops`, `mac_ops_avoided`, `energy_saved_pj` (see `profiler.py`'s
  `evaluate_efficiency()` for the exact formulas — briefly: `fps = 1/elapsed`,
  sparsity = fraction of neuron-states that never fired, energy saved
  compares a per-fired-spike AC-op cost of 0.9pJ against a per-neuron-state
  MAC-op cost of 4.6pJ). **Practical consequence**: the "Speed" metric this
  dashboard displays measures model+grid latency only, not the full
  click-to-visible-render time a user actually experiences — see §12 for a
  concrete observed example of this gap.

### 6.6 Build the per-cell display dataframe

```python
records = []
for key, cell in active_map:
    r = math.sqrt(cell.center_x ** 2 + cell.center_y ** 2)
    records.append({
        "x": float(cell.center_x), "y": float(cell.center_y),
        "z": float(max(0.05, cell.elevation_max)),
        "Class": CLASS_NAMES.get(int(cell.class_id), "Drivable"),
        "size": 2.5 if cell.is_fine else 4.5,
        "radius": r,
    })
df = pd.DataFrame(records)
objects = cluster_objects(df, cluster_eps, cluster_min_pts) if show_bboxes else []
```

One row per active cell returned in `active_map`, with these exact
per-field semantics:

- **`x`, `y`**: the cell's center coordinates, taken directly from the
  `CellRecord`'s `center_x`/`center_y` (already ego-centric meters, no
  further transformation).
- **`z`**: `max(0.05, cell.elevation_max)` — the cell's max recorded height,
  **floored at 0.05m**. This floor exists purely so that flat-ground cells
  (`elevation_max` at or near 0) render as a visibly-nonzero point above the
  ground plane in the 3D scatter, rather than sitting exactly at `z=0` where
  they'd be harder to distinguish from the plot's floor/background.
- **`Class`**: looked up via `CLASS_NAMES.get(int(cell.class_id), "Drivable")`
  — note the fallback default is `"Drivable"` if `class_id` is somehow not
  0/1/2; in practice this should never trigger since `aggregate_cells()`
  upstream (in `aggregate.py`) validates class IDs and raises on anything
  outside `[0,3)` before a `CellRecord` can even be constructed with a bad
  value — the fallback here is defensive, not expected to be exercised.
- **`size`**: **2.5** for fine (`is_fine=True`, inside the 10m foveation
  radius) cells, **4.5** for coarse (outside 10m) cells — coarse cells are
  drawn visibly *larger* to stay perceptible despite each one covering 100x
  the ground area of a fine cell (50cm vs 5cm side length).
- **`radius`**: `sqrt(center_x² + center_y²)` — the cell's planar distance
  from the UGV (origin), used later for the fine/coarse debug caption (§10)
  and internally by `cluster_objects()`'s distance sort (§7).

`cluster_objects(...)` is only called at all if `show_bboxes` is checked in
the sidebar — if unchecked, `objects` is simply set to `[]` and no DBSCAN
clustering runs that click at all (a minor performance saving when the
bounding-box display feature isn't wanted).

---

## 7. `cluster_objects()` — DBSCAN-based object grouping

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
        sub = sub.copy()
        sub["cluster"] = clustering.labels_
        for cid in sub["cluster"].unique():
            if cid == -1:
                continue
            cp = sub[sub["cluster"] == cid]
            cx, cy = cp["x"].mean(), cp["y"].mean()
            objects.append({
                "class": class_label,
                "x_min": cp["x"].min() - 0.15, "x_max": cp["x"].max() + 0.15,
                "y_min": cp["y"].min() - 0.15, "y_max": cp["y"].max() + 0.15,
                "z_min": 0.0, "z_max": max(0.3, cp["z"].max()),
                "distance": math.sqrt(cx ** 2 + cy ** 2),
                "point_count": len(cp),
            })
    objects.sort(key=lambda o: o["distance"])
    return objects
```

- **Early-out**: returns `[]` immediately if `scikit-learn` isn't installed
  (`HAS_SKLEARN` is set at import time by the `try/except ImportError`
  around `from sklearn.cluster import DBSCAN`) or if the dataframe is
  empty — no bounding boxes and no object panel entries in either case, but
  the rest of the dashboard still functions.
- **Runs separately per class**, and only for **"Static Obstacle"** and
  **"Dynamic Threat"** — Drivable/ground cells are never clustered into
  "objects" regardless of how many there are.
- **Per-class DBSCAN**: `eps` and `min_samples` come straight from the
  sidebar's "Cluster distance (m)" and "Min points per object" sliders
  (§3.1) — these are genuinely live-tunable per click, not fixed constants.
  Clustering runs on the cell centers' `(x, y)` only — Z is not part of the
  clustering distance metric, so two cells at very different heights but
  close in the horizontal plane can still be grouped into one "object".
  DBSCAN's noise label (`-1`, points that don't belong to any dense cluster)
  is explicitly skipped when building `objects` — isolated single/sparse
  cells that don't form a real cluster produce no bounding box and no panel
  entry, even though they're still visible as raw point markers in the 3D
  scatter itself.
- **Per-object fields computed**:
  - `x_min/x_max/y_min/y_max`: the cluster's cell-center bounding box,
    padded by **0.15m** on all four sides (a fixed visual margin so the box
    doesn't hug the outermost points exactly).
  - `z_min = 0.0`, `z_max = max(0.3, cp["z"].max())` — boxes always start
    at ground level and extend up to at least 0.3m even if every cell in
    the cluster reported a lower elevation (so a very flat/low object still
    gets a visually-nonzero-height box).
  - `distance`: straight-line distance from the UGV (origin) to the
    cluster's `(x,y)` centroid (mean of member cells' centers, not a
    weighted mean by point count).
  - `point_count`: number of **cells** in the cluster (i.e. `len(cp)`,
    counting active grid cells, not raw LiDAR points).
- **Final sort**: `objects.sort(key=lambda o: o["distance"])` — nearest
  object first; this ordering is what both the bounding-box draw order and
  the "Detected Objects" panel's top-to-bottom order (§11) inherit.

---

## 8. Metrics row population (after inference, before rendering)

```python
fps_metric.metric("Speed", f"{perf['fps']:.0f} FPS")
cells_metric.metric("Active Cells", mem_stats['active_cell_count'])
sparsity_metric.metric("Sparsity", f"{perf['sparsity_pct']:.1f}%")
energy_metric.metric("Energy Saved", f"{perf['energy_saved_pj']/1e6:.2f} uJ")
objects_metric.metric("Objects", len(objects))
```

Each call targets the corresponding `st.empty()` placeholder from §3.3,
using Streamlit's `.metric(label, value)` widget (renders as a large bold
value with a smaller label above it). Exact formatting:

- **Speed**: FPS rounded to the nearest whole number (`{:.0f}`) — no
  decimal places shown, even though `perf['fps']` itself is a float.
- **Active Cells**: raw integer, no formatting applied (Streamlit's
  `.metric()` will still apply its own thousands-separator display for
  large integers).
- **Sparsity**: one decimal place, with a literal `%` suffix appended in
  the format string (not using `.metric()`'s built-in delta/percent
  features).
- **Energy Saved**: `perf['energy_saved_pj']` (picojoules) divided by `1e6`
  to convert to **microjoules (µJ)**, shown to two decimal places with a
  literal `uJ` suffix (note: lowercase "u" is used as an ASCII substitute
  for the µ symbol in the source, not an actual micro sign character).
- **Objects**: `len(objects)` — the count of DBSCAN-clustered objects from
  §7 (i.e. only counts as "Objects" what survived clustering with `cid !=
  -1`; noise cells are not counted here even though they may still be
  visible as raw points in the scatter).

These five calls run **unconditionally** every click — even if the scene
ends up with zero active cells (§9), the metrics row is still populated
first (all five values would just reflect that degenerate case, e.g.
"Active Cells: 0").

---

## 9. Empty-scene handling

```python
if len(df) == 0:
    map_placeholder.warning("No active cells in this scene.")
else:
    fig = go.Figure()
    ...  # §10
    map_placeholder.plotly_chart(fig, use_container_width=True)
```

If the generated scene produced zero active grid cells (a possible but
unlikely edge case — would require every sampled point to fall completely
outside the grid engine's valid range), `map_placeholder` is set to a
plain Streamlit warning box instead of attempting to build/render a Plotly
figure from an empty dataframe. The debug caption (§10) and detected-objects
panel (§11) still execute afterward regardless of which branch was taken —
they read from `df`/`objects` directly and handle the empty case
independently (the debug caption's own conditional expressions guard
against dividing/indexing into an empty `df`).

---

## 10. Figure construction (only when `len(df) > 0`)

In the exact order traces are added to the `go.Figure()`:

### 10.1 Per-class point markers

```python
for class_label, color in CLASS_COLOR.items():
    sub = df[df["Class"] == class_label]
    if sub.empty:
        continue
    fig.add_trace(go.Scatter3d(
        x=sub["x"], y=sub["y"], z=sub["z"], mode="markers",
        name=f"{class_label} ({len(sub)})",
        marker=dict(size=sub["size"], color=color, opacity=0.75),
    ))
```

Iterates `CLASS_COLOR` in its declared order — **Drivable, then Static
Obstacle, then Dynamic Threat** (Python dicts preserve insertion order) —
adding one `Scatter3d` marker trace per class that has at least one active
cell that click. Each trace's legend name includes the **live count** of
cells in that class (e.g. `"Static Obstacle (7598)"`), so the legend itself
doubles as a quick per-class count readout. `marker.size` is passed the
per-row `sub["size"]` Series directly — i.e. **fine and coarse cells within
the same class render at their own individual sizes (2.5 vs 4.5) within one
trace**, not a fixed size per trace. `opacity=0.75` is applied uniformly to
give the point cloud a slightly translucent look, useful for seeing through
denser clusters.

### 10.2 UGV marker

```python
fig.add_trace(go.Scatter3d(
    x=[0], y=[0], z=[0], mode="markers", name="UGV",
    marker=dict(size=10, color="yellow", symbol="diamond"),
))
```

Always added, unconditionally — a single yellow diamond marker at the
origin `(0,0,0)`, representing the UGV itself (which is always exactly at
the ego-centric origin by the pipeline's coordinate-frame contract).
Rendered noticeably larger (`size=10`) than any point-cloud marker (2.5-4.5)
so it's unmistakable regardless of zoom level.

### 10.3 Optional 10m foveation boundary ring

```python
if show_zone_ring:
    theta = np.linspace(0, 2 * np.pi, 100)
    fig.add_trace(go.Scatter3d(
        x=10.0 * np.cos(theta), y=10.0 * np.sin(theta), z=[0.02] * 100,
        mode="lines", name="10m Foveation Boundary",
        line=dict(color="lime", width=4, dash="dash"),
    ))
```

Only added if the "Show 10m foveation boundary" sidebar checkbox is on
(default: on). A 100-point circle of radius exactly 10.0m, drawn at a fixed
height `z=0.02` (just above the nominal ground plane, so it doesn't
z-fight/overlap with ground-level markers), styled as a dashed lime line.
**This ring is a purely visual/illustrative reference** — its radius is
hardcoded to `10.0` directly in this file, not read from `grid.py`'s
`INNER_RADIUS` constant, so it is not a rendering of actual grid data and
would silently go out of sync if that constant were ever changed elsewhere
in the module (contrast with `radial_filter.py`, which does import the
shared constant — see `Reports/AUDIT.md` §8/§10 non-blocking finding #2 for
the parallel issue there).

### 10.4 Optional bounding boxes

```python
if show_bboxes:
    for obj in objects:
        add_bbox_trace(fig, obj, BOX_COLOR.get(obj["class"], "white"))
```

Only added if "Show 3D bounding boxes" is checked (default: on) — note
`objects` itself was already computed conditionally on this same flag back
in §6.6, so if the checkbox is off, both the clustering work *and* the box
rendering are skipped. Iterates `objects` in their distance-sorted order
(§7) and calls `add_bbox_trace()` (§13) once per object, colored via
`BOX_COLOR` (falling back to `"white"` for any class not in that dict —
not expected to trigger since only Static Obstacle/Dynamic Threat objects
ever exist).

### 10.5 Layout

```python
fig.update_layout(
    scene=dict(
        xaxis=dict(title="X (m)", backgroundcolor="rgb(6,6,10)"),
        yaxis=dict(title="Y (m)", backgroundcolor="rgb(6,6,10)"),
        zaxis=dict(title="Z (m)", backgroundcolor="rgb(6,6,10)"),
        aspectmode="data",
        camera=dict(eye=dict(x=1.3, y=1.3, z=1.0)),
    ),
    paper_bgcolor="rgb(6,6,10)", plot_bgcolor="rgb(6,6,10)",
    font=dict(color="white"), margin=dict(l=0, r=0, t=0, b=0),
    legend=dict(bgcolor="rgba(0,0,0,0.5)"), height=700,
    uirevision="constant",
)
```

- **Axis titles**: "X (m)" / "Y (m)" / "Z (m)" — labels the 3D scene's axes
  in meters, matching the ego-centric coordinate convention.
- **`backgroundcolor="rgb(6,6,10)"`** on each axis pane — a near-black
  background for the 3D scene's walls, matching the overall dark theme.
- **`aspectmode="data"`**: makes the plot's X/Y/Z axes scale
  **proportionally to their actual data ranges** rather than being
  stretched independently to fill a cube — critical for a spatial scene
  like this one, where an unequal aspect ratio would visually distort
  distances (e.g. making the scene look artificially "squashed" in Z
  relative to X/Y).
- **`camera.eye = (1.3, 1.3, 1.0)`**: the default initial camera position
  (an isometric-ish angle looking down and across at the origin) — only
  actually applies on the very first render, because of `uirevision`
  (below).
- **`paper_bgcolor` / `plot_bgcolor`**: both set to the same near-black,
  covering the figure's outer background and the plotting area background
  respectively.
- **`font.color="white"`**: applies to all text in the figure (axis labels,
  legend text, hover text) by default.
- **`margin=dict(l=0,r=0,t=0,b=0)`**: zero margins on all four sides — the
  3D scene fills its container completely, with no title bar or padding
  reserved.
- **`legend.bgcolor="rgba(0,0,0,0.5)"`**: a semi-transparent black backing
  behind the legend, so it stays legible when it overlaps bright parts of
  the point cloud.
- **`height=700`**: fixed plot height in pixels (width is responsive via
  `use_container_width=True` on the `plotly_chart()` call, see §14).
- **`uirevision="constant"`**: this is what makes camera pan/zoom/rotation
  **persist across reruns** despite a brand-new `Figure` object being
  constructed and pushed into the same `map_placeholder` on every click.
  Without this, Plotly would treat each new figure as entirely unrelated to
  the previous one and reset the camera to the default `eye` position
  (§10.5 above) on every single click, making it impossible to keep a
  chosen viewing angle across multiple "Generate Scene" clicks. Because the
  value is a constant string (not tied to any per-click identifier), Plotly
  treats every rerun's figure as "the same UI" for camera-state purposes.

The finished figure is pushed to the placeholder via:

```python
map_placeholder.plotly_chart(fig, use_container_width=True)
```

---

## 11. Debug caption and Detected Objects panel

### 11.1 Debug caption (below the plot, inside `main_col`)

```python
debug_placeholder.caption(
    f"{len(df)} cells | "
    f"fine (≤10m): {int((df['radius'] <= 10.0).sum()) if len(df) else 0} | "
    f"coarse (>10m): {int((df['radius'] > 10.0).sum()) if len(df) else 0}"
)
```

Always runs (regardless of the §9 empty/non-empty branch), reusing the
`debug_placeholder` created in §3.4. Reports the total active-cell count
plus a fine/coarse split computed **directly from `df["radius"]`** (the
per-cell planar distance from the UGV computed back in §6.6) — this is an
independent recomputation from the dataframe, not a value read back from
the grid engine itself, so it serves as an implicit sanity cross-check that
the dataframe's `radius` column and the actual fine/coarse boundary agree.
`≤10m` renders as "≤10m" (the Unicode less-than-or-equal-to sign). The
`if len(df) else 0` guards exist so this line doesn't raise on an empty
dataframe (§9's warning branch) — dividing/summing over an empty Series
would otherwise still technically work in pandas, but the guard makes the
zero-case explicit rather than relying on that.

### 11.2 Detected Objects panel (in `side_col`)

```python
if objects:
    panel_rows = []
    for obj in objects:
        panel_rows.append(
            f"**{obj['class']}**  \n"
            f"Dist: {obj['distance']:.1f}m | Pts: {obj['point_count']}"
        )
    detection_panel.markdown("\n\n---\n\n".join(panel_rows))
else:
    detection_panel.info("No discrete objects detected this frame.")
```

If `objects` (§7) is non-empty, builds one Markdown block per object — bold
class name on its own line, then a "Dist: Xm | Pts: N" line below it (`\n`
after two trailing spaces is Markdown's explicit line-break syntax) — and
joins all blocks with `"\n\n---\n\n"` (a blank line, a horizontal rule, a
blank line) so each detected object appears as a visually separated card
in the sidebar-width column, in the same nearest-first order established by
§7's distance sort. If `objects` is empty (either because no clusters were
found, or because "Show 3D bounding boxes" was unchecked and clustering
never ran at all — the panel cannot distinguish between these two causes),
a plain `st.info(...)` message is shown instead.

---

## 12. Observed live behavior (Round 3 audit, no checkpoint present)

During this repo's Round 3 audit, `dashboard_pro.py` was actually launched
(`streamlit run dashboard_pro.py`, no `snn_weights.pth` present) and driven
from a real Chrome browser. Clicking "Generate Scene" produced:

- **7,718 active cells** (777 fine ≤10m, 6,941 coarse >10m).
- **7,598 of 7,718 cells (98.4%) classified "Static Obstacle"** — only 115
  "Drivable" and 5 "Dynamic Threat", a direct visual demonstration of the
  untrained-model class-collapse behavior described in §4/§6.4.
- **76 separate "Static Obstacle" objects** in the Detected Objects panel,
  most containing only a handful of points — a large number of small,
  likely-spurious clusters, consistent with clustering a near-uniformly-
  misclassified scene rather than a scene with a small number of genuine
  obstacles.
- **"Speed" metric read "2 FPS"** — illustrating §6.5's point that this
  number excludes DBSCAN clustering cost: with ~7,600 points being
  clustered, the actual click-to-fully-rendered time users experience is
  higher than "2 FPS" alone would suggest, even though the number itself
  is computed correctly for the window it's scoped to.

This is stated here as a concrete, verified example of the no-checkpoint
failure mode, not as a bug in this file's logic — every piece of code
described above behaved exactly as designed given an untrained model as
input. Full writeup: `Reports/AUDIT.md` §11.3.2.

---

## 13. `add_bbox_trace()` — wireframe box construction

```python
def add_bbox_trace(fig, obj, color):
    x0, x1, y0, y1, z0, z1 = obj["x_min"], obj["x_max"], obj["y_min"], obj["y_max"], obj["z_min"], obj["z_max"]
    edges = [
        [(x0, y0, z0), (x1, y0, z0)], [(x1, y0, z0), (x1, y1, z0)],
        [(x1, y1, z0), (x0, y1, z0)], [(x0, y1, z0), (x0, y0, z0)],
        [(x0, y0, z1), (x1, y0, z1)], [(x1, y0, z1), (x1, y1, z1)],
        [(x1, y1, z1), (x0, y1, z1)], [(x0, y1, z1), (x0, y0, z1)],
        [(x0, y0, z0), (x0, y0, z1)], [(x1, y0, z0), (x1, y0, z1)],
        [(x1, y1, z0), (x1, y1, z1)], [(x0, y1, z0), (x0, y1, z1)],
    ]
    for edge in edges:
        xs, ys, zs = zip(*edge)
        fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines",
                                    line=dict(color=color, width=5),
                                    showlegend=False, hoverinfo="skip"))
```

Draws a rectangular-box wireframe as **12 separate `Scatter3d` line
traces**, one per cube edge — 4 edges forming the bottom face (`z0`), 4
forming the top face (`z1`), and 4 vertical edges connecting corresponding
corners between the two faces. Each edge is its own trace (Plotly has no
single "3D box wireframe" primitive), all sharing the same `color` (passed
in from `BOX_COLOR`, §10.4) and `width=5`. `showlegend=False` keeps these
12 traces per object out of the legend (which would otherwise be swamped
by dozens of near-identical "edge" entries); `hoverinfo="skip"` disables
hover tooltips on the box edges themselves, so hovering over a wireframe
line doesn't produce a confusing tooltip — only the underlying point
markers (§10.1) are meant to be hoverable.

---

## 14. State, reruns, and reset behavior (specific to this file)

- **No `st.session_state` use anywhere in this file.** All state that
  matters for one click's result — `df`, `objects`, the figure, the
  metrics — lives entirely in local variables inside the `if
  st.button(...)` block, and in the `st.empty()` placeholders those
  locals get written into.
- **What *is* cached across reruns**: only `(ai_model, device)`, via
  `@st.cache_resource` on `load_model()` (§4) — this persists for the
  lifetime of the Streamlit server process, shared across every rerun and
  every browser tab connected to that process.
- **What resets on every single click**: the entire grid engine cache
  (`handoff._state = GridState()`, §6.1) — meaning `mem_stats`'s
  `active_cell_count` and everything else in this dashboard reflects
  **one scene only**, never an accumulation across multiple "Generate
  Scene" clicks. This is a deliberate difference from `dashboard_driving.py`,
  where the same reset happens once per *sequence* (not per *frame*), so
  that dashboard's cache does accumulate across its many internal frames
  within one click.
- **What's lost on a page reload or a fresh browser connection**:
  everything except the cached model — a full page reload starts from the
  pre-click `else` branch (§3.5) again, with all metrics/plot/panel back to
  their initial empty state. `dashboard_pro.py`'s single-shot nature makes
  this far less disruptive than the equivalent situation in
  `dashboard_driving.py` (where a reconnect can silently discard an
  in-progress multi-frame animation — see `Reports/AUDIT.md` §11.3.3), since
  there's no multi-step "in progress" state to lose here in the first
  place — the worst case is simply needing to click "Generate Scene" again.

---

## 15. Known quirks / limitations specific to this file

Full details and current status for all of these live in
`Reports/AUDIT.md` §11 — summarized here just enough for this document to
be complete on its own:

1. **`CHECKPOINT_PATH` is CWD-relative** (§2), unlike `dashboard_driving.py`'s
   absolute path — launching from a directory other than the repo root
   silently fails to find an existing checkpoint. (AUDIT.md §11.2's parent
   discussion, §2 above.)
2. **`torch.load(...)` omits `weights_only=True`** (§4) — present in
   `dashboard_driving.py`'s loader, absent here; low real-world risk since
   only self-trained checkpoints are ever loaded, but a genuine difference
   between the two files' loader code.
3. **Dead `sys.path.append(os.path.join(os.path.dirname(__file__),
   'AVRLM'))`** at the top of the file — appends a nonexistent
   `.../AVRLM/AVRLM` path; harmless no-op, imports resolve fine without it.
   (AUDIT.md §11.2.2.)
4. **`spikes_np` is `float`, not `uint8`/`int32`** as CLAUDE.md's interface
   contract nominally specifies, and `handoff.validate_inputs()` doesn't
   check `spikes`' dtype (§6.4) — functionally harmless, but an open
   decision on whether to enforce it or formally loosen the documented
   contract. (AUDIT.md §11.2.3.)
5. **`use_container_width=True`** (used on both the "Generate Scene" button,
   §3.5, and `plotly_chart(...)`, §10.5) triggers a Streamlit deprecation
   warning on every single call in the currently-installed Streamlit
   version, which wants `width="stretch"` instead — cosmetic today (the
   calls still work), but the stated removal date has already passed
   relative to this repo's audit session, so a future Streamlit upgrade
   could break this file outright if left unaddressed. (AUDIT.md §11.3.5.)
6. **No checkpoint ships with the repo**, and predictions are unseeded and
   unstable process-to-process without one (§4, §12) — the single biggest
   factor in whether this dashboard looks like a coherent demo or a
   near-uniform single-class scene on any given run.
7. **The "Speed" metric's timer window excludes DBSCAN clustering cost**
   (§6.5, §12) — a minor labeling-accuracy issue, not a functional bug: the
   number itself is computed correctly for what it measures, but "Speed"
   as a label implies more than that window actually covers.

None of the above are correctness bugs in the grid engine itself (`grid.py`,
`aggregate.py`, `grid_state.py`, `handoff.py`) — those are tracked
separately and in much more depth in `Reports/AUDIT.md` §§1-10.
