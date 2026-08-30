# Quickstart — Running the Dashboards

This gets either of the two Streamlit dashboards (`dashboard_pro.py`,
`dashboard_driving.py`) running from a clean checkout of this repo. For how
they work internally, see `DESIGN.md`. For known issues/limitations, see
`Reports/AUDIT.md` (section 11 covers the dashboards specifically).

## 1. Prerequisites

There is no `requirements.txt`/`pyproject.toml` in this repo — install these
packages directly:

```
pip install streamlit pandas numpy plotly torch snntorch numba scipy scikit-learn
```

`scikit-learn` is optional: both dashboards import it in a `try/except` and
degrade gracefully (bounding boxes and noise-filtering just get disabled,
with a sidebar warning in `dashboard_driving.py`) if it's missing.

## 2. (Optional but recommended) Train a checkpoint

Neither dashboard ships a trained model. Without a checkpoint, both still
run, but the `SpikingPointNet` is randomly initialized and untrained —
predictions collapse to a single dominant class and are unstable from one
process launch to the next (see `DESIGN.md`'s "State & lifecycle" section
and `Reports/AUDIT.md` §11.2.1/§11.3.2/§11.3.3 for why). You'll see a red
"No checkpoint found" warning in the sidebar in this case — the app still
works, it's just not a meaningful demo.

To get a real checkpoint:

```
py synthetic_train_loop_v5.py
```

This writes `snn_weights.pth` to the repo root, which both dashboards look
for on startup.

## 3. Launch a dashboard

Run either command **from the repo root** (`AVRLM/`) — both dashboards
resolve `snn_weights.pth` relative to this directory:

```
streamlit run dashboard_pro.py
```

or

```
streamlit run dashboard_driving.py
```

Streamlit prints a local URL (default `http://localhost:8501`) — open it in
a browser.

**First load is slow.** Expect roughly 10-15 seconds of a blank page (just
the Streamlit toolbar, with a spinner) before any UI content appears — this
is the `torch`/`snntorch` import chain and model construction, not a hang.

If you want to run both at once, give them different ports:

```
streamlit run dashboard_pro.py --server.port 8501
streamlit run dashboard_driving.py --server.port 8502
```

## 4. What you'll see

### `dashboard_pro.py` — single-frame perception

- Sidebar: checkpoint status, "Show 10m foveation boundary" /
  "Show 3D bounding boxes" toggles, cluster distance (DBSCAN `eps`) and min
  points per object sliders.
- Click **"Generate Scene"** — builds one synthetic scene
  (`synthetic_lidar_data.build_scene()`), runs it through the SNN and the
  grid engine, and renders a 3D point cloud (Plotly) color-coded by class
  (grey = Drivable, cyan = Static Obstacle, red = Dynamic Threat), plus a
  metrics row (Speed, Active Cells, Sparsity, Energy Saved, Objects) and a
  detected-objects side panel.
- Each click resets the grid engine's cache, so this is a fresh single-frame
  snapshot every time, not an accumulating scene.

### `dashboard_driving.py` — animated multi-frame sequence

- Sidebar: the same toggles plus a noise-filter toggle, sequence length
  (10-60 frames), frame delay, and a fixed scenario description (UGV drives
  through a corridor of 5 poles, a pedestrian crosses its path, a car
  overtakes it from behind).
- Click **"Start Driving Sequence"** — steps through the scripted scenario
  frame by frame, re-rendering the 3D view, metrics (now 6 columns,
  including a Memory savings ratio), and detected objects in place after
  each frame, with `frame_delay` seconds between frames.
- Because state accumulates frame-to-frame within one run, this is the
  dashboard that actually demonstrates the grid engine's event-driven
  caching (later frames only touch cells near moving objects) and its
  memory-savings story.

## 5. Shutting down

`Ctrl+C` in the terminal running `streamlit run` stops that dashboard.
