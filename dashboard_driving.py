import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import torch
import time
import os
import math
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'AVRLM'))
from spiking_model import SpikingPointNet
from handoff import generate_2_5d_grid, memory_metrics
from grid_state import GridState
from profiler import EdgeProfiler
from driving_sequence import build_driving_sequence

try:
    from sklearn.cluster import DBSCAN
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

st.set_page_config(page_title="DRDO Tactical UGV Perception", layout="wide")

CHECKPOINT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snn_weights.pth")
NUM_POINTS = 8192
MAX_RANGE = 100.0

# Ground/"Drivable" cells dominate the count (tens of thousands) but carry
# little visual information beyond showing the terrain extent. Rebuilding
# and re-sending that many WebGL points every frame_delay seconds is real
# work for Plotly + the browser, and is a likely contributor to visible
# per-frame stutter -- capping it keeps the redraw light without touching
# the actual Static/Dynamic detections.
DRIVABLE_DISPLAY_CAP = 6000

CLASS_NAMES = {0: "Drivable", 1: "Static Obstacle", 2: "Dynamic Threat"}
CLASS_COLOR = {
    "Drivable": "rgb(120,120,130)",
    "Static Obstacle": "rgb(0,170,255)",
    "Dynamic Threat": "rgb(255,30,60)",
}
BOX_COLOR = {"Static Obstacle": "rgb(0,200,255)", "Dynamic Threat": "rgb(255,60,80)"}

profiler = EdgeProfiler()

st.sidebar.title("Perception Controls")
show_zone_ring = st.sidebar.checkbox("Show 10m foveation boundary", value=True)
show_bboxes = st.sidebar.checkbox("Show 3D bounding boxes", value=True)
filter_noise = st.sidebar.checkbox(
    "Filter noisy detections (display only)", value=True,
    help="Hides isolated Static/Dynamic cells that don't form a real DBSCAN "
         "cluster. This only affects what's drawn -- it doesn't change the "
         "model's raw predictions or the Grid Cells count above.",
)
n_frames = st.sidebar.slider("Sequence length (frames)", 10, 60, 40)
cluster_eps = st.sidebar.slider("Cluster distance (m)", 0.3, 3.0, 1.0, 0.1)
cluster_min_pts = st.sidebar.slider("Min points per object", 1, 10, 2)
frame_delay = st.sidebar.slider("Frame delay (s)", 0.0, 1.0, 0.2, 0.05)

if not HAS_SKLEARN:
    st.sidebar.warning("scikit-learn not found -- bounding boxes and noise filtering are disabled.")

st.sidebar.markdown("---")
st.sidebar.caption("Scenario")
st.sidebar.markdown(
    "- UGV drives forward through a corridor of 5 static poles\n"
    "- A pedestrian crosses its path from the side\n"
    "- A vehicle overtakes it from behind at higher speed"
)

st.sidebar.markdown("---")
if os.path.exists(CHECKPOINT_PATH):
    st.sidebar.success(f"Loaded: {CHECKPOINT_PATH}")
else:
    st.sidebar.error("No checkpoint found -- run synthetic_train_loop_v5.py")

header_col1, header_col2 = st.columns([3, 1])
with header_col1:
    st.title("Neuromorphic Variable Resolution Perception")
    st.caption("Live driving sequence | UGV-centric ego frame | Static + dynamic object tracking")
with header_col2:
    frame_counter_placeholder = st.empty()
    ugv_pos_placeholder = st.empty()

metrics_row = st.columns(6)
fps_metric = metrics_row[0].empty()
cells_metric = metrics_row[1].empty()
savings_metric = metrics_row[2].empty()
sparsity_metric = metrics_row[3].empty()
energy_metric = metrics_row[4].empty()
objects_metric = metrics_row[5].empty()

main_col, side_col = st.columns([3, 1])
with main_col:
    map_placeholder = st.empty()
    debug_placeholder = st.caption("")
with side_col:
    st.markdown("### Detected Objects")
    detection_panel = st.empty()

progress_placeholder = st.empty()


@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ai_model = SpikingPointNet(num_steps=10).to(device)
    checkpoint_loaded = False
    if os.path.exists(CHECKPOINT_PATH):
        ai_model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True))
        checkpoint_loaded = True
    ai_model.eval()
    return ai_model, device, checkpoint_loaded


ai_model, device, checkpoint_loaded = load_model()


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


def cluster_objects(df, eps, min_samples):
    """
    Runs DBSCAN per Static/Dynamic class to group raw predicted cells into
    discrete objects (for bounding boxes + the side panel), and also hands
    back the subset of points that actually belong to a cluster -- i.e.
    with DBSCAN "noise" (label -1: isolated cells that don't form a real
    object) dropped. Returns (objects, clustered_points_df). The second
    value is None if clustering didn't run at all (no sklearn / empty df).
    """
    objects = []
    clustered_frames = []
    if not HAS_SKLEARN or df.empty:
        return objects, None
    for class_label in ["Static Obstacle", "Dynamic Threat"]:
        sub = df[df["Class"] == class_label]
        if len(sub) < min_samples:
            continue
        coords = sub[["x", "y"]].to_numpy()
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
        sub = sub.copy()
        sub["cluster"] = clustering.labels_
        clustered_frames.append(sub[sub["cluster"] != -1])
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
    clustered_points = pd.concat(clustered_frames, ignore_index=True) if clustered_frames else None
    return objects, clustered_points


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


if st.button("Start Driving Sequence", use_container_width=True):
    import handoff
    handoff._state = GridState()
    sequence = build_driving_sequence(n_frames=n_frames)
    progress_bar = progress_placeholder.progress(0)

    with torch.no_grad():
        for frame_idx, (points, labels, spikes, ugv_world_pos) in enumerate(sequence):
            start_time = time.perf_counter()

            norm_tensor, raw_sampled = scene_to_tensor(points)
            inputs = torch.tensor(norm_tensor, dtype=torch.float32).unsqueeze(0).to(device)

            spk_rec = ai_model(inputs)
            total_spikes = spk_rec.sum(dim=0)
            preds_np = torch.argmax(total_spikes, dim=1).squeeze().cpu().numpy()
            spikes_np = total_spikes.sum(dim=1).squeeze().cpu().numpy()

            active_map = generate_2_5d_grid(raw_sampled, preds_np, spikes_np)
            mem_stats = memory_metrics()
            perf = profiler.evaluate_efficiency(spk_rec, time.perf_counter() - start_time, 1)

            records = []
            for key, cell in active_map:
                r = math.sqrt(cell.center_x ** 2 + cell.center_y ** 2)
                records.append({
                    "x": float(cell.center_x), "y": float(cell.center_y),
                    "z": float(max(0.05, cell.elevation_max)),
                    "Class": CLASS_NAMES.get(int(cell.class_id), "Drivable"),
                    "size": 2.0 if cell.is_fine else 4.0,
                    "radius": r,
                })
            df = pd.DataFrame(records)
            need_clusters = HAS_SKLEARN and (show_bboxes or filter_noise) and not df.empty
            if need_clusters:
                objects, clustered_points = cluster_objects(df, cluster_eps, cluster_min_pts)
            else:
                objects, clustered_points = [], None

            frame_counter_placeholder.metric("Frame", f"{frame_idx + 1}/{n_frames}")
            ugv_pos_placeholder.caption(f"UGV world position: x={ugv_world_pos[0]:.1f}m, y={ugv_world_pos[1]:.1f}m")

            fps_metric.metric("Speed", f"{perf['fps']:.1f} FPS")
            cells_metric.metric("Grid Cells", mem_stats['active_cell_count'])
            savings_metric.metric("Memory", f"{mem_stats['savings_ratio']:.1f}x")
            sparsity_metric.metric("Sparsity", f"{perf['sparsity_pct']:.1f}%")
            energy_metric.metric("Energy Saved", f"{perf['energy_saved_pj']/1e6:.2f} uJ")
            objects_metric.metric("Objects", len(objects))

            if len(df) == 0:
                # Deliberately NOT touching map_placeholder here. Swapping it
                # to a .warning() box and back every time a frame has zero
                # active cells is exactly what produces a chart that appears
                # to flash/disappear -- better to just leave the previous
                # frame on screen (nothing changed, so nothing to redraw).
                debug_placeholder.caption(f"Frame {frame_idx} | 0 active cells (no change this frame)")
                detection_panel.info("No discrete objects detected this frame.")
                progress_bar.progress((frame_idx + 1) / n_frames)
                time.sleep(frame_delay)
                continue

            fig = go.Figure()
            for class_label, color in CLASS_COLOR.items():
                use_filtered = (
                    filter_noise and clustered_points is not None
                    and class_label in ("Static Obstacle", "Dynamic Threat")
                )
                sub = clustered_points[clustered_points["Class"] == class_label] if use_filtered \
                    else df[df["Class"] == class_label]

                if class_label == "Drivable" and len(sub) > DRIVABLE_DISPLAY_CAP:
                    sub = sub.sample(DRIVABLE_DISPLAY_CAP, random_state=frame_idx)

                if sub.empty:
                    continue
                label_suffix = " (filtered)" if use_filtered else ""
                fig.add_trace(go.Scatter3d(
                    x=sub["x"], y=sub["y"], z=sub["z"], mode="markers",
                    name=f"{class_label} ({len(sub)}){label_suffix}",
                    marker=dict(size=sub["size"], color=color, opacity=0.75),
                ))

            fig.add_trace(go.Scatter3d(
                x=[0], y=[0], z=[0], mode="markers", name="UGV",
                marker=dict(size=10, color="yellow", symbol="diamond"),
            ))

            if show_zone_ring:
                theta = np.linspace(0, 2 * np.pi, 100)
                fig.add_trace(go.Scatter3d(
                    x=10.0 * np.cos(theta), y=10.0 * np.sin(theta), z=[0.02] * 100,
                    mode="lines", name="10m Foveation Boundary",
                    line=dict(color="lime", width=4, dash="dash"),
                ))

            if show_bboxes:
                for obj in objects:
                    add_bbox_trace(fig, obj, BOX_COLOR.get(obj["class"], "white"))

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
                legend=dict(bgcolor="rgba(0,0,0,0.5)"), height=680,
                # Keeps the camera angle/zoom from resetting every time the
                # figure below is re-sent to the same placeholder.
                uirevision="constant",
            )

            # NOTE: no explicit `key` here. Streamlit requires an explicit
            # `key` to be unique across the WHOLE script run, and this call
            # happens once per frame inside the same run -- reusing the same
            # literal key every iteration is exactly what raised
            # StreamlitDuplicateElementKey on frame 2. We don't need a key
            # for the flicker fix anyway: `map_placeholder` is the same
            # st.empty() slot every iteration (so content is replaced in
            # place, not remounted), and `uirevision="constant"` above is
            # what actually preserves the camera angle/zoom across updates.
            map_placeholder.plotly_chart(fig, use_container_width=True)

            debug_placeholder.caption(
                f"Frame {frame_idx} | {len(df)} cells | "
                f"fine (\u226410m): {int((df['radius'] <= 10.0).sum())} | "
                f"coarse (>10m): {int((df['radius'] > 10.0).sum())}"
            )

            if objects:
                panel_rows = []
                for obj in objects:
                    tag = "Dynamic Threat" if obj["class"] == "Dynamic Threat" else "Static Obstacle"
                    panel_rows.append(
                        f"**{tag}**  \n"
                        f"Dist: {obj['distance']:.1f}m | Pts: {obj['point_count']}"
                    )
                detection_panel.markdown("\n\n---\n\n".join(panel_rows))
            else:
                detection_panel.info("No discrete objects detected this frame.")

            progress_bar.progress((frame_idx + 1) / n_frames)
            time.sleep(frame_delay)

    progress_placeholder.success("Sequence complete.")