import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import torch
import time
import os
import math

from spiking_model import SpikingPointNet
from handoff import generate_2_5d_grid, memory_metrics
from grid_state import GridState
from profiler import EdgeProfiler
from synthetic_lidar_data import build_scene

try:
    from sklearn.cluster import DBSCAN
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

st.set_page_config(page_title="DRDO Tactical UGV Perception", layout="wide")

# ---------------------------------------------------------------------------
# Nocturne theme — global CSS. Palette resolved to literal hex/rgba values
# (not the design-system's CSS variables, which only exist inside the
# Claude Design canvas runtime) per ENHANCE-dash_pro.md §0.
# ---------------------------------------------------------------------------
st.markdown("""
/* Hide Streamlit's top header */
[data-testid="stHeader"] {
    display: none;
}

<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap">
<style>
:root{
  --nb:#101220; --ns:rgba(233,233,237,.04); --nt:#e9e9ed;
  --na:#9184d9; --na3:#d2cefd; --na4:#b5abfc; --na5:#968ae0;
  --na6:#796cbf; --na7:#5c5299; --na8:#423a6a; --na9:#2b2646;
  --nred:#ff5a6e; --nline:rgba(233,233,237,.09);
}
@keyframes noct-sweep {
  0% { transform: translateY(0); opacity: 0; }
  8% { opacity: .5; }
  92% { opacity: .5; }
  100% { transform: translateY(470px); opacity: 0; }
}
.stApp{
  background: radial-gradient(120% 80% at 18% -10%, #1d2036 0%, #141626 45%, #101220 100%);
}
html, body, [class*="css"]{ font-family:'Inter',system-ui,sans-serif; color:var(--nt); }
.block-container{ padding-top:1.6rem; max-width:100%; }
section[data-testid="stSidebar"]{ background:transparent; border-right:1px solid var(--nline); }
div[data-testid="stMetric"]{ display:none; }
*:focus-visible{ outline:2px solid var(--na); outline-offset:2px; }
div[data-testid="stButton"] button{
  background:var(--na5); color:#101220; border:none; border-radius:8px;
  font-weight:600; font-size:13px; padding:.5rem 1rem; width:100%;
}
div[data-testid="stButton"] button:hover{ background:var(--na4); color:#101220; }
div[data-testid="stSlider"] [data-testid="stTickBar"]{ display:none; }
div[data-testid="stSlider"] div[role="slider"]{
  background:var(--na3) !important; box-shadow:0 0 0 3px rgba(145,132,217,.22) !important;
}
div[data-testid="stSlider"] [data-baseweb="slider"] div div div{ background:var(--na5) !important; }
div[data-testid="stSlider"] [data-testid="stThumbValue"]{ display:none; }
div[data-testid="stCheckbox"] label span:first-child{
  border-color:var(--na6) !important; background:transparent !important; border-radius:4px !important;
}
div[data-testid="stCheckbox"] label:has(input:checked) span:first-child{
  background:var(--na5) !important; border-color:var(--na5) !important;
}
div[data-testid="stPlotlyChart"]{
  position:relative; border-radius:10px; overflow:hidden;
  box-shadow:0 0 0 1px rgba(233,233,237,.12), 0 24px 60px -20px rgba(0,0,0,.9);
}
div[data-testid="stPlotlyChart"]::before{
  content:""; position:absolute; inset:0; pointer-events:none; z-index:2;
  background:radial-gradient(120% 90% at 50% 45%, transparent 45%, rgba(6,6,10,.75) 100%);
}
div[data-testid="stPlotlyChart"]::after{
  content:""; position:absolute; left:0; right:0; top:0; height:90px; pointer-events:none; z-index:2;
  background:linear-gradient(180deg, transparent, rgba(145,132,217,.10) 70%, rgba(145,132,217,.22));
  animation: noct-sweep 7s linear infinite;
}
</style>
""", unsafe_allow_html=True)

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


def pill(text, warn=False):
    color = "#ffc379" if warn else "rgba(233,233,237,.68)"
    bg = "rgba(255,176,74,.12)" if warn else "rgba(233,233,237,.07)"
    dot = "#ffb04a" if warn else "var(--na4)"
    return (
        f'<span style="display:inline-flex;align-items:center;gap:6px;'
        f'padding:5.6px 11.2px;border-radius:999px;font-size:11.5px;'
        f'background:{bg};color:{color}">'
        f'<span style="width:6px;height:6px;border-radius:50%;background:{dot}"></span>{text}</span>'
    )


@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ai_model = SpikingPointNet(num_steps=10).to(device)
    if os.path.exists(CHECKPOINT_PATH):
        ai_model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    ai_model.eval()
    return ai_model, device


ai_model, device = load_model()
checkpoint_ok = os.path.exists(CHECKPOINT_PATH)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.markdown("""
<div style="display:flex;align-items:center;gap:8.4px;margin-bottom:6px">
  <div style="width:26px;height:26px;border-radius:6px;border:1px solid var(--na6);
              background:var(--na9);display:grid;place-items:center;
              box-shadow:0 0 18px -4px var(--na7)">
    <div style="width:8px;height:8px;border-radius:50%;background:var(--na4);box-shadow:0 0 8px var(--na4)"></div>
  </div>
  <div style="font-weight:500;font-size:13px;letter-spacing:.06em;text-transform:uppercase">Perception</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown(
    '<div style="font-size:10px;letter-spacing:.14em;text-transform:uppercase;'
    'color:rgba(233,233,237,.42);margin-top:12px">Display</div>',
    unsafe_allow_html=True,
)
show_zone_ring = st.sidebar.checkbox("10m foveation boundary", value=True, key="show_zone_ring")
show_bboxes = st.sidebar.checkbox("3D bounding boxes", value=True, key="show_bboxes")
show_fog = st.sidebar.checkbox("Distance fog on point cloud", value=False, key="show_fog")

st.sidebar.markdown(
    '<div style="height:1px;margin:14px 0;background:linear-gradient(90deg,transparent,'
    'rgba(233,233,237,.14) 20%,rgba(233,233,237,.14) 80%,transparent)"></div>',
    unsafe_allow_html=True,
)
st.sidebar.markdown(
    '<div style="font-size:10px;letter-spacing:.14em;text-transform:uppercase;'
    'color:rgba(233,233,237,.42)">Clustering</div>',
    unsafe_allow_html=True,
)

_eps_now = st.session_state.get("cluster_eps", 1.0)
st.sidebar.markdown(
    f'<div style="display:flex;justify-content:space-between;align-items:baseline;margin-top:8px">'
    f'<span style="font-size:13px">Cluster distance</span>'
    f'<span style="font-size:13px;font-variant-numeric:tabular-nums;color:var(--na3)">'
    f'{_eps_now:.2f} <span style="font-size:10px;color:rgba(233,233,237,.4)">m</span></span></div>',
    unsafe_allow_html=True,
)
cluster_eps = st.sidebar.slider(
    "Cluster distance (m)", 0.3, 3.0, 1.0, 0.1, key="cluster_eps", label_visibility="collapsed"
)
st.sidebar.markdown(
    '<div style="display:flex;justify-content:space-between;font-size:10px;'
    'color:rgba(233,233,237,.32);margin-top:-10px"><span>0.3</span><span>3.0</span></div>',
    unsafe_allow_html=True,
)

_minpts_now = st.session_state.get("cluster_min_pts", 5)
st.sidebar.markdown(
    f'<div style="display:flex;justify-content:space-between;align-items:baseline;margin-top:10px">'
    f'<span style="font-size:13px">Min points per object</span>'
    f'<span style="font-size:13px;font-variant-numeric:tabular-nums;color:var(--na3)">{_minpts_now}</span></div>',
    unsafe_allow_html=True,
)
cluster_min_pts = st.sidebar.slider(
    "Min points per object", 1, 15, 5, key="cluster_min_pts", label_visibility="collapsed"
)
st.sidebar.markdown(
    '<div style="display:flex;justify-content:space-between;font-size:10px;'
    'color:rgba(233,233,237,.32);margin-top:-10px"><span>1</span><span>15</span></div>',
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    '<div style="height:1px;margin:14px 0;background:linear-gradient(90deg,transparent,'
    'rgba(233,233,237,.14) 20%,rgba(233,233,237,.14) 80%,transparent)"></div>',
    unsafe_allow_html=True,
)
st.sidebar.markdown(f"""
<div style="font-size:10px;letter-spacing:.14em;text-transform:uppercase;
            color:rgba(233,233,237,.42);margin-bottom:8px">Runtime</div>
<div style="display:flex;flex-direction:column;gap:5.6px;padding:11.2px;border-radius:8px;
            background:rgba(233,233,237,.04)">
  <div style="display:flex;justify-content:space-between;font-size:12px">
    <span style="color:rgba(233,233,237,.5)">Checkpoint</span>
    <span style="color:{'#ffb04a' if not checkpoint_ok else 'var(--na3)'}">{'missing' if not checkpoint_ok else 'loaded'}</span>
  </div>
  <div style="display:flex;justify-content:space-between;font-size:12px">
    <span style="color:rgba(233,233,237,.5)">Device</span>
    <span style="font-variant-numeric:tabular-nums">{device}</span>
  </div>
  <div style="display:flex;justify-content:space-between;font-size:12px">
    <span style="color:rgba(233,233,237,.5)">Timesteps</span>
    <span style="font-variant-numeric:tabular-nums">10</span>
  </div>
  <div style="display:flex;justify-content:space-between;font-size:12px">
    <span style="color:rgba(233,233,237,.5)">Points in</span>
    <span style="font-variant-numeric:tabular-nums">{NUM_POINTS:,}</span>
  </div>
</div>
<div style="font-size:11px;line-height:1.4;color:rgba(233,233,237,.4);margin-top:8px">
  Untrained weights &mdash; predictions will collapse toward one class.</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
head_col, action_col = st.columns([3.4, 1])
with head_col:
    st.markdown("""
    <div>
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">
        <span style="font-size:10px;letter-spacing:.16em;text-transform:uppercase;color:var(--na4)">DRDO Tactical UGV</span>
        <span style="width:32px;height:1px;background:linear-gradient(90deg,var(--na5),transparent)"></span>
      </div>
      <h1 style="font-size:38px;line-height:1.05;letter-spacing:-.02em;margin:0;font-weight:500">
        Neuromorphic Variable&#8209;Resolution Perception</h1>
      <div style="margin-top:8px;font-size:13px;color:rgba(233,233,237,.52)">
        Spiking neural network &nbsp;&middot;&nbsp; UGV&#8209;centric LiDAR &nbsp;&middot;&nbsp; single&#8209;frame snapshot</div>
    </div>
    """, unsafe_allow_html=True)
with action_col:
    _ckpt_label = os.path.basename(CHECKPOINT_PATH) if checkpoint_ok else "No checkpoint"
    st.markdown(
        f'<div style="display:flex;gap:8px;justify-content:flex-end;margin-bottom:10px">'
        f'{pill(_ckpt_label, warn=not checkpoint_ok)}{pill(str(device))}</div>',
        unsafe_allow_html=True,
    )
    header_clicked = st.button("Generate Scene", key="btn_header", width="stretch")

# ---------------------------------------------------------------------------
# Pipeline helper functions (unchanged from the pre-redesign file, except
# add_bbox_trace's line styling — see ENHANCE-dash_pro.md §3.1)
# ---------------------------------------------------------------------------


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


def add_bbox_trace(fig, obj, color):
    """Corner-bracket wireframe (3 short segments per corner) instead of a
    full 12-edge box — ENHANCE-dash_pro.md §3.1: 'draw corner brackets
    instead of full 12-edge wireframes ... it looks like a tracker.'"""
    x0, x1, y0, y1, z0, z1 = obj["x_min"], obj["x_max"], obj["y_min"], obj["y_max"], obj["z_min"], obj["z_max"]
    dx, dy, dz = (x1 - x0) * 0.25, (y1 - y0) * 0.25, (z1 - z0) * 0.25
    corners = [
        (x0, y0, z0, 1, 1, 1), (x1, y0, z0, -1, 1, 1),
        (x1, y1, z0, -1, -1, 1), (x0, y1, z0, 1, -1, 1),
        (x0, y0, z1, 1, 1, -1), (x1, y0, z1, -1, 1, -1),
        (x1, y1, z1, -1, -1, -1), (x0, y1, z1, 1, -1, -1),
    ]
    for cx, cy, cz, sx, sy, sz in corners:
        xs = [cx + sx * dx, cx, None, cx, cx, None, cx, cx]
        ys = [cy, cy, None, cy + sy * dy, cy, None, cy, cy]
        zs = [cz, cz, None, cz, cz, None, cz, cz + sz * dz]
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs, mode="lines",
            line=dict(color=color, width=2),
            showlegend=False, hoverinfo="skip",
        ))


def _add_polar_grid(fig):
    theta = np.linspace(0, 2 * np.pi, 100)
    for rr in (25, 50, 75):
        fig.add_trace(go.Scatter3d(
            x=rr * np.cos(theta), y=rr * np.sin(theta), z=[0.01] * 100,
            mode="lines", showlegend=False, hoverinfo="skip",
            line=dict(color="rgba(233,233,237,0.07)", width=1),
        ))
    for ang in (0, np.pi / 2, np.pi, 3 * np.pi / 2):
        fig.add_trace(go.Scatter3d(
            x=[0, 100 * np.cos(ang)], y=[0, 100 * np.sin(ang)], z=[0.01, 0.01],
            mode="lines", showlegend=False, hoverinfo="skip",
            line=dict(color="rgba(233,233,237,0.07)", width=1),
        ))


def _add_legend_overlay(fig, class_counts, total):
    """Custom 'class distribution' legend baked into the figure as paper-
    coordinate shapes/annotations, replacing Plotly's default legend
    (ENHANCE-dash_pro.md §3.2). Not a pixel-identical port of the mockup's
    absolutely-positioned HTML box — Plotly annotations/shapes in paper
    coordinates achieve the same information design robustly inside a
    Streamlit-rendered chart, without relying on fragile DOM overlay tricks."""
    x0, y0, x1, y1 = 0.72, 0.70, 0.995, 0.995
    fig.add_shape(type="rect", xref="paper", yref="paper", x0=x0, y0=y0, x1=x1, y1=y1,
                  fillcolor="rgba(16,18,32,0.85)", line=dict(width=0), layer="below")
    fig.add_annotation(xref="paper", yref="paper", x=x0 + 0.012, y=y1 - 0.03,
                        text="CLASS DISTRIBUTION", showarrow=False, align="left",
                        font=dict(size=9, color="rgba(233,233,237,.55)"), xanchor="left")
    bar_y = y1 - 0.065
    cursor = x0 + 0.012
    bar_w = (x1 - x0) - 0.024
    colors = {"Drivable": "rgb(120,120,130)", "Static Obstacle": "rgb(0,170,255)", "Dynamic Threat": "#ff5a6e"}
    if total > 0:
        for cls in ("Drivable", "Static Obstacle", "Dynamic Threat"):
            frac = class_counts.get(cls, 0) / total
            if frac <= 0:
                continue
            fig.add_shape(type="rect", xref="paper", yref="paper",
                          x0=cursor, x1=cursor + bar_w * frac, y0=bar_y, y1=bar_y + 0.008,
                          fillcolor=colors[cls], line=dict(width=0), layer="above")
            cursor += bar_w * frac
    row_y = bar_y - 0.045
    for cls in ("Drivable", "Static Obstacle", "Dynamic Threat"):
        count = class_counts.get(cls, 0)
        text_color = "#ffdde1" if cls == "Dynamic Threat" else "rgba(233,233,237,.78)"
        fig.add_shape(type="rect", xref="paper", yref="paper",
                      x0=x0 + 0.012, x1=x0 + 0.02, y0=row_y - 0.006, y1=row_y + 0.006,
                      fillcolor=colors[cls], line=dict(width=0), layer="above")
        fig.add_annotation(xref="paper", yref="paper", x=x0 + 0.03, y=row_y,
                            text=cls, showarrow=False, align="left",
                            font=dict(size=10, color=text_color), xanchor="left")
        fig.add_annotation(xref="paper", yref="paper", x=x1 - 0.012, y=row_y,
                            text=f"{count:,}", showarrow=False, align="right",
                            font=dict(size=10, color="rgba(233,233,237,.55)"), xanchor="right")
        row_y -= 0.045


def _add_hud_overlay(fig, raw_point_count):
    cols = [
        ("FOVEATION", "10 m · 5 cm cells"),
        ("PERIPHERY", "to 100 m · 50 cm cells"),
        ("CAMERA", "az 45° · el 38°"),
    ]
    x = 0.02
    for label, value in cols:
        fig.add_annotation(xref="paper", yref="paper", x=x, y=0.05,
                            text=f"<span style='font-size:9px;letter-spacing:.14em'>{label}</span><br>{value}",
                            showarrow=False, align="left", xanchor="left",
                            font=dict(size=11, color="rgba(233,233,237,.6)"))
        x += 0.16


# ---------------------------------------------------------------------------
# Telemetry strip + Detected Objects rendering — shared by the pre-click and
# post-click branches (ENHANCE-dash_pro.md §2, §6).
# ---------------------------------------------------------------------------


def render_telemetry_html(data):
    """data is None for the pre-click ('—') state, or a dict with the live
    metrics for the post-click state."""
    def card(label, value_html, bar_html, caption, extra_bg=None, extra_shadow=None):
        bg = extra_bg or "linear-gradient(180deg, rgba(233,233,237,.06), rgba(233,233,237,.025))"
        shadow = extra_shadow or "0 0 0 1px rgba(63,66,77,.6)"
        tick_color = "#ff5a6e" if extra_bg else "var(--na5)"
        return f"""
        <div style="position:relative;padding:14px 14px 12px;border-radius:8px;
                    background:{bg};box-shadow:{shadow};overflow:hidden">
          <div style="position:absolute;top:0;left:0;width:22px;height:1px;background:{tick_color}"></div>
          <div style="font-size:9.5px;letter-spacing:.15em;text-transform:uppercase;
                      color:rgba(233,233,237,.45)">{label}</div>
          <div style="margin-top:6px;display:flex;align-items:baseline;gap:5px;
                      font-variant-numeric:tabular-nums">{value_html}</div>
          <div style="margin-top:12px">{bar_html}</div>
          <div style="margin-top:6px;font-size:10px;color:rgba(233,233,237,.35)">{caption}</div>
        </div>"""

    if data is None:
        dash = '<span style="font-size:34px;line-height:1;color:rgba(233,233,237,.3)">&mdash;</span>'
        flat_bar = '<div style="height:3px;border-radius:2px;background:rgba(233,233,237,.1)"></div>'
        cards = [
            card("Speed", dash, flat_bar, "model + grid only &middot; target 10"),
            card("Active cells", dash, flat_bar, "fine &mdash; &middot; coarse &mdash;"),
            card("Sparsity", dash, flat_bar, "neuron states that never fired"),
            card("Energy saved", dash, flat_bar, "vs dense MAC baseline"),
            card("Objects", dash, flat_bar, "nearest at &mdash;"),
        ]
        return f'<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:11.2px;margin-bottom:22.4px">{"".join(cards)}</div>'

    fps = data["fps"]
    fps_pct = min(fps / 10.0, 1.0) * 100
    active = data["active_cells"]
    fine, coarse = data["fine"], data["coarse"]
    fine_pct = (fine / active * 100) if active else 0
    sparsity = data["sparsity_pct"]
    energy_uj = data["energy_pj"] / 1e6
    hist = data["energy_hist"]
    n_objects = data["n_objects"]
    nearest = data["nearest_dist"]

    speed_card = card(
        "Speed",
        f'<span style="font-size:34px;line-height:1;letter-spacing:-.02em">{fps:.0f}</span>'
        f'<span style="font-size:13px;color:rgba(233,233,237,.45)">FPS</span>',
        f'<div style="height:3px;border-radius:2px;background:rgba(233,233,237,.1);position:relative">'
        f'<div style="position:absolute;inset:0 {100-fps_pct:.1f}% 0 0;border-radius:2px;background:#ff5a6e"></div>'
        f'<div style="position:absolute;left:66%;top:-3px;bottom:-3px;width:1px;background:rgba(233,233,237,.35)"></div></div>',
        "model + grid only &middot; target 10",
    )
    cells_card = card(
        "Active cells",
        f'<span style="font-size:34px;line-height:1;letter-spacing:-.02em">{active:,}</span>',
        f'<div style="display:flex;height:3px;border-radius:2px;overflow:hidden;background:rgba(233,233,237,.1)">'
        f'<div style="width:{fine_pct:.1f}%;background:var(--na4)"></div>'
        f'<div style="width:{100-fine_pct:.1f}%;background:var(--na8)"></div></div>',
        f"fine {fine:,} &middot; coarse {coarse:,}",
    )
    sparsity_card = card(
        "Sparsity",
        f'<span style="font-size:34px;line-height:1;letter-spacing:-.02em">{sparsity:.1f}</span>'
        f'<span style="font-size:13px;color:rgba(233,233,237,.45)">%</span>',
        f'<div style="height:3px;border-radius:2px;background:rgba(233,233,237,.1);position:relative">'
        f'<div style="position:absolute;inset:0 {100-sparsity:.1f}% 0 0;border-radius:2px;'
        f'background:var(--na4);box-shadow:0 0 10px -1px var(--na5)"></div></div>',
        "neuron states that never fired",
    )
    max_hist = max(hist) if hist else 1.0
    bars = "".join(
        f'<div style="flex:1;height:{max(8, h/max_hist*100):.0f}%;'
        f'background:{"var(--na4)" if i == len(hist)-1 else "var(--na8)"}"></div>'
        for i, h in enumerate(hist)
    ) or '<div style="flex:1;height:20%;background:var(--na8)"></div>' * 7
    energy_card = card(
        "Energy saved",
        f'<span style="font-size:34px;line-height:1;letter-spacing:-.02em">{energy_uj:.2f}</span>'
        f'<span style="font-size:13px;color:rgba(233,233,237,.45)">&micro;J</span>',
        f'<div style="display:flex;align-items:flex-end;gap:3px;height:12px">{bars}</div>',
        "vs dense MAC baseline",
    )
    pips = "".join(
        f'<div style="flex:1;height:3px;border-radius:2px;'
        f'background:{"#ff5a6e" if i < min(n_objects,5) else "rgba(255,90,110,.2)"}"></div>'
        for i in range(5)
    )
    objects_card = card(
        "Objects",
        f'<span style="font-size:34px;line-height:1;letter-spacing:-.02em;color:#ffdde1">{n_objects}</span>'
        f'<span style="font-size:11px;color:#ff97a4">threats</span>',
        f'<div style="display:flex;gap:4px">{pips}</div>',
        f"nearest at {nearest:.1f} m" if nearest is not None else "no objects detected",
        extra_bg="linear-gradient(180deg, rgba(255,90,110,.12), rgba(255,90,110,.04))",
        extra_shadow="0 0 0 1px rgba(255,90,110,.28)",
    )
    cards = [speed_card, cells_card, sparsity_card, energy_card, objects_card]
    return f'<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:11.2px;margin-bottom:22.4px">{"".join(cards)}</div>'


def render_objects_panel_html(objects, latency, foveation_radius=10.0):
    rows = []
    for i, obj in enumerate(objects, start=1):
        frac = min(obj["distance"] / foveation_radius, 1.0) * 100
        near = i <= 2
        bg = ("linear-gradient(180deg, rgba(255,90,110,.10), rgba(233,233,237,.03))" if near
              else "rgba(233,233,237,.04)")
        shadow = "0 0 0 1px rgba(255,90,110,.22)" if near else "0 0 0 1px rgba(63,66,77,.6)"
        bar_color = "#ff5a6e" if near else "rgba(255,90,110,.7)"
        dist_color = "#ffdde1" if near else "var(--nt)"
        label_color = "#ff97a4" if near else "rgba(255,151,164,.8)"
        rows.append(f"""
        <div style="position:relative;padding:12px 14px;border-radius:8px;background:{bg};box-shadow:{shadow};margin-bottom:8.4px">
          <div style="position:absolute;left:0;top:12px;bottom:12px;width:2px;background:#ff5a6e;border-radius:2px"></div>
          <div style="display:flex;align-items:center;gap:7px">
            <span style="font-size:10px;letter-spacing:.1em;text-transform:uppercase;color:{label_color}">{obj['class']}</span>
            <span style="margin-left:auto;font-size:10px;font-variant-numeric:tabular-nums;color:rgba(233,233,237,.35)">{i:02d}</span>
          </div>
          <div style="margin-top:7px;display:flex;align-items:baseline;gap:5px;font-variant-numeric:tabular-nums">
            <span style="font-size:27px;line-height:1;color:{dist_color}">{obj['distance']:.1f}</span>
            <span style="font-size:12px;color:rgba(233,233,237,.45)">m</span>
            <span style="margin-left:auto;font-size:11px;color:rgba(233,233,237,.5)">{obj['point_count']} cells</span>
          </div>
          <div style="margin-top:10px;height:2px;border-radius:2px;background:rgba(233,233,237,.1);position:relative">
            <div style="position:absolute;inset:0 {100-frac:.1f}% 0 0;border-radius:2px;background:{bar_color}"></div>
          </div>
        </div>""")
    objects_html = "".join(rows) if rows else (
        '<div style="padding:12px 14px;border-radius:8px;background:rgba(233,233,237,.04);'
        'color:rgba(233,233,237,.5);font-size:13px">No discrete objects detected this frame.</div>'
    )

    if latency is not None:
        total = latency["sample"] + latency["infer"] + latency["grid"]
        seg = lambda s: (s / total * 100) if total else 0
        latency_card = f"""
        <div style="display:flex;flex-direction:column;gap:8.4px;padding:12px 14px;border-radius:8px;
                    background:rgba(233,233,237,.03);box-shadow:0 0 0 1px rgba(63,66,77,.6);margin-top:8.4px">
          <div style="font-size:9.5px;letter-spacing:.15em;text-transform:uppercase;color:rgba(233,233,237,.42)">Latency breakdown</div>
          <div style="display:flex;height:5px;border-radius:3px;overflow:hidden">
            <div style="width:{seg(latency['sample']):.1f}%;background:var(--na4)"></div>
            <div style="width:{seg(latency['infer']):.1f}%;background:var(--na6)"></div>
            <div style="width:{seg(latency['grid']):.1f}%;background:var(--na8)"></div>
          </div>
          <div style="display:grid;grid-template-columns:1fr auto;gap:3px 8px;font-size:11px;font-variant-numeric:tabular-nums">
            <span style="color:rgba(233,233,237,.6)">Sample + normalise</span><span>{latency['sample']*1000:.0f} ms</span>
            <span style="color:rgba(233,233,237,.6)">SNN inference</span><span>{latency['infer']*1000:.0f} ms</span>
            <span style="color:rgba(233,233,237,.6)">Grid engine</span><span>{latency['grid']*1000:.0f} ms</span>
            <span style="color:rgba(233,233,237,.45)">DBSCAN</span><span style="color:rgba(233,233,237,.45)">not measured</span>
          </div>
        </div>"""
    else:
        latency_card = ""

    return objects_html + latency_card


# ---------------------------------------------------------------------------
# Empty-state viewport (pre-click) — ENHANCE-dash_pro.md §6.
# ---------------------------------------------------------------------------
EMPTY_STATE_HTML = """
<div style="position:relative;height:560px;border-radius:10px;overflow:hidden;
            background:radial-gradient(80% 100% at 50% 100%, #191c2e, #0a0b14);
            box-shadow:0 0 0 1px rgba(233,233,237,.12), 0 24px 60px -20px rgba(0,0,0,.9)">
  <div style="position:absolute;inset:0;
              background-image:repeating-linear-gradient(90deg, rgba(233,233,237,.06) 0 1px, transparent 1px 56px),
                                repeating-linear-gradient(0deg, rgba(233,233,237,.06) 0 1px, transparent 1px 56px);
              mask-image:radial-gradient(70% 90% at 50% 90%, #000 0%, transparent 75%);
              -webkit-mask-image:radial-gradient(70% 90% at 50% 90%, #000 0%, transparent 75%)"></div>
  <div style="position:absolute;left:50%;bottom:34px;width:220px;height:220px;margin-left:-110px;
              border-radius:50%;border:1px dashed rgba(86,255,155,.4);transform:scaleY(.34)"></div>
  <div style="position:absolute;left:50%;bottom:66px;width:9px;height:9px;margin-left:-4.5px;
              background:#ffd84a;transform:rotate(45deg);box-shadow:0 0 14px #ffd84a"></div>
  <div style="position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;
              justify-content:center;gap:11.2px;text-align:center">
    <div style="font-size:20px;font-weight:500">Sensor idle</div>
    <div style="font-size:13px;max-width:380px;color:rgba(233,233,237,.52)">
      Generate a synthetic LiDAR frame to run the spiking network and populate the grid.</div>
  </div>
  <div style="position:absolute;top:10px;left:10px;width:16px;height:16px;border-top:1px solid var(--na5);border-left:1px solid var(--na5);opacity:.7"></div>
  <div style="position:absolute;top:10px;right:10px;width:16px;height:16px;border-top:1px solid var(--na5);border-right:1px solid var(--na5);opacity:.7"></div>
  <div style="position:absolute;bottom:10px;left:10px;width:16px;height:16px;border-bottom:1px solid var(--na5);border-left:1px solid var(--na5);opacity:.7"></div>
  <div style="position:absolute;bottom:10px;right:10px;width:16px;height:16px;border-bottom:1px solid var(--na5);border-right:1px solid var(--na5);opacity:.7"></div>
</div>
"""

# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------
if "has_generated" not in st.session_state:
    st.session_state.has_generated = False
if "energy_hist" not in st.session_state:
    st.session_state.energy_hist = []

metrics_placeholder = st.empty()
main_col, side_col = st.columns([3.4, 1])
with main_col:
    map_placeholder = st.empty()
    debug_placeholder = st.caption("")
with side_col:
    st.markdown("### Detected Objects")
    detection_panel = st.empty()

show_empty_cta = not st.session_state.has_generated and not header_clicked
empty_clicked = False
if show_empty_cta:
    with map_placeholder.container():
        st.markdown(EMPTY_STATE_HTML, unsafe_allow_html=True)
        _, cta_col, _ = st.columns([1, 1, 1])
        with cta_col:
            empty_clicked = st.button("Generate Scene", key="btn_empty_cta", width="stretch")

do_generate = header_clicked or empty_clicked

if do_generate:
    import handoff
    handoff._state = GridState()

    points, labels_gt, spikes_gt = build_scene()
    t_start = time.perf_counter()

    norm_tensor, raw_sampled = scene_to_tensor(points)
    inputs = torch.tensor(norm_tensor, dtype=torch.float32).unsqueeze(0).to(device)
    t_sample = time.perf_counter()

    with torch.no_grad():
        spk_rec = ai_model(inputs)
    t_infer = time.perf_counter()
    total_spikes = spk_rec.sum(dim=0)
    preds_np = torch.argmax(total_spikes, dim=1).squeeze().cpu().numpy()
    spikes_np = total_spikes.sum(dim=1).squeeze().cpu().numpy()

    active_map = generate_2_5d_grid(raw_sampled, preds_np, spikes_np)
    mem_stats = memory_metrics()
    t_grid = time.perf_counter()
    perf = profiler.evaluate_efficiency(spk_rec, time.perf_counter() - t_start, 1)

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

    st.session_state.has_generated = True
    st.session_state.last_df = df
    st.session_state.last_perf = perf
    st.session_state.last_mem_stats = mem_stats
    st.session_state.last_latency = {
        "sample": t_sample - t_start, "infer": t_infer - t_sample, "grid": t_grid - t_infer,
    }
    st.session_state.last_raw_point_count = int(points.shape[0])
    st.session_state.energy_hist = (st.session_state.energy_hist + [perf["energy_saved_pj"]])[-7:]

if st.session_state.has_generated:
    df = st.session_state.last_df
    perf = st.session_state.last_perf
    mem_stats = st.session_state.last_mem_stats
    latency = st.session_state.last_latency
    raw_point_count = st.session_state.last_raw_point_count

    objects = cluster_objects(df, cluster_eps, cluster_min_pts) if show_bboxes else []
    fine = int((df['radius'] <= 10.0).sum()) if len(df) else 0
    coarse = int((df['radius'] > 10.0).sum()) if len(df) else 0

    metrics_placeholder.markdown(render_telemetry_html({
        "fps": perf["fps"], "active_cells": mem_stats["active_cell_count"],
        "fine": fine, "coarse": coarse, "sparsity_pct": perf["sparsity_pct"],
        "energy_pj": perf["energy_saved_pj"], "energy_hist": st.session_state.energy_hist,
        "n_objects": len(objects),
        "nearest_dist": objects[0]["distance"] if objects else None,
    }), unsafe_allow_html=True)

    if len(df) == 0:
        with map_placeholder.container():
            st.warning("No active cells in this scene.")
    else:
        fig = go.Figure()
        class_counts = df["Class"].value_counts().to_dict()

        for class_label, color in CLASS_COLOR.items():
            sub = df[df["Class"] == class_label]
            if sub.empty:
                continue
            if show_fog:
                near = sub[sub["radius"] <= 10.0]
                far = sub[sub["radius"] > 10.0]
                if not near.empty:
                    fig.add_trace(go.Scatter3d(
                        x=near["x"], y=near["y"], z=near["z"], mode="markers",
                        name=f"{class_label} ({len(sub)})", showlegend=False,
                        marker=dict(size=near["size"], color=color, opacity=0.85, line=dict(width=0)),
                    ))
                if not far.empty:
                    fig.add_trace(go.Scatter3d(
                        x=far["x"], y=far["y"], z=far["z"], mode="markers",
                        name=f"{class_label} ({len(sub)})", showlegend=False,
                        marker=dict(size=far["size"], color=color, opacity=0.45, line=dict(width=0)),
                    ))
            else:
                fig.add_trace(go.Scatter3d(
                    x=sub["x"], y=sub["y"], z=sub["z"], mode="markers",
                    name=f"{class_label} ({len(sub)})", showlegend=False,
                    marker=dict(size=sub["size"], color=color, opacity=0.75, line=dict(width=0)),
                ))

        fig.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[0], mode="markers", name="UGV", showlegend=False,
            marker=dict(size=10, color="yellow", symbol="diamond"),
        ))

        _add_polar_grid(fig)

        if show_zone_ring:
            theta = np.linspace(0, 2 * np.pi, 100)
            fig.add_trace(go.Scatter3d(
                x=10.0 * np.cos(theta), y=10.0 * np.sin(theta), z=[0.02] * 100,
                mode="lines", name="10m Foveation Boundary", showlegend=False,
                line=dict(color="#56ff9b", width=2, dash="dash"),
            ))

        if show_bboxes:
            for obj in objects:
                add_bbox_trace(fig, obj, BOX_COLOR.get(obj["class"], "white"))

        _add_legend_overlay(fig, class_counts, len(df))
        _add_hud_overlay(fig, raw_point_count)

        fig.update_layout(
            showlegend=False,
            scene=dict(
                xaxis=dict(title="", showticklabels=False, showspikes=False,
                           gridcolor="rgba(233,233,237,.06)", zeroline=False, backgroundcolor="#06060a"),
                yaxis=dict(title="", showticklabels=False, showspikes=False,
                           gridcolor="rgba(233,233,237,.06)", zeroline=False, backgroundcolor="#06060a"),
                zaxis=dict(title="", showticklabels=False, showspikes=False,
                           gridcolor="rgba(233,233,237,.06)", zeroline=False, backgroundcolor="#06060a"),
                aspectmode="data",
                camera=dict(eye=dict(x=1.3, y=1.3, z=1.0)),
            ),
            paper_bgcolor="#06060a", plot_bgcolor="#06060a",
            font=dict(color="white"), margin=dict(l=0, r=0, t=0, b=0),
            hoverlabel=dict(bgcolor="#1d2036", bordercolor="#423a6a", font=dict(color="#e9e9ed", size=11)),
            height=560,
            uirevision="constant",
        )

        with map_placeholder.container():
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    debug_placeholder.markdown(
        f'<div style="display:flex;align-items:center;gap:11.2px;font-size:11px;'
        f'color:rgba(233,233,237,.38);margin-top:8px">'
        f'<span>Drag to orbit &middot; scroll to zoom &middot; camera persists across runs</span>'
        f'<span style="margin-left:auto;font-variant-numeric:tabular-nums">'
        f'frame 01 &middot; {NUM_POINTS:,} pts sampled of {raw_point_count:,}</span></div>',
        unsafe_allow_html=True,
    )

    detection_panel.markdown(render_objects_panel_html(objects, latency), unsafe_allow_html=True)

else:
    metrics_placeholder.markdown(render_telemetry_html(None), unsafe_allow_html=True)
    debug_placeholder.caption("")
    detection_panel.info("No discrete objects detected this frame.")
