"""
engine_adapter.py
------------------------------------------------------------------
Isolation layer between your real SNN pipeline and the DRDO-style
radar UI. The UI only ever talks to `Engine`, never to your model
directly — so swapping in the real spiking_model.py / profiler.py /
kitti_loader.py later means editing ONLY this file, not the GUI.

Two responsibilities:
  1. Produce a per-frame `Frame` object (points, labels, fps, etc.)
     either from the real model (if importable and a checkpoint is
     present) or from a realistic synthetic generator.
  2. Track objects across frames (nearest-centroid matching) so the
     UI can draw comet trails, velocity vectors, and static/dynamic
     classification — the real grid engine emits classified cells, not
     persistent object IDs, so `_live_step()` clusters cells into
     discrete detections (DBSCAN, mirroring dashboard_pro.py's
     cluster_objects) and feeds them through the same tracker as the
     synthetic path.

Verified in a 4000-frame headless simulation: object tracking,
clustering, threat evaluation, and the evasion state machine all run
without exceptions and produce clean, occasional evasion engage/clear
cycles (not a permanently-stuck state) — see the "engaged" / "Path
clear" log lines this module emits.
"""

from __future__ import annotations
import math
import os
import sys
import time
import itertools
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

TARGET_FPS = 261
SENSOR_RANGE_M = 100.0          # matches your 100m validation goal
DYNAMIC_SPEED_THRESHOLD_MPS = 0.35   # centroid displacement/sec above this = "dynamic"
TRACK_MAX_AGE_FRAMES = 15       # drop a track if unseen for this many frames
# 7.0m (widened from 4.0m): measured frame-to-frame centroid jitter for real
# clustered detections has vehicle-class p75=5.35m, obstacle-class p75=1.71m,
# with both distributions' p90+ tail past ~12m -- 7.0m clears typical jitter
# without reaching into the tail where merging distinct nearby objects
# becomes a real risk. 4.0m was tuned for 8 hand-placed synthetic objects,
# not jittery real cluster centroids.
TRACK_MATCH_RADIUS_M = 7.0
TRACK_CONFIRM_HITS = 2          # a track must match at least this many times before it's rendered

# Measured raw (unsmoothed) velocity-arrow length (|velocity|*2.5) over 40
# live frames: median 48.4m, p90=155.6m, max=352.6m -- the max implies a raw
# speed of ~141 m/s, physically absurd for this scene (driving_sequence.py's
# UGV cruises at 1.5 m/s; its pedestrian/overtaking-car actors are far
# slower). MAX_VELOCITY_MPS clamps each raw velocity component BEFORE it
# enters the EMA below, so a single bad cluster-centroid match can never
# contribute an unbounded spike even transiently.
MAX_VELOCITY_MPS = 15.0
VELOCITY_EMA_ALPHA = 0.35       # aggressive smoothing, justified by how extreme the measured raw noise is

# Render/density split: which confirmed vehicle-class tracks get full icon
# treatment vs. get demoted to background density points. Measured smoothed
# speed does NOT discriminate real motion from noise here -- frame-to-frame
# re-clustering jitter gives ~88% of confirmed tracks a smoothed speed
# >= 5 m/s regardless of whether they're a real object, since the noise
# floor itself is persistently high, not just occasional spikes. Hit count
# (how many real frames a track has stayed matched) DOES discriminate:
# measured distribution at frame 40 ranged 2-40 with a wide spread (median
# 8) -- genuine/stable detections keep re-matching near the same place and
# accumulate hits, while transient noise blips stay low before aging out
# (TRACK_MAX_AGE_FRAMES=15). RENDER_MIN_HITS=10 keeps roughly the top ~45%
# most persistent tracks (measured: 36/80) as full icons.
RENDER_MIN_HITS = 10
# Hit count at which a track STARTS fading in. Promotion used to be a hard
# switch at RENDER_MIN_HITS: one frame a track was a dim 1.4px density dot, the
# next it was a full 16px glowing cube with a trail and a velocity arrow. That
# instantaneous appearance is a large part of why objects read as coming out of
# nowhere. Between these two counts a track is drawn as a real icon but ramped
# in opacity, so it resolves into view instead of snapping into it.
RENDER_FADE_IN_HITS = 6

# Forward-corridor obstacle scan -- the input to closed-loop steering. This
# reads the grid engine's CELLS, not ObjectTracker's detections, on purpose:
# ~42% of confirmed tracks are ghosts (see Reports/AUDIT-V3.md §7), so a
# controller driven by them would swerve at phantoms.
#
# The discriminator is ELEVATION, not semantic class. Measured over 40 live
# frames: cells sitting on a real pole have median elevation_max 1.76m
# (p90 1.97m), while every other non-drivable cell -- ground plus the SNN's
# misclassification noise -- has median 0.00m (p90 0.04m). At a 0.5m floor
# that keeps 85.4% of real obstacle cells and only 5.5% of noise. Class
# alone does NOT separate them: the model emits 1,000-3,600 class-2 cells
# per frame vs. only 27-112 class-1.
OBSTACLE_MIN_ELEVATION_M = 0.5
# Measured blocked-frame rate over 80 live frames at min_cells=8:
# 20m/2.0m -> 23.8% of frames (mean closest 8.8m), 20m/2.5m -> 25.0%,
# 12m/2.0m -> 13.8%, 20m/3.0m@min3 -> 60.0% (too twitchy). 20m/2.0m/8
# fires often enough to demo and rarely enough to read as a discrete event.
CORRIDOR_LOOKAHEAD_M = 20.0
CORRIDOR_HALF_WIDTH_M = 2.0
CORRIDOR_MIN_CELLS = 8
# Cells nearly abeam of the UGV are not something it can still steer around,
# and counting them made the readout nonsensical: measured over 500 frames the
# corridor reported obstacles at 0.07-0.42m while the nearest real object was
# 2.0-2.7m away and passing alongside -- the forward PROJECTION of a cell beside
# the vehicle is near zero. Requiring real forward separation keeps the range
# readout and the "Obstacle in corridor at Xm" log line honest.
CORRIDOR_MIN_FORWARD_M = 1.0
EVASION_TURN_DEG = 35.0         # matches _advance_ugv's original turn magnitude
# driving_sequence.py's actors advance per FRAME INDEX, not per wall-clock
# second: the pedestrian moves 1.2m, the overtaking car 3.5m and the scripted
# UGV 1.5m for each +1 of frame_idx -- i.e. one sequence frame is implicitly
# one second of scene time. The UGV's pose is now integrated by this module
# instead of read from get_ugv_position(), so it has to be integrated on that
# same clock. Using the real ~0.12s step interval instead moved the UGV only
# ~0.18m per frame while the car still moved 3.5m, so the car overran the UGV
# within a few frames and sat on top of it (obstacles reported at 0.1m).
# One engine step used to advance the scene a FULL second, so with the engine
# running at ~10-15 Hz the world played at roughly 8-15x real time: the car
# crossed the whole display in three frames and every actor moved in large
# visible jumps. build_driving_frame only ever uses frame_idx in linear
# START + VELOCITY*idx terms, so a FRACTIONAL index is valid and is how the
# scene is slowed without desynchronising the actors from the UGV -- both are
# advanced by SCENE_STEP scene-seconds per engine step.
SCENE_STEP = 0.25
SCENE_DT_S = SCENE_STEP
# Both of these are per scene-SECOND, not per engine frame, so they are
# unaffected by SCENE_STEP: a manoeuvre still completes over the same distance
# of travel, just spread across 1/SCENE_STEP times as many rendered frames --
# which is exactly what makes the turn read as smooth instead of a snap.
LIVE_TURN_RATE_DEG_S = 18.0     # a 35-degree turn takes ~2 scene-seconds
EVASION_HOLD_S = 3.0            # scene-seconds to hold a manoeuvre before resuming
# Lane recovery. Without it the UGV swerves around an obstacle, resumes the
# nominal +X heading, and simply keeps the offset -- measured 17.7m of
# accumulated lateral drift over one 40-frame replay, i.e. it wandered out of
# the scene entirely and stopped meeting obstacles at all. Steering back
# toward the corridor centreline turns each avoidance into a swerve-and-
# recover instead of a one-way departure.
LANE_RETURN_GAIN_DEG_PER_M = 6.0
LANE_RETURN_MAX_DEG = 25.0

NUM_POINTS_LIVE = 8192          # matches dashboard_pro.py/dashboard_driving.py's scene_to_tensor
LIVE_CLUSTER_EPS_M = 1.0        # mirrors dashboards' UI-slider default for cluster_objects
LIVE_CLUSTER_MIN_SAMPLES = 5    # mirrors dashboards' UI-slider default for cluster_objects
LIVE_CLASS_LABELS = {1: "obstacle", 2: "vehicle"}   # 0=drivable excluded; grid engine has no "vegetation" class
# Measured cluster-size distribution (one frame, 135 raw clusters): min=3,
# p50=7, p90=16.6 -- ~98% already sit at/above DBSCAN's own min_samples=5
# floor, so this filter only cleans up rare 3-4 cell outliers; it is not
# the primary fix for track churn (that's TRACK_MATCH_RADIUS_M/TRACK_CONFIRM_HITS).
LIVE_MIN_CLUSTER_CELLS = 5
LIVE_CLUSTER_MERGE_RADIUS_M = 2.0   # merge same-class cluster centroids within this distance
# Measured: generate_2_5d_grid's snapshot() and the per-class cell filter in
# _cluster_cells_to_detections are both O(total accumulated cells), which
# only reset once per full scene replay -- step time grows from
# ~57ms to ~360ms as accumulated cells grow from 7.5k to 131k across one
# cycle. Resetting more often caps that growth (the actual measured FPS
# bottleneck -- GPU inference is only ~12ms and already not the issue).
LIVE_RESET_EVERY_N_FRAMES = 10


# ======================================================================
# Try importing the real engine. If any import fails, LIVE_MODEL_AVAILABLE
# stays False and everything falls back to the synthetic generator below.
# ======================================================================
LIVE_MODEL_AVAILABLE = True
try:
    import torch
    from sklearn.cluster import DBSCAN
    from spiking_model import SpikingPointNet
    from profiler import EdgeProfiler
    from driving_sequence import (build_driving_frame, get_ugv_position,
                                  build_static_ground,
                                  UGV_SPEED as LIVE_UGV_SPEED_MPS)
    import handoff
    from handoff import generate_2_5d_grid
    from grid_state import GridState
except Exception:
    LIVE_MODEL_AVAILABLE = False


@dataclass
class TrackedObject:
    track_id: int
    x: float
    y: float
    z: float
    label: str                     # "obstacle" | "vehicle" | "vegetation" | "drivable"
    history: list = field(default_factory=list)   # list of (x, y) for comet trail
    velocity: tuple = (0.0, 0.0)    # (vx, vy) in m/s, estimated
    is_dynamic: bool = False
    age_frames: int = 0
    last_seen_frame: int = 0
    threat: bool = False            # True if projected path intersects UGV safety margin
    hits: int = 1                   # incremented on each successful match; rendered once >= TRACK_CONFIRM_HITS
    render_alpha: float = 1.0       # 0..1 display opacity; see Engine.step's fade-in/fade-out


@dataclass
class Frame:
    frame_id: int
    t: float
    ugv_x: float
    ugv_y: float
    ugv_heading_deg: float          # 0 = facing +Y ("up" on screen), clockwise positive
    ugv_speed_mps: float
    fps: float
    latency_ms: float
    spike_rate: float
    sparsity: float
    tracks: list                    # list[TrackedObject], current frame's live tracks
    evasion_active: bool
    evasion_target_heading_deg: Optional[float]
    log_events: list                # list[str], new log lines generated this frame
    live_cells: Optional[list] = None   # list[(cell_key, CellRecord)] this tick if live, else None
    density_points: list = field(default_factory=list)   # list[(x, y, label)], non-tracked background classification
    # Forward-corridor scan this frame, so radar_view_2d draws what the
    # controller actually decided on rather than recomputing it.
    corridor_blocked: bool = False
    corridor_closest_m: Optional[float] = None
    corridor_lookahead_m: float = CORRIDOR_LOOKAHEAD_M
    corridor_half_width_m: float = CORRIDOR_HALF_WIDTH_M


def _scene_to_tensor(points, num_points=NUM_POINTS_LIVE, max_range=SENSOR_RANGE_M):
    """Copied from dashboard_pro.py/dashboard_driving.py's scene_to_tensor
    (both files duplicate it locally rather than sharing a module -- we
    follow that convention here too, and avoid importing from either
    Streamlit dashboard, which would drag in a streamlit dependency)."""
    n = points.shape[0]
    if n >= num_points:
        # Deterministic evenly-spaced stride, NOT np.random.choice. The random
        # draw picked a different ~8k of the ~24.8k points every frame, so even
        # with a fixed scene the surviving points -- and therefore the occupied
        # grid cells, the DBSCAN clusters and the density dots -- were re-rolled
        # each frame. Combined with build_static_ground()'s fixed ring pattern,
        # a fixed stride selects the SAME ground points every frame, which is
        # what makes cells persist. This uses no label or per-object knowledge,
        # only the point count, so it stays an honest sampler.
        idx = np.linspace(0, n - 1, num_points).astype(np.intp)
    else:
        idx = np.pad(np.arange(n), (0, num_points - n), mode="wrap")
    pts_sampled = points[idx]
    coords = pts_sampled[:, :3]
    norm_coords = np.clip((coords + max_range) / (2.0 * max_range), 0.0, 1.0)
    norm_tensor = np.zeros((4, num_points), dtype=np.float32)
    norm_tensor[:3, :] = norm_coords.T
    norm_tensor[3, :] = pts_sampled[:, 3]
    return norm_tensor, pts_sampled


def _merge_nearby_clusters(clusters, merge_radius_m):
    """Iterative nearest-merge of same-class preliminary clusters within
    merge_radius_m of each other, so one over-segmented obstacle region
    becomes one detection instead of several overlapping ones. `clusters`
    is a list of (ego_x, ego_y, elevation, cell_count) tuples; merged
    centroid is cell-count-weighted, elevation is the max of the merged
    set."""
    clusters = list(clusters)
    while True:
        n = len(clusters)
        best_pair, best_dist = None, merge_radius_m
        for i in range(n):
            for j in range(i + 1, n):
                d = math.hypot(clusters[i][0] - clusters[j][0], clusters[i][1] - clusters[j][1])
                if d < best_dist:
                    best_dist, best_pair = d, (i, j)
        if best_pair is None:
            return clusters
        i, j = best_pair
        xi, yi, zi, ni = clusters[i]
        xj, yj, zj, nj = clusters[j]
        total = ni + nj
        merged = (
            (xi * ni + xj * nj) / total,
            (yi * ni + yj * nj) / total,
            max(zi, zj),
            total,
        )
        clusters = [c for k, c in enumerate(clusters) if k not in (i, j)]
        clusters.append(merged)


def _cluster_cells_to_detections(active_map, ugv_world_pos):
    """DBSCAN over real grid-engine cells -> discrete (x, y, z, label)
    detections, mirroring dashboard_pro.py's cluster_objects: drivable
    cells excluded, DBSCAN run separately per class (so a cluster's label
    is simply the class it was scoped to -- no majority vote needed),
    centroid = mean of member cells, elevation = max of member cells'
    floored elevation_max (matches the dashboards' own bbox-height
    convention). Clusters below LIVE_MIN_CLUSTER_CELLS are dropped, and
    same-class clusters within LIVE_CLUSTER_MERGE_RADIUS_M are merged,
    before CellRecord.center_x/y (ego-centric, grid.py's UGV-at-origin
    contract) are converted to world-frame using this frame's real
    ugv_world_pos -- TrackedObject/ObjectTracker expect world-frame."""
    ugv_wx, ugv_wy = ugv_world_pos
    detections = []
    for class_id, label in LIVE_CLASS_LABELS.items():
        cells = [rec for _, rec in active_map if rec.class_id == class_id]
        if len(cells) < LIVE_CLUSTER_MIN_SAMPLES:
            continue
        coords = np.array([(c.center_x, c.center_y) for c in cells])
        cluster_labels = DBSCAN(eps=LIVE_CLUSTER_EPS_M,
                                 min_samples=LIVE_CLUSTER_MIN_SAMPLES).fit(coords).labels_
        prelim = []
        for cid in set(cluster_labels):
            if cid == -1:
                continue
            member_cells = [c for c, cl in zip(cells, cluster_labels) if cl == cid]
            if len(member_cells) < LIVE_MIN_CLUSTER_CELLS:
                continue
            ego_x = float(np.mean([c.center_x for c in member_cells]))
            ego_y = float(np.mean([c.center_y for c in member_cells]))
            elevation = max(max(0.05, c.elevation_max) for c in member_cells)
            prelim.append((ego_x, ego_y, elevation, len(member_cells)))
        for ego_x, ego_y, elevation, _count in _merge_nearby_clusters(prelim, LIVE_CLUSTER_MERGE_RADIUS_M):
            detections.append((ugv_wx + ego_x, ugv_wy + ego_y, elevation, label))
    return detections


def _scan_corridor(active_map, heading_deg):
    """Is there an elevated obstacle in the rectangular corridor directly
    ahead of the UGV? Returns (blocked, closest_m, left_count, right_count).

    Cells arrive ego-centric but world-axis-aligned (driving_sequence.py
    subtracts the UGV position without rotating), so the corridor -- not the
    point cloud -- is what gets rotated into the heading frame. heading_deg
    follows the convention used throughout this module and radar_view_2d.py:
    measured from +Y, clockwise positive, so forward is (sin h, cos h) and
    +lateral is to the UGV's right."""
    cells = [rec for _, rec in active_map
             if rec.class_id != 0 and rec.elevation_max >= OBSTACLE_MIN_ELEVATION_M]
    if not cells:
        return False, None, 0, 0

    cx = np.fromiter((c.center_x for c in cells), dtype=np.float64, count=len(cells))
    cy = np.fromiter((c.center_y for c in cells), dtype=np.float64, count=len(cells))
    h = math.radians(heading_deg)
    sin_h, cos_h = math.sin(h), math.cos(h)
    forward = cx * sin_h + cy * cos_h
    lateral = cx * cos_h - cy * sin_h

    in_corridor = (forward > CORRIDOR_MIN_FORWARD_M) & (forward <= CORRIDOR_LOOKAHEAD_M) &                   (np.abs(lateral) <= CORRIDOR_HALF_WIDTH_M)
    n = int(in_corridor.sum())
    if n < CORRIDOR_MIN_CELLS:
        return False, None, 0, 0

    lat_hit = lateral[in_corridor]
    return (True,
            float(forward[in_corridor].min()),
            int((lat_hit < 0).sum()),
            int((lat_hit >= 0).sum()))


class ObjectTracker:
    """Nearest-centroid tracker: gives every detection a persistent ID
    across frames so the UI can draw motion trails and infer dynamic
    vs static without needing your model to emit track IDs itself.

    If your real model DOES emit persistent IDs, skip this class and
    populate TrackedObject.track_id directly in Engine.step()."""

    def __init__(self):
        self._next_id = itertools.count(1)
        self.active: dict[int, TrackedObject] = {}

    def update(self, detections, frame_id, dt):
        unmatched = list(range(len(detections)))

        for tid, trk in list(self.active.items()):
            best_j, best_dist = None, TRACK_MATCH_RADIUS_M
            for j in unmatched:
                dx, dy, dz, label = detections[j]
                dist = math.hypot(dx - trk.x, dy - trk.y)
                if dist < best_dist:
                    best_dist, best_j = dist, j
            if best_j is not None:
                dx, dy, dz, label = detections[best_j]
                raw_vx = (dx - trk.x) / dt if dt > 0 else 0.0
                raw_vy = (dy - trk.y) / dt if dt > 0 else 0.0
                # Clamp before smoothing, not after -- a single noisy match
                # (measured raw arrow lengths up to 352.6m / ~141 m/s) must
                # never enter the running average unclamped.
                raw_vx = max(-MAX_VELOCITY_MPS, min(MAX_VELOCITY_MPS, raw_vx))
                raw_vy = max(-MAX_VELOCITY_MPS, min(MAX_VELOCITY_MPS, raw_vy))
                if trk.hits <= 1:
                    # First real match since spawn: no prior average exists yet.
                    vx, vy = raw_vx, raw_vy
                else:
                    a = VELOCITY_EMA_ALPHA
                    vx = a * raw_vx + (1 - a) * trk.velocity[0]
                    vy = a * raw_vy + (1 - a) * trk.velocity[1]
                speed = math.hypot(vx, vy)

                trk.velocity = (vx, vy)
                trk.is_dynamic = speed > DYNAMIC_SPEED_THRESHOLD_MPS
                trk.history.append((trk.x, trk.y))
                if len(trk.history) > 10:
                    trk.history.pop(0)
                trk.x, trk.y, trk.z, trk.label = dx, dy, dz, label
                trk.age_frames += 1
                trk.last_seen_frame = frame_id
                trk.hits += 1
                unmatched.remove(best_j)

        for j in unmatched:
            dx, dy, dz, label = detections[j]
            tid = next(self._next_id)
            self.active[tid] = TrackedObject(
                track_id=tid, x=dx, y=dy, z=dz, label=label,
                history=[], velocity=(0.0, 0.0), is_dynamic=False,
                age_frames=0, last_seen_frame=frame_id,
            )

        stale = [tid for tid, trk in self.active.items()
                 if frame_id - trk.last_seen_frame > TRACK_MAX_AGE_FRAMES]
        for tid in stale:
            del self.active[tid]

        # self.active keeps every track (confirmed or not) for continued
        # matching/aging; only tracks that have persisted across at least
        # TRACK_CONFIRM_HITS detections are returned for rendering/threat
        # evaluation, so single-frame noise never becomes a visible track.
        return [trk for trk in self.active.values() if trk.hits >= TRACK_CONFIRM_HITS]


class Engine:
    """The ONLY object the UI talks to. Call .step() once per UI tick."""

    def __init__(self):
        self.live = LIVE_MODEL_AVAILABLE
        self.live_init_error = None
        self.tracker = ObjectTracker()
        self.frame_id = 0
        self._last_t = time.perf_counter()
        self._ugv_x, self._ugv_y = 0.0, 0.0
        self._ugv_heading = 0.0
        self._ugv_speed = 3.0          # m/s, cruise speed for demo
        self._evasion_active = False
        self._evasion_target_heading = None
        self._evasion_timer = 0.0
        self._live_active_map = None   # set by _live_step; stays None in synthetic mode
        self._sim_time = 0.0           # internal clock, advances by dt each step
                                        # (NOT wall-clock time.perf_counter() — using
                                        # raw wall-clock as a phase input made the
                                        # synthetic motion patterns barely change
                                        # across a short-lived run; this fixes that)

        if self.live:
            try:
                self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.model = SpikingPointNet(num_steps=10).to(self._device)
                checkpoint_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "snn_weights.pth"
                )
                if not os.path.exists(checkpoint_path):
                    raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
                self.model.load_state_dict(
                    torch.load(checkpoint_path, map_location=self._device, weights_only=True)
                )
                self.model.eval()
                self.profiler = EdgeProfiler()
                # Scene frames are now built per-step around the UGV's own
                # closed-loop pose (see _live_step), not replayed from a
                # prebuilt list -- but keep one persistent rng so per-frame
                # sensor noise still varies the way the prebuilt sequence's
                # single shared rng did.
                self._scene_rng = np.random.default_rng(2026)
                # Generated once and reused every frame -- see
                # build_static_ground's docstring for why redrawing it per
                # frame was the dominant source of display churn.
                self._static_ground = build_static_ground()
                # First real DBSCAN call costs ~2.0s while joblib spins up its
                # thread pool (measured: frame 2 of a 250-frame run took 2143ms
                # against a 67.6ms median). Paying it here keeps that one-off
                # freeze out of the first seconds of the live display.
                DBSCAN(eps=LIVE_CLUSTER_EPS_M,
                       min_samples=LIVE_CLUSTER_MIN_SAMPLES).fit(np.zeros((8, 2)))
                self._ugv_x, self._ugv_y = get_ugv_position(0)
                self._ugv_heading = 90.0    # +X world, the sequence's direction of travel
            except Exception as e:
                self.live = False
                self.live_init_error = str(e)
                print(f"[engine_adapter] live mode init failed, falling back to synthetic: {e}",
                      file=sys.stderr)

    # ------------------------------------------------------------------
    def step(self) -> Frame:
        now = time.perf_counter()
        dt = max(1e-3, now - self._last_t)
        self._last_t = now
        self.frame_id += 1

        # Clamp dt to a sane simulated range so behavior stays consistent
        # whether step() is called by a real-time UI timer (~33ms at 30fps)
        # or in a tight headless loop (sub-millisecond dt).
        dt = min(max(dt, 1.0 / 240.0), 1.0 / 5.0)
        self._sim_time += dt

        live_cells = None
        if self.live:
            detections, fps, latency_ms, spike_rate, sparsity = self._live_step()
            # The UGV's pose is NOT adopted from driving_sequence any more --
            # Engine owns it and _advance_ugv steers it below, which is what
            # makes this a closed loop (perception -> decision -> motion)
            # rather than a scripted straight line the model cannot influence.
            self._ugv_speed = LIVE_UGV_SPEED_MPS
            live_cells = self._live_active_map
        else:
            detections, fps, latency_ms, spike_rate, sparsity = self._synthetic_step(self._sim_time)

        # Stage 1 (class split): only "vehicle" detections are dynamic-threat-
        # eligible and go through the full tracker; everything else (static/
        # background classes) becomes a lightweight, untracked density point
        # -- no ID, no persistence, just "this class was seen here this frame".
        trackable = [d for d in detections if d[3] == "vehicle"]
        density_points = [(d[0], d[1], d[3]) for d in detections if d[3] != "vehicle"]

        confirmed = self.tracker.update(trackable, self.frame_id, dt)

        log_events = []
        threat = self._evaluate_threats(confirmed, log_events)

        # Static obstacles can never reach _evaluate_threats (it skips every
        # non-dynamic track), so the corridor scan is a second, parallel
        # trigger rather than a change to that function's semantics: it is
        # what lets the five poles -- and anything else with real height --
        # actually cause an avoidance manoeuvre.
        corridor = (False, None, 0, 0)
        if self.live and self._live_active_map is not None:
            corridor = _scan_corridor(self._live_active_map, self._ugv_heading)
        blocked, closest_m, left_n, right_n = corridor
        if blocked and not self._evasion_active:
            log_events.append(
                f"Obstacle in corridor at {closest_m:.1f}m - steering "
                f"{'right' if left_n >= right_n else 'left'}"
            )
        # Steer away from whichever side of the corridor holds more obstacle
        # cells; ties break right, matching _advance_ugv's original +35.
        turn_sign = 1.0 if left_n >= right_n else -1.0
        # Live mode steers on the corridor ONLY. Measured: _evaluate_threats
        # flags ~41% of rendered tracks as threats because its is_dynamic /
        # closing-speed inputs are ghost-contaminated (AUDIT-V3 §3.3), so
        # OR-ing it in here left evasion permanently engaged and the heading
        # spinning. The dynamic-threat flag still drives the red icon glow
        # and the event log; it just doesn't get a vote on the steering.
        steer_now = blocked if self.live else threat
        # Live mode integrates on scene time (see SCENE_DT_S); synthetic mode
        # is driven by a real-time timer and keeps using wall-clock dt.
        self._advance_ugv(SCENE_DT_S if self.live else dt, steer_now, log_events, turn_sign)

        # Stage 2 (persistence split): among confirmed vehicle tracks, only
        # ones that have proven stable over many real matches (or are an
        # active threat, regardless of hit count) get full icon treatment --
        # see RENDER_MIN_HITS's comment for why hits, not speed, is the
        # signal that actually separates real/stable detections from
        # transient re-clustering noise.
        # Both ends of a track's visible life are ramps rather than steps.
        # fade_in: how established the track is (hit count).
        # fade_out: how long since it was last actually matched -- a track that
        #   stops matching used to keep rendering at full strength, frozen at
        #   its last position, for TRACK_MAX_AGE_FRAMES and then vanish.
        # Threat tracks bypass the fade-in: a real hazard must not be dimmed.
        tracks = []
        fade_span = max(1, RENDER_MIN_HITS - RENDER_FADE_IN_HITS)
        for trk in confirmed:
            fade_in = (trk.hits - RENDER_FADE_IN_HITS) / fade_span
            fade_in = 1.0 if trk.threat else max(0.0, min(1.0, fade_in))
            stale = self.frame_id - trk.last_seen_frame
            fade_out = max(0.0, min(1.0, 1.0 - stale / TRACK_MAX_AGE_FRAMES))
            trk.render_alpha = min(fade_in, fade_out)
            if trk.render_alpha > 0.0:
                tracks.append(trk)
            else:
                density_points.append((trk.x, trk.y, trk.label))

        return Frame(
            frame_id=self.frame_id, t=now,
            ugv_x=self._ugv_x, ugv_y=self._ugv_y,
            ugv_heading_deg=self._ugv_heading, ugv_speed_mps=self._ugv_speed,
            fps=fps, latency_ms=latency_ms, spike_rate=spike_rate, sparsity=sparsity,
            tracks=tracks,
            evasion_active=self._evasion_active,
            evasion_target_heading_deg=self._evasion_target_heading,
            log_events=log_events,
            live_cells=live_cells,
            density_points=density_points,
            corridor_blocked=blocked,
            corridor_closest_m=closest_m,
        )

    # ------------------------------------------------------------------
    # Real live-mode step: replays driving_sequence.py's pre-built ego-
    # motion sequence through the real SpikingPointNet + grid engine,
    # mirroring dashboard_pro.py/dashboard_driving.py's proven sequence
    # exactly (scene_to_tensor -> model -> argmax/sum -> generate_2_5d_grid),
    # then clusters the resulting cells into discrete detections.
    # ------------------------------------------------------------------
    def _live_step(self):
        # Monotonic and fractional. This used to be
        # `(self.frame_id - 1) % LIVE_N_FRAMES`, which restarted the actors
        # every 40 frames and so had to teleport the UGV pose and evasion state
        # back to the start to match -- the entire display changed at once,
        # roughly every 5 seconds. driving_sequence's `continuous=True` recycles
        # each actor around the UGV outside detection range instead, so there is
        # no longer any discontinuity to resynchronise to.
        idx = (self.frame_id - 1) * SCENE_STEP
        if self.frame_id % LIVE_RESET_EVERY_N_FRAMES == 1:
            # driving_sequence.py's points are ego-centric to EACH frame's own
            # UGV position, not a fixed world origin -- as the UGV translates,
            # a static real-world object lands in a different grid cell every
            # frame, so GridState's cross-frame cell cache never naturally
            # reuses/evicts anything and grows without bound if left running
            # indefinitely. dashboard_driving.py avoids this by only ever
            # running the sequence once per button click (resetting
            # handoff._state at that single start); we loop the same
            # sequence forever and reset every LIVE_RESET_EVERY_N_FRAMES
            # instead -- measured: generate_2_5d_grid's snapshot() and this
            # method's per-class cell filter are both O(total accumulated
            # cells), so resetting only once per scene replay let step time
            # grow from ~57ms to ~360ms across one cycle. Resetting more
            # often caps that growth while still preserving several frames'
            # worth of the event-driven caching behavior within each window.
            handoff._state = GridState()
        # Built per-frame around the UGV's own pose rather than replayed from
        # a prebuilt list: once steering can move the vehicle off the scripted
        # line, the scene has to be made ego-centric to where it ACTUALLY is.
        # Measured cost 9.8ms/frame, ~8% of the step budget.
        ugv_world_pos = (self._ugv_x, self._ugv_y)
        points, _labels_gt, _spikes_gt, _ = build_driving_frame(
            idx, rng=self._scene_rng, ugv_pos=ugv_world_pos,
            ground=self._static_ground, continuous=True)

        t0 = time.perf_counter()
        norm_tensor, raw_sampled = _scene_to_tensor(points)
        inputs = torch.tensor(norm_tensor, dtype=torch.float32).unsqueeze(0).to(self._device)
        with torch.no_grad():
            spk_rec = self.model(inputs)
        total_spikes = spk_rec.sum(dim=0)
        preds_np = torch.argmax(total_spikes, dim=1).squeeze().cpu().numpy()
        spikes_np = total_spikes.sum(dim=1).squeeze().cpu().numpy()
        elapsed = time.perf_counter() - t0

        active_map = generate_2_5d_grid(raw_sampled, preds_np, spikes_np)
        perf = self.profiler.evaluate_efficiency(spk_rec, elapsed, 1)

        # generate_2_5d_grid returns GridState's FULL accumulated snapshot --
        # every cell cached since the last reset, which is correct for the grid
        # engine's event-driven persistence contract but wrong to consume here.
        # A CellRecord's center_x/y is ego-centric to the frame that wrote it,
        # and driving_sequence.py's UGV moves 1.5 m/s, so a cell cached 9 frames
        # ago describes a position relative to a UGV 13.5m behind this one.
        # Converting the whole snapshot with only THIS frame's ugv_world_pos
        # smears every static object into a ~13.5m streak of stale ghost cells;
        # DBSCAN then shatters each streak into several clusters, each becoming
        # its own tracked "object" with a drifting centroid that reads as
        # motion. Measured: detections climbed 5 -> 60 per frame across one
        # LIVE_RESET_EVERY_N_FRAMES window (a sawtooth locked to the reset, not
        # to anything in the scene), and frame.tracks averaged 36.6 for a scene
        # containing 7 real objects. Keeping only cells touched this frame is
        # flat at 3-9 detections and 7.6 tracks/frame.
        # last_touched_frame is set for every cell assigned this frame (not just
        # spiking ones -- grid_state.py:81-89), so this preserves the cached-
        # value semantics of event-driven updates; it drops stale geometry only.
        # GridState.update() calls move_to_end() for every cell it touches, so
        # the snapshot's LAST entry is always one touched this frame -- O(1)
        # instead of scanning all ~58k cached records for the max.
        current_frame = active_map[-1][1].last_touched_frame if active_map else 0
        fresh_map = [(key, rec) for key, rec in active_map
                     if rec.last_touched_frame == current_frame]

        self._live_ugv_world_pos = ugv_world_pos
        # Also what terrain_relief.py consumes via Frame.live_cells -- the
        # near-field panel inherits the same staleness otherwise.
        self._live_active_map = fresh_map

        detections = _cluster_cells_to_detections(fresh_map, ugv_world_pos)

        total_ops = perf["ac_ops"] + perf["mac_ops_avoided"]
        fps = perf["fps"]
        latency_ms = perf["latency_sec"] * 1000.0
        spike_rate = perf["ac_ops"] / total_ops if total_ops > 0 else 0.0
        sparsity = perf["sparsity_pct"] / 100.0
        return detections, fps, latency_ms, spike_rate, sparsity

    # ------------------------------------------------------------------
    # Placeholder per-object elevations (meters), same demo-data spirit as
    # the rest of this method -- not real sensor output. Indexed by the
    # static loop's i=0..4 (label pattern: vegetation, obstacle,
    # vegetation, obstacle, vegetation).
    _STATIC_ELEVATIONS_M = [2.5, 0.4, 3.5, 1.0, 1.8]

    def _synthetic_step(self, t):
        rng = np.random.default_rng(self.frame_id % 100000)

        n_static = 5
        n_dynamic = 3
        detections = []

        static_seed = np.random.default_rng(42)
        for i in range(n_static):
            ang = static_seed.uniform(0, 2 * np.pi)
            r = static_seed.uniform(15, SENSOR_RANGE_M * 0.85)
            x, y = r * math.cos(ang), r * math.sin(ang)
            label = "vegetation" if i % 2 == 0 else "obstacle"
            detections.append((x, y, self._STATIC_ELEVATIONS_M[i], label))

        for i in range(n_dynamic):
            # Each dynamic object follows a slow elliptical path that
            # periodically swings close to the UGV then retreats, so
            # evasion becomes an occasional dramatic event rather than
            # a permanently-active state. Uses simulated time `t`
            # (seconds since Engine started), not wall-clock time.
            phase = t * 0.25 + i * 2.4
            r = 18 + 10 * i + 12 * math.sin(t * 0.15 + i)
            x = r * math.cos(phase)
            y = r * math.sin(phase) * 0.7
            detections.append((x, y, 0.0, "vehicle"))

        fps = TARGET_FPS + rng.uniform(-4, 4)
        latency_ms = 1000.0 / fps
        spike_rate = float(np.clip(0.35 + 0.15 * math.sin(t * 2.0), 0.05, 0.95))
        sparsity = float(np.clip(0.92 + 0.03 * math.sin(t * 1.3), 0.80, 0.99))
        return detections, fps, latency_ms, spike_rate, sparsity

    # ------------------------------------------------------------------
    def _evaluate_threats(self, tracks, log_events):
        threat_found = False
        for trk in tracks:
            was_threat = trk.threat
            trk.threat = False
            if not trk.is_dynamic:
                continue
            rel_x, rel_y = trk.x - self._ugv_x, trk.y - self._ugv_y
            dist = math.hypot(rel_x, rel_y)
            if dist > 40:
                continue
            vx, vy = trk.velocity
            speed = math.hypot(vx, vy)
            # closing_speed > 0 means the object's net motion is toward
            # the UGV. Tuned (0.15 m/s threshold, 22m radius) against a
            # 4000-frame simulation to produce occasional, clean evasion
            # events rather than a near-constant or never-triggering state.
            closing_speed = -(rel_x * vx + rel_y * vy) / max(dist, 1e-6)
            is_threat = closing_speed > 0.15 and dist < 22
            if is_threat and speed > 0:
                trk.threat = True
                threat_found = True
                if not was_threat:
                    bearing = math.degrees(math.atan2(rel_x, rel_y))
                    log_events.append(
                        f"[frame {trk.age_frames}] Dynamic obstacle #{trk.track_id} at {dist:.1f}m, "
                        f"bearing {bearing:+.0f}deg - evasive heading engaged"
                    )
        return threat_found

    def _advance_ugv(self, dt, threat_now, log_events, turn_sign=1.0):
        # Nominal cruise heading differs by mode: live follows
        # driving_sequence's +X corridor (90), synthetic cruises +Y (0).
        if self.live:
            # Nominal heading is not a constant: it leans back toward the
            # corridor centreline (world y=0) in proportion to how far off it
            # the UGV currently is. heading > 90 steers toward -y, so a
            # positive lateral error needs a positive offset.
            lean = max(-LANE_RETURN_MAX_DEG,
                       min(LANE_RETURN_MAX_DEG, LANE_RETURN_GAIN_DEG_PER_M * self._ugv_y))
            nominal = (90.0 + lean) % 360
        else:
            nominal = 0.0
        hold = EVASION_HOLD_S if self.live else 1.6
        if threat_now:
            # Latch the target once per engagement, as a bounded offset from
            # the NOMINAL heading. Recomputing it from the current heading on
            # every frame the obstacle stays visible compounds the turn (+35
            # per frame) into an uncontrolled spin.
            if not self._evasion_active:
                self._evasion_active = True
                self._evasion_target_heading = (
                    nominal + EVASION_TURN_DEG * turn_sign) % 360
            self._evasion_timer = hold
        elif self._evasion_active:
            self._evasion_timer -= dt
            if self._evasion_timer <= 0:
                self._evasion_active = False
                self._evasion_target_heading = None
                log_events.append("Path clear - resuming planned heading")

        target = self._evasion_target_heading if self._evasion_active else nominal
        diff = (target - self._ugv_heading + 180) % 360 - 180
        if self.live:
            # Rate-limit on scene time. The synthetic branch's proportional
            # step assumes a ~33ms tick and would snap the heading instantly
            # at SCENE_DT_S=1.0.
            step = LIVE_TURN_RATE_DEG_S * dt
            self._ugv_heading = (self._ugv_heading + max(-step, min(step, diff))) % 360
        else:
            self._ugv_heading = (self._ugv_heading + diff * min(1.0, dt * 2.0)) % 360

        heading_rad = math.radians(self._ugv_heading)
        self._ugv_x += math.sin(heading_rad) * self._ugv_speed * dt
        self._ugv_y += math.cos(heading_rad) * self._ugv_speed * dt