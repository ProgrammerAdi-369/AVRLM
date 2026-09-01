"""
Synthetic LiDAR frame generator for testing the Variable Resolution Grid Engine
(Member 3) without needing Member 1's trained model or Member 2's real dataset.

Matches the confirmed team interface contract:
    points  : (N, 4) float32  -> X, Y, Z, Intensity (ego-centric meters, UGV at origin)
    labels  : (N,)   int64    -> 0 = drivable terrain, 1 = static obstacle, 2 = dynamic object
    spikes  : (N,)   uint8    -> spike COUNT per point this frame (0 = neuron never fired)

Grid zones: 0-10m -> 5cm cells, 10-100m -> 50cm cells.

Each generator function below exists to stress-test one specific part of the
grid engine:

    generate_ground_plane        -> baseline: flat terrain, mostly silent
                                     (tests: radial filter, event-driven "skip"
                                     path for non-spiking cells)
    generate_pole                -> a static obstacle
                                     (tests: elevation/height-variance calc,
                                     majority-class aggregation per cell)
    generate_dynamic_cluster     -> a moving object
                                     (tests: dynamic-class handling, dense spikes)
    generate_boundary_stress_ring-> a dense ring exactly at r=10m
                                     (tests: the 5cm/50cm seam directly - the
                                     thing the whole quadtree design exists for)
    build_moving_sequence        -> multi-frame sequence, cluster physically
                                     moves each frame
                                     (tests: event-driven updates ACROSS frames
                                     - only cells under the moving object should
                                     refresh; everything else must stay cached)
"""

import numpy as np

RNG = np.random.default_rng(42)

LABEL_DRIVABLE = 0
LABEL_STATIC = 1
LABEL_DYNAMIC = 2


def _lidar_like_ring_sample(n_rings, pts_per_ring, r_min, r_max, rng):
    """
    Approximates real LiDAR density falloff: roughly constant points-per-ring,
    so density (points per m^2) drops off with radius, like a real spinning
    LiDAR rather than uniform random scatter.
    """
    radii = rng.uniform(r_min, r_max, size=n_rings)
    xs, ys = [], []
    for r in radii:
        angles = rng.uniform(0, 2 * np.pi, size=pts_per_ring)
        xs.append(r * np.cos(angles))
        ys.append(r * np.sin(angles))
    return np.concatenate(xs), np.concatenate(ys)


def generate_ground_plane(r_max=100.0, n_rings=600, pts_per_ring=40, z_noise=0.02, rng=RNG):
    """Flat drivable terrain filling the full 100m radius. Label 0, mostly silent."""
    x, y = _lidar_like_ring_sample(n_rings, pts_per_ring, 0.3, r_max, rng)
    n = x.shape[0]
    z = rng.normal(0.0, z_noise, size=n).astype(np.float32)
    intensity = rng.uniform(0.1, 0.4, size=n).astype(np.float32)
    points = np.stack([x, y, z, intensity], axis=1).astype(np.float32)
    labels = np.full(n, LABEL_DRIVABLE, dtype=np.int64)
    # Ground is "boring" -> SNN mostly stays silent. Small % spikes = sensor noise / tiny bumps.
    spikes = (rng.random(n) < 0.02).astype(np.uint8)
    return points, labels, spikes


def generate_pole(center_xy, height=2.0, radius=0.08, n_points=120, rng=RNG):
    """A vertical static obstacle (pole / tree trunk). Label 1, fires - it's a new shape."""
    cx, cy = center_xy
    z = rng.uniform(0.0, height, size=n_points).astype(np.float32)
    angle = rng.uniform(0, 2 * np.pi, size=n_points)
    x = (cx + radius * np.cos(angle)).astype(np.float32)
    y = (cy + radius * np.sin(angle)).astype(np.float32)
    intensity = rng.uniform(0.4, 0.9, size=n_points).astype(np.float32)
    points = np.stack([x, y, z, intensity], axis=1).astype(np.float32)
    labels = np.full(n_points, LABEL_STATIC, dtype=np.int64)
    spikes = rng.integers(1, 4, size=n_points).astype(np.uint8)
    return points, labels, spikes


def generate_dynamic_cluster(center_xy, n_points=60, spread=0.35, height=1.7, rng=RNG):
    """A moving blob (pedestrian-ish). Label 2, fires heavily - motion means high spike counts."""
    cx, cy = center_xy
    x = rng.normal(cx, spread, size=n_points).astype(np.float32)
    y = rng.normal(cy, spread, size=n_points).astype(np.float32)
    z = rng.uniform(0.0, height, size=n_points).astype(np.float32)
    intensity = rng.uniform(0.3, 0.7, size=n_points).astype(np.float32)
    points = np.stack([x, y, z, intensity], axis=1).astype(np.float32)
    labels = np.full(n_points, LABEL_DYNAMIC, dtype=np.int64)
    spikes = rng.integers(3, 8, size=n_points).astype(np.uint8)
    return points, labels, spikes


def generate_boundary_stress_ring(radius=10.0, n_points=400, jitter=0.02, rng=RNG):
    """
    Dense ring of points straddling the exact 5cm/50cm transition at r=10m.
    Purpose-built to catch alignment/seam bugs where the two resolutions meet -
    this is the single most important test scene for your quadtree logic.
    """
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    r = radius + rng.uniform(-jitter, jitter, size=n_points)
    x = (r * np.cos(angles)).astype(np.float32)
    y = (r * np.sin(angles)).astype(np.float32)
    z = rng.normal(0.0, 0.01, size=n_points).astype(np.float32)
    intensity = rng.uniform(0.1, 0.3, size=n_points).astype(np.float32)
    points = np.stack([x, y, z, intensity], axis=1).astype(np.float32)
    labels = np.full(n_points, LABEL_DRIVABLE, dtype=np.int64)
    spikes = np.zeros(n_points, dtype=np.uint8)
    return points, labels, spikes


def build_scene(include_boundary_stress=True, rng=RNG):
    """One complete synthetic frame: ground + a couple of poles + one dynamic cluster."""
    parts = [
        generate_ground_plane(rng=rng),
        generate_pole((4.0, 2.5), rng=rng),
        generate_pole((15.0, -8.0), rng=rng),
        generate_dynamic_cluster((6.0, -3.0), rng=rng),
    ]
    if include_boundary_stress:
        parts.append(generate_boundary_stress_ring(rng=rng))

    points = np.concatenate([p[0] for p in parts], axis=0)
    labels = np.concatenate([p[1] for p in parts], axis=0)
    spikes = np.concatenate([p[2] for p in parts], axis=0)
    return points, labels, spikes


def build_moving_sequence(n_frames=10, start_xy=(6.0, -3.0), velocity=(0.4, 0.15), rng=RNG):
    """
    Multi-frame sequence where the dynamic cluster physically moves each frame.
    Use this to test event-driven updates across time: after frame 0, only the
    grid cells under the moving cluster (plus the static pole once, on frame 0)
    should update on later frames - everything else should stay cached.
    """
    frames = []
    for t in range(n_frames):
        cx = start_xy[0] + velocity[0] * t
        cy = start_xy[1] + velocity[1] * t
        ground = generate_ground_plane(rng=rng)
        pole = generate_pole((4.0, 2.5), rng=rng)
        moving = generate_dynamic_cluster((cx, cy), rng=rng)
        pts = np.concatenate([ground[0], pole[0], moving[0]], axis=0)
        lbl = np.concatenate([ground[1], pole[1], moving[1]], axis=0)
        spk = np.concatenate([ground[2], pole[2], moving[2]], axis=0) 
        frames.append((pts, lbl, spk))
    return frames


if __name__ == "__main__":
    pts, lbl, spk = build_scene()
    r = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
    print(f"points {pts.shape} {pts.dtype} | labels {lbl.shape} {lbl.dtype} | spikes {spk.shape} {spk.dtype}")
    print(f"radius range: {r.min():.2f}m to {r.max():.2f}m")
    print(f"points near the 10m boundary (9.9-10.1m): {((r > 9.9) & (r < 10.1)).sum()}")
    print(f"label counts: drivable={np.sum(lbl==0)}, static={np.sum(lbl==1)}, dynamic={np.sum(lbl==2)}")
    print(f"points that spiked this frame: {np.sum(spk > 0)} / {len(spk)}")
