import numpy as np

from radial_filter import prefilter_mask
from grid import VariableResolutionGrid
from aggregate import aggregate_cells
from synthetic_lidar_data import generate_pole, generate_ground_plane

grid = VariableResolutionGrid()


def run_pipeline(points, labels, spikes):
    mask = prefilter_mask(points)
    points, labels, spikes = points[mask], labels[mask], spikes[mask]
    assignment = grid.assign_cells(points)
    return aggregate_cells(points, labels, spikes, assignment)


# --- Pole scene: static obstacle, height 2.0m ---
pole_center = (4.0, 2.5)
pole_height = 2.0
pts, lbl, spk = generate_pole(pole_center, height=pole_height)
stats = run_pipeline(pts, lbl, spk)

dist_to_pole = np.sqrt((stats.center_x - pole_center[0]) ** 2 + (stats.center_y - pole_center[1]) ** 2)
pole_cells = dist_to_pole <= 0.15  # pole radius 0.08m + a little slack

n_pole_cells = int(pole_cells.sum())
print(f"pole scene: {len(stats)} occupied cells total, {n_pole_cells} near the pole")
assert n_pole_cells > 0, "expected at least one occupied cell near the pole"

max_elev_near_pole = stats.elevation_max[pole_cells].max()
print(f"pole scene: max elevation near pole = {max_elev_near_pole:.3f} (pole height = {pole_height})")
assert abs(max_elev_near_pole - pole_height) < 0.3, "pole max elevation should be close to pole height"
assert np.all(stats.class_id[pole_cells] == 1), "pole cells should be class_id 1 (static obstacle)"

# --- Ground plane scene: flat drivable terrain near z=0 ---
gpts, glbl, gspk = generate_ground_plane(z_noise=0.02)
gstats = run_pipeline(gpts, glbl, gspk)

sample = np.arange(min(20, len(gstats)))
print("ground plane sample cells (elevation_max, elevation_var, class_id, point_count):")
for i in sample:
    print(f"  {gstats.elevation_max[i]:.4f}, {gstats.elevation_var[i]:.6f}, "
          f"{gstats.class_id[i]}, {gstats.point_count[i]}")

assert np.all(np.abs(gstats.elevation_max) < 0.2), "ground elevation should stay near 0"
assert np.all(gstats.elevation_var < 0.01), "ground height variance should be low"
assert np.all(gstats.class_id == 0), "ground cells should be class_id 0 (drivable)"

print("test_aggregate.py: all assertions passed")
