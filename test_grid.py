import numpy as np

from grid import VariableResolutionGrid, INNER_RES, OUTER_RES, INNER_RADIUS, OUTER_RADIUS, SUBDIV_FACTOR
from radial_filter import radial_filter
from synthetic_lidar_data import build_scene, generate_boundary_stress_ring

assert SUBDIV_FACTOR == 10, "OUTER_RES / INNER_RES must compute to 10"
print(f"SUBDIV_FACTOR computed as {SUBDIV_FACTOR} (OUTER_RES={OUTER_RES}, INNER_RES={INNER_RES})")

grid = VariableResolutionGrid()

# --- Boundary stress ring: sub-cell containment must hold exactly ---
ring_points, ring_labels, ring_spikes = generate_boundary_stress_ring()
assignment = grid.assign_cells(ring_points)

fine_idx = np.where(assignment.is_fine)[0]
print(f"boundary_stress_ring(): N={ring_points.shape[0]}, "
      f"fine(subdivided)={fine_idx.shape[0]}, coarse={ring_points.shape[0] - fine_idx.shape[0]}")

TOL = 1e-6
for i in fine_idx:
    parent_origin_x = assignment.parent_ix[i] * OUTER_RES - OUTER_RADIUS
    parent_origin_y = assignment.parent_iy[i] * OUTER_RES - OUTER_RADIUS

    sub_ix = assignment.sub_ix[i]
    sub_iy = assignment.sub_iy[i]
    assert 0 <= sub_ix < SUBDIV_FACTOR, f"sub_ix {sub_ix} out of range at point {i}"
    assert 0 <= sub_iy < SUBDIV_FACTOR, f"sub_iy {sub_iy} out of range at point {i}"

    sub_min_x = parent_origin_x + sub_ix * INNER_RES
    sub_max_x = sub_min_x + INNER_RES
    sub_min_y = parent_origin_y + sub_iy * INNER_RES
    sub_max_y = sub_min_y + INNER_RES

    assert parent_origin_x - TOL <= sub_min_x, f"point {i}: sub_min_x below parent origin"
    assert sub_max_x <= parent_origin_x + OUTER_RES + TOL, f"point {i}: sub_max_x exceeds parent extent"
    assert parent_origin_y - TOL <= sub_min_y, f"point {i}: sub_min_y below parent origin"
    assert sub_max_y <= parent_origin_y + OUTER_RES + TOL, f"point {i}: sub_max_y exceeds parent extent"

print(f"containment check: all {fine_idx.shape[0]} subdivided points' sub-cells "
      f"fall exactly within their parent cell bounds (tol={TOL})")

# --- Full pipeline sanity: build_scene() -> radial_filter -> grid ---
points, labels, spikes = build_scene()
inner_mask, outer_mask = radial_filter(points)
full_assignment = grid.assign_cells(points)

parent_cells = set(zip(full_assignment.parent_ix.tolist(), full_assignment.parent_iy.tolist()))
fine_mask = full_assignment.is_fine
sub_cells = set(zip(
    full_assignment.parent_ix[fine_mask].tolist(),
    full_assignment.parent_iy[fine_mask].tolist(),
    full_assignment.sub_ix[fine_mask].tolist(),
    full_assignment.sub_iy[fine_mask].tolist(),
))

print(f"build_scene() pipeline: total points={points.shape[0]}, "
      f"inner(radial)={inner_mask.sum()}, outer(radial)={outer_mask.sum()}, "
      f"unique parent cells touched={len(parent_cells)}, "
      f"unique sub-cells touched={len(sub_cells)}")

# --- Task 0: out-of-range point fed directly to assign_cells() ---
out_of_range_point = np.array([[150.0, 0.0, 0.0, 0.0]], dtype=np.float32)
oor_assignment = grid.assign_cells(out_of_range_point)

assert oor_assignment.in_range[0] == False, "r=150 point should be flagged out of range"
assert oor_assignment.parent_ix[0] == -1
assert oor_assignment.parent_iy[0] == -1
assert oor_assignment.is_fine[0] == False
assert oor_assignment.sub_ix[0] == -1
assert oor_assignment.sub_iy[0] == -1

oor_parent_cells = set(zip(
    oor_assignment.parent_ix[oor_assignment.in_range].tolist(),
    oor_assignment.parent_iy[oor_assignment.in_range].tolist(),
))
assert len(oor_parent_cells) == 0, "out-of-range point must not appear in unique parent cell count"

print("out-of-range point (r=150): in_range=False, all indices=-1, "
      "excluded from unique parent cell count - handled safely")

print("test_grid.py: all assertions passed")
