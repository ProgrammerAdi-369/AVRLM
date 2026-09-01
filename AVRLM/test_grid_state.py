from radial_filter import prefilter_mask
from grid import VariableResolutionGrid
from grid_state import GridState
from synthetic_lidar_data import build_moving_sequence, LABEL_DYNAMIC

START_XY = (6.0, -3.0)
VELOCITY = (0.4, 0.15)

frames = build_moving_sequence(n_frames=10, start_xy=START_XY, velocity=VELOCITY)
state = GridState()
probe_grid = VariableResolutionGrid()


def cluster_cell_keys(points, labels):
    """Ground truth: which cell keys the frame's LABEL_DYNAMIC points fall
    into, computed independently of GridState's internal bookkeeping."""
    cluster_pts = points[labels == LABEL_DYNAMIC]
    mask = prefilter_mask(cluster_pts)
    cluster_pts = cluster_pts[mask]
    assignment = probe_grid.assign_cells(cluster_pts)
    keys = set(zip(
        assignment.parent_ix.tolist(), assignment.parent_iy.tolist(),
        assignment.sub_ix.tolist(), assignment.sub_iy.tolist(),
    ))
    return keys


for t, (pts, lbl, spk) in enumerate(frames):
    before = dict(state._cells)  # key -> CellRecord object refs, pre-update
    touched = state.update(pts, lbl, spk)

    # --- Core invariant (post cold-start fix): a zero-spike cell that was
    # ALREADY cached keeps its exact pre-update value (robust to
    # ground-plane's own random spiking and pole's always-nonzero spiking -
    # see plan Task 2 note). A zero-spike cell seen for the FIRST time is
    # now committed as a baseline record instead of staying absent. ---
    for key, spike_sum in touched.items():
        if spike_sum == 0:
            if key in before:
                assert state._cells[key] is before[key], \
                    f"frame {t}: zero-spike cached cell {key} was overwritten"
            else:
                assert key in state._cells, \
                    f"frame {t}: zero-spike first-sight cell {key} was not baselined"

    # --- Regression check for the cold-start gating bug: frame 0 must
    # ingest EVERY touched cell as baseline, not just the spiking subset. ---
    if t == 0:
        assert len(state._cells) == len(touched), \
            f"frame 0: expected full baseline ({len(touched)} cells), got {len(state._cells)}"

    # --- Cells the moving cluster's points actually land in this frame
    # must update (dynamic cluster spikes are always >0 by construction). ---
    cluster_keys = cluster_cell_keys(pts, lbl)
    assert len(cluster_keys) > 0, f"frame {t}: expected cluster to touch at least one cell"
    for key in cluster_keys:
        assert touched.get(key, 0) > 0, f"frame {t}: cluster cell {key} failed to update"

    n_changed = sum(1 for s in touched.values() if s > 0)
    n_elsewhere_changed = n_changed - len(cluster_keys)
    print(f"frame {t}: touched={len(touched)}, changed={n_changed}, "
          f"cluster_cells={len(cluster_keys)}, other_changed={n_elsewhere_changed}, "
          f"total_cached={len(state._cells)}")


# --- Dedicated regression check for the reported bug: build_scene() through
# a fresh GridState must recover the true ungated occupied-cell count, not
# just the ~3% of cells that happened to spike (see
# debug_guide_cell_count_discrepancy.md). ---
from aggregate import aggregate_cells
from synthetic_lidar_data import build_scene

scene_points, scene_labels, scene_spikes = build_scene()
scene_mask = prefilter_mask(scene_points)
f_scene_points = scene_points[scene_mask]
f_scene_labels = scene_labels[scene_mask]
f_scene_spikes = scene_spikes[scene_mask]
true_assignment = probe_grid.assign_cells(f_scene_points)
true_stats = aggregate_cells(f_scene_points, f_scene_labels, f_scene_spikes, true_assignment)
true_ungated_count = len(true_stats)

fresh_state = GridState()
fresh_state.update(scene_points, scene_labels, scene_spikes)
print(f"build_scene() cold-start regression check: true_ungated={true_ungated_count}, "
      f"GridState_after_one_update={len(fresh_state._cells)}")
assert len(fresh_state._cells) == true_ungated_count, \
    f"cold-start baseline mismatch: expected {true_ungated_count}, got {len(fresh_state._cells)}"

print("test_grid_state.py: all assertions passed")
