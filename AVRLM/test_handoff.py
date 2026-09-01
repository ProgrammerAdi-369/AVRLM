from handoff import generate_2_5d_grid, memory_metrics
from synthetic_lidar_data import build_scene, build_moving_sequence

points, labels, spikes = build_scene()
grid_out = generate_2_5d_grid(points, labels, spikes)

print(f"generate_2_5d_grid(build_scene()) -> {len(grid_out)} cells")
key, record = grid_out[0]
print(f"schema: key={key} (parent_ix, parent_iy, sub_ix, sub_iy)")
print(f"record fields: is_fine={record.is_fine}, center_x={record.center_x:.3f}, "
      f"center_y={record.center_y:.3f}, elevation_max={record.elevation_max:.3f}, "
      f"elevation_var={record.elevation_var:.5f}, class_id={record.class_id}, "
      f"point_count={record.point_count}")

metrics = memory_metrics()
print(f"memory_metrics after build_scene(): {metrics}")

frames = build_moving_sequence(n_frames=1)
fpoints, flabels, fspikes = frames[0]
grid_out2 = generate_2_5d_grid(fpoints, flabels, fspikes)
print(f"generate_2_5d_grid(one moving-sequence frame) -> {len(grid_out2)} cells "
      f"(accumulated on top of build_scene() call above)")

metrics2 = memory_metrics()
print(f"memory_metrics after second call: {metrics2}")
assert metrics2["active_cell_count"] >= metrics["active_cell_count"], \
    "state should accumulate across calls, not reset"

print("test_handoff.py: all assertions passed")
