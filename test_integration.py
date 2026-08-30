import time

from grid_state import GridState
from synthetic_lidar_data import build_moving_sequence

frames = build_moving_sequence(n_frames=10)
state = GridState()

REALTIME_BUDGET_S = 1.0 / 10  # sanity threshold: 10 fps, not a real benchmark

print("frame | wall_time_ms | cells_touched | cells_changed")
for t, (pts, lbl, spk) in enumerate(frames):
    start = time.perf_counter()
    touched = state.update(pts, lbl, spk)
    elapsed = time.perf_counter() - start

    n_changed = sum(1 for s in touched.values() if s > 0)
    flag = "  <-- SLOWER THAN 10fps BUDGET" if elapsed > REALTIME_BUDGET_S else ""
    print(f"{t:5d} | {elapsed * 1000:10.2f} | {len(touched):13d} | {n_changed:13d}{flag}")

print(f"final accumulated cell count: {len(state._cells)}")
print("test_integration.py: done (sanity check only, not a benchmark)")
