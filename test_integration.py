import time

from grid_state import GridState
from synthetic_lidar_data import build_moving_sequence


def test_timing_budget():
    frames = build_moving_sequence(n_frames=10)
    state = GridState()

    REALTIME_BUDGET_S = 1.0 / 10  # sanity threshold: 10 fps, not a real benchmark

    print("frame | wall_time_ms | cells_touched | cells_changed")
    frame_times = []
    for t, (pts, lbl, spk) in enumerate(frames):
        start = time.perf_counter()
        touched = state.update(pts, lbl, spk)
        elapsed = time.perf_counter() - start
        frame_times.append(elapsed)

        n_changed = sum(1 for s in touched.values() if s > 0)
        flag = "  <-- SLOWER THAN 10fps BUDGET" if elapsed > REALTIME_BUDGET_S else ""
        print(f"{t:5d} | {elapsed * 1000:10.2f} | {len(touched):13d} | {n_changed:13d}{flag}")

    print(f"final accumulated cell count: {len(state._cells)}")

    # --- Timing budget regression guard (AUDIT-v2 test-rebuild Phase 1). Uses a
    # tolerance (>=8/10 frames under budget), not exact ms thresholds, since
    # timing has documented run-to-run variance (Reports/AUDIT-v2.md §9 Phase 7
    # final regression: 9/10 under 100ms, worst case 106.0ms) - not hardcoding
    # those specific ms values here on purpose. ---
    under_budget = sum(1 for t in frame_times if t <= REALTIME_BUDGET_S)
    assert under_budget >= 8, \
        f"only {under_budget}/10 frames under the {REALTIME_BUDGET_S * 1000:.0f}ms budget"
    print(f"timing budget: {under_budget}/10 frames under {REALTIME_BUDGET_S * 1000:.0f}ms")


if __name__ == "__main__":
    test_timing_budget()
    print("test_integration.py: all assertions passed")
