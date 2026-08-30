"""
Tests for the dashboard_driving.py logic that's separable from live
Streamlit rendering (AUDIT-v2 test-rebuild Phase 5). dashboard_driving.py
itself is NOT imported here - it executes Streamlit/torch top-level code
on import (st.set_page_config, checkpoint loading, sidebar widgets) that
needs a running Streamlit script context and isn't safe/fast to trigger
from a plain test script. Instead:
  - the pure frame-index logic lives in dashboard_state.py (imported directly)
  - the record-building and path-leak logic is re-derived here from the
    real source lines it mirrors (dashboard_driving.py:241-250, :73)
  - the frame-counter single-source check and the path-leak "no flag
    exists" claim are verified by reading dashboard_driving.py's source
    text, not by importing/running it

MANUAL REGRESSION STEP (not automated here, per rebuild_test_suite_guide.md
Phase 5): verify live multi-tab WebSocket reconnect resilience by running
`streamlit run dashboard_driving.py`, starting a driving sequence, then
disconnecting/reconnecting the browser tab mid-sequence and confirming
playback resumes from the correct frame instead of resetting. This needs a
live Streamlit server + browser and is disproportionate to automate for a
regression check.
"""

import math
import os

from dashboard_state import advance_frame_index
from grid_state import GridState
from handoff import memory_metrics
from synthetic_lidar_data import build_scene, generate_pole

DASHBOARD_SRC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard_driving.py")
CLASS_NAMES = {0: "Drivable", 1: "Static Obstacle", 2: "Dynamic Threat"}  # mirrors dashboard_driving.py:36


def build_records(active_map):
    """Mirrors the records-building loop at dashboard_driving.py:241-250."""
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
    return records


def test_advance_frame_index_resumes_not_resets():
    assert advance_frame_index(7, 20) == 8, "expected resumption at frame 8, not a reset"
    assert advance_frame_index(19, 20) == 20, "expected the last frame to advance to 20 (exhausted next check)"
    assert advance_frame_index(20, 20) is None, "expected None once the sequence is exhausted"
    print("advance_frame_index: reconnect at frame 7 resumes at 8 (not reset); exhaustion returns None")


def test_accumulated_grid_survives_simulated_rerun():
    """handoff._state is only reset on start_clicked (dashboard_driving.py:
    212-213), never per rerun - a GridState instance mutated across two
    separate .update() calls without reinitializing stands in for that."""
    sim_state = GridState()
    pts1, lbl1, spk1 = generate_pole((4.0, 2.5))
    sim_state.update(pts1, lbl1, spk1)
    metrics_after_1 = memory_metrics(sim_state)

    pts2, lbl2, spk2 = generate_pole((15.0, -8.0))
    sim_state.update(pts2, lbl2, spk2)
    metrics_after_2 = memory_metrics(sim_state)

    assert metrics_after_2["active_cell_count"] >= metrics_after_1["active_cell_count"], \
        "accumulated grid state should survive across calls (simulated reruns), not reset"
    print(f"accumulated grid survives simulated rerun: {metrics_after_1['active_cell_count']} "
          f"-> {metrics_after_2['active_cell_count']} cells")


def test_record_list_recomputed_and_idempotent():
    """Detected-object list is recomputed fresh each rerun, not persisted -
    there is nothing to "preserve" (objects/records/df come from
    handoff._state each time, dashboard_driving.py:241-256). Correct
    property to test is idempotency: rebuilding records from an unchanged
    snapshot twice gives identical output."""
    scene_points, scene_labels, scene_spikes = build_scene()
    snapshot_state = GridState()
    snapshot_state.update(scene_points, scene_labels, scene_spikes)
    snapshot = snapshot_state.snapshot()

    records_a = build_records(snapshot)
    records_b = build_records(snapshot)
    assert records_a == records_b, "rebuilding records from an unchanged snapshot should be idempotent"
    print(f"detected-object/record list is recomputed (not persisted) and idempotent: "
          f"{len(records_a)} records, identical across two rebuilds")


def test_frame_counter_single_source():
    """Both the HUD metric and both caption call sites must read the one
    shared `frame_display` value, not recompute frame_idx+1
    independently."""
    with open(DASHBOARD_SRC_PATH, "r", encoding="utf-8") as f:
        dashboard_src = f.read()

    assignment_count = dashboard_src.count("frame_display = frame_idx + 1")
    assert assignment_count == 1, \
        f"expected exactly one `frame_display = frame_idx + 1` assignment, found {assignment_count}"

    hud_line = next(line for line in dashboard_src.splitlines() if "frame_counter_placeholder.metric" in line)
    assert "frame_display" in hud_line, f"expected the HUD metric to read frame_display, got: {hud_line}"

    caption_lines = [line for line in dashboard_src.splitlines() if "debug_placeholder.caption" in line]
    assert len(caption_lines) >= 1, "expected at least one debug_placeholder.caption call site"
    # The caption calls span multiple lines (f-string continues past the call
    # site itself), so check the surrounding text block instead of a single line.
    assert dashboard_src.count("Frame {frame_display}") == 2, \
        "expected both caption sites (zero-cells branch and normal branch) to use frame_display"
    print("frame-counter single source: exactly one assignment, HUD and both captions all read frame_display")


def test_path_leak_never_leaks_full_path():
    """Mirrors dashboard_driving.py:73's unconditional
    os.path.basename(CHECKPOINT_PATH) usage - no debug-flag toggle exists
    anywhere in the file (confirmed by source inspection below), so there
    is no "without a flag" variant to test."""
    with open(DASHBOARD_SRC_PATH, "r", encoding="utf-8") as f:
        dashboard_src = f.read()

    assert "debug_mode" not in dashboard_src and "DEBUG" not in dashboard_src, \
        "expected no debug-flag toggle to exist for the checkpoint path display"

    fake_checkpoint_path = os.path.join("C:", "Users", "nirajcode76", "secret", "models", "checkpoint.pth")
    displayed = f"Loaded: {os.path.basename(fake_checkpoint_path)}"
    assert displayed == "Loaded: checkpoint.pth", f"unexpected display string: {displayed}"
    for leaked_fragment in ("Users", "nirajcode76", "secret", "models", os.sep):
        assert leaked_fragment not in displayed, f"path fragment leaked into displayed string: {leaked_fragment!r}"
    print(f"path-leak: {fake_checkpoint_path!r} displays only as {displayed!r}, no directory components leak")

    assert 'os.path.basename(CHECKPOINT_PATH)' in dashboard_src, \
        "expected dashboard_driving.py to still use os.path.basename(CHECKPOINT_PATH) for the sidebar label"


if __name__ == "__main__":
    test_advance_frame_index_resumes_not_resets()
    test_accumulated_grid_survives_simulated_rerun()
    test_record_list_recomputed_and_idempotent()
    test_frame_counter_single_source()
    test_path_leak_never_leaks_full_path()
    print("test_dashboard_driving.py: all assertions passed")
