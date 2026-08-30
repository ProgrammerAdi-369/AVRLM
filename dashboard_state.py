"""
Pure playback state-transition helpers for dashboard_driving.py, kept in
their own module (no streamlit/torch imports) so they're importable and
unit-testable without loading the full dashboard script or a running
Streamlit context. See Reports/AUDIT-v2.md test-rebuild Phase 5.
"""


def advance_frame_index(frame_idx: int, total_frames: int):
    """Returns the next frame_idx after committing the current frame, or
    None if the sequence is already exhausted (frame_idx >= total_frames)."""
    if frame_idx >= total_frames:
        return None
    return frame_idx + 1
