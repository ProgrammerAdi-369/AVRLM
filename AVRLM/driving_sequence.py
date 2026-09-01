"""
A coherent, UGV-centric driving sequence for the dashboard demo -- built on
top of the team's synthetic_lidar_data.py primitives, but composed into a
scene that actually tells a story across frames:

- The UGV moves forward through a corridor of static obstacles (poles),
  each one entering the 100m range, then the 10m foveated zone, then
  falling behind -- demonstrating variable-resolution perception in action
  as objects cross the 10m boundary.
- A pedestrian (dynamic cluster) walks a fixed real-world path that the
  UGV's forward motion causes to sweep across its relative view, entering
  from one side and exiting the other -- classic "crossing the road"
  scenario.
- A second, faster dynamic object (car-like) overtakes the UGV from
  behind, demonstrating detection of a fast-closing threat.

All object world-positions are fixed; what changes frame to frame is the
UGV's own position, and every point cloud returned is already transformed
into the UGV's own ego-centric frame (points relative to the UGV), which
is the coordinate convention the rest of the pipeline (grid.py, etc.)
expects (UGV always at local origin).

NOTE: this module intentionally has NO if __name__ == "__main__" self-test
block. A previous version had one, and its print statements were observed
firing on every `streamlit run dashboard_driving.py` launch before any
button was clicked -- Streamlit's script-execution/reload model does not
reliably respect that guard the way plain `python file.py` does. Test this
module directly via a separate throwaway script if needed, never via a
guarded block inside the module itself.
"""

import numpy as np

from synthetic_lidar_data import (
    generate_ground_plane,
    generate_pole,
    generate_dynamic_cluster,
    LABEL_DRIVABLE,
)

STATIC_POLE_WORLD_POSITIONS = [
    (20.0, 3.0), (35.0, -4.0), (55.0, 2.5), (70.0, -3.5), (90.0, 4.0),
]

PEDESTRIAN_START_WORLD = (40.0, -15.0)
PEDESTRIAN_VELOCITY = (0.0, 1.2)

OVERTAKING_CAR_START_WORLD = (-15.0, -2.0)
OVERTAKING_CAR_VELOCITY = (3.5, 0.0)

UGV_SPEED = 1.5


def get_ugv_position(frame_idx: int):
    return (UGV_SPEED * frame_idx, 0.0)


def build_driving_frame(frame_idx: int, rng):
    ugv_pos = get_ugv_position(frame_idx)
    parts = []

    ground = generate_ground_plane(rng=rng)
    parts.append(ground)

    for pole_x, pole_y in STATIC_POLE_WORLD_POSITIONS:
        ego_x = pole_x - ugv_pos[0]
        ego_y = pole_y - ugv_pos[1]
        if abs(ego_x) <= 100.0 and abs(ego_y) <= 100.0:
            parts.append(generate_pole((ego_x, ego_y), height=2.0, rng=rng))

    ped_x = PEDESTRIAN_START_WORLD[0] + PEDESTRIAN_VELOCITY[0] * frame_idx
    ped_y = PEDESTRIAN_START_WORLD[1] + PEDESTRIAN_VELOCITY[1] * frame_idx
    ego_ped_x = ped_x - ugv_pos[0]
    ego_ped_y = ped_y - ugv_pos[1]
    if abs(ego_ped_x) <= 100.0 and abs(ego_ped_y) <= 100.0:
        parts.append(generate_dynamic_cluster((ego_ped_x, ego_ped_y), spread=0.35, n_points=90, rng=rng))

    car_x = OVERTAKING_CAR_START_WORLD[0] + OVERTAKING_CAR_VELOCITY[0] * frame_idx
    car_y = OVERTAKING_CAR_START_WORLD[1] + OVERTAKING_CAR_VELOCITY[1] * frame_idx
    ego_car_x = car_x - ugv_pos[0]
    ego_car_y = car_y - ugv_pos[1]
    if abs(ego_car_x) <= 100.0 and abs(ego_car_y) <= 100.0:
        parts.append(generate_dynamic_cluster((ego_car_x, ego_car_y), spread=0.8, n_points=140, height=1.5, rng=rng))

    points = np.concatenate([p[0] for p in parts], axis=0)
    labels = np.concatenate([p[1] for p in parts], axis=0)
    spikes = np.concatenate([p[2] for p in parts], axis=0)
    return points, labels, spikes, ugv_pos


def build_driving_sequence(n_frames=40, seed=2026):
    rng = np.random.default_rng(seed)
    return [build_driving_frame(i, rng=rng) for i in range(n_frames)]