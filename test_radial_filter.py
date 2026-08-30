import numpy as np

from radial_filter import radial_filter
from synthetic_lidar_data import build_scene, generate_boundary_stress_ring


def test_radial_split_within_bounds():
    points, labels, spikes = build_scene()
    inner_mask, outer_mask = radial_filter(points)
    r = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)

    assert inner_mask.sum() + outer_mask.sum() <= points.shape[0]
    assert not np.any((r > 100.0) & inner_mask)
    assert not np.any((r > 100.0) & outer_mask)

    print(f"build_scene(): N={points.shape[0]}, inner={inner_mask.sum()}, "
          f"outer={outer_mask.sum()}, dropped(r>100)={np.sum(r > 100.0)}")


def test_boundary_ring_split_complete():
    ring_points, ring_labels, ring_spikes = generate_boundary_stress_ring()
    ring_inner, ring_outer = radial_filter(ring_points)
    ring_r = np.sqrt(ring_points[:, 0] ** 2 + ring_points[:, 1] ** 2)

    print(f"boundary_stress_ring(): N={ring_points.shape[0]}, "
          f"inner(r<=10)={ring_inner.sum()}, outer(10<r<=100)={ring_outer.sum()}, "
          f"r range=[{ring_r.min():.4f}, {ring_r.max():.4f}]")

    assert ring_inner.sum() + ring_outer.sum() == ring_points.shape[0]


if __name__ == "__main__":
    test_radial_split_within_bounds()
    test_boundary_ring_split_complete()
    print("test_radial_filter.py: all assertions passed")
