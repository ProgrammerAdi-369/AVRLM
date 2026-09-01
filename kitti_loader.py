import glob
import os
import re

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class OptimizedKITTIPipeline(Dataset):
    """High-throughput KITTI-like point cloud loader for SNN training.

    The pipeline performs three critical stages per frame:
    1) Fast voxel downsampling via integer voxel hashing.
    2) Range-weighted farthest point sampling (FPS) to preserve distant targets.
    3) Temporal residual encoding by comparing the current frame against the previous
       frame in the same deterministic loading pass.

    The returned tensors are:
    - norm_tensor: [4, target_pts], normalized spatial channels in [0, 1] and an
      intensity channel in [0, 1]; suitable for 1D-conv SNN input.
    - raw_coords: [target_pts, 4], original meter-space positions plus modulated
      intensity values for mapping modules.
    """

    def __init__(
        self,
        bin_filepaths: list,
        max_range: float = 80.0,
        target_pts: int = 8192,
        voxel_size: float = 0.3,
    ):
        if not isinstance(bin_filepaths, (list, tuple)) or len(bin_filepaths) == 0:
            raise ValueError("bin_filepaths must be a non-empty list/tuple of .bin files.")

        self.bin_filepaths = sorted(bin_filepaths)
        self.max_range = float(max_range)
        self.target_pts = int(target_pts)
        self.voxel_size = float(voxel_size)

        if self.max_range <= 0.0:
            raise ValueError("max_range must be positive.")
        if self.target_pts <= 0:
            raise ValueError("target_pts must be positive.")
        if self.voxel_size <= 0.0:
            raise ValueError("voxel_size must be positive.")

        # Preflight validation to fail early with deterministic error messages.
        for fp in self.bin_filepaths:
            if not os.path.exists(fp):
                raise FileNotFoundError(f"Point cloud file does not exist: {fp}")

    def __len__(self):
        return len(self.bin_filepaths)

    @staticmethod
    def _to_float32_points(points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] < 4:
            raise ValueError(f"Expected shape [N, 4], got {points.shape}.")
        return points

    def _read_bin_file(self, filepath: str) -> np.ndarray:
        """Load KITTI-like binary float32 point clouds.

        The binary file is assumed to be raw float32 values laid out as
        [x, y, z, intensity], which is a common KITTI Velodyne representation.
        """
        data = np.fromfile(filepath, dtype=np.float32)
        if data.size == 0:
            return np.zeros((0, 4), dtype=np.float32)

        # Reshape to [N, 4] while tolerating any extra values.
        if data.size % 4 != 0:
            raise ValueError(f"Binary file has a non-4-column payload: {filepath}")
        points = data.reshape(-1, 4).astype(np.float32, copy=False)
        return points

    def _voxel_hash(self, coords: np.ndarray) -> np.ndarray:
        """Hash 3D coordinates into a 1D voxel key using integer arithmetic.

        This is O(N) and reduces the cost of per-frame occupancy comparisons. Real
        LiDAR coordinates are often signed (e.g., Y in [-40, 40], Z in [-3, 3]), so
        the voxel space must be shifted to a non-negative domain before hashing.
        """
        if coords.shape[0] == 0:
            return np.empty((0,), dtype=np.int64)

        # Shift the signed point cloud to a non-negative coordinate system before
        # voxelization: [-R, +R] -> [0, 2R]. This preserves the full 3D scene without
        # flattening negative Y/Z points to zero.
        shifted_coords = coords + self.max_range
        voxel_xyz = np.floor(
            np.clip(shifted_coords, 0.0, 2.0 * self.max_range) / self.voxel_size
        ).astype(np.int64)
        grid = int(np.ceil((2.0 * self.max_range) / self.voxel_size)) + 1
        voxel_key = (
            voxel_xyz[:, 0]
            + voxel_xyz[:, 1] * grid
            + voxel_xyz[:, 2] * grid * grid
        )
        return voxel_key.astype(np.int64, copy=False)

    def _voxel_downsample(self, points: np.ndarray) -> np.ndarray:
        """Stage 1: O(N) voxel-grid filter using unique voxel hashing.

        Each occupied voxel contributes exactly one representative point. This keeps
        the later FPS stage bounded while removing redundant near-field points.
        """
        points = self._to_float32_points(points)
        if points.shape[0] == 0:
            return points.copy()

        coords = points[:, :3]
        voxel_keys = self._voxel_hash(coords)
        unique_keys, first_indices = np.unique(voxel_keys, return_index=True)
        # Keep one representative point per voxel. first_indices is deterministic as
        # np.unique returns sorted keys, which makes the loader deterministic.
        return points[first_indices].copy()

    def _range_weighted_fps(self, points: np.ndarray) -> np.ndarray:
        """Stage 2: range-weighted farthest point sampling.

        We sample a subset of points using a distance score boosted by radial range:
            w_i = 1.0 + 0.05 * range_i

        The implementation uses the iterative minimum-distance FPS formulation, which
        keeps memory bounded and avoids the quadratic pairwise-distance blow-up of a
        naive implementation. This retains the farthest-point behavior while pushing
        the sampler toward distant, sparse targets and away from dense ground clutter.
        """
        points = self._to_float32_points(points)
        n_points = points.shape[0]

        if n_points == 0:
            return np.zeros((self.target_pts, 4), dtype=np.float32)

        if n_points <= self.target_pts:
            idx = np.random.choice(n_points, size=self.target_pts, replace=True)
            return points[idx].copy()

        coords = points[:, :3]
        ranges = np.linalg.norm(coords, axis=1).astype(np.float32)
        weights = 1.0 + 0.05 * ranges

        selected = np.empty(self.target_pts, dtype=np.int64)
        selected[0] = int(np.argmax(weights))

        chosen = np.zeros(n_points, dtype=bool)
        chosen[selected[0]] = True

        # min_dist_sq stores the minimum squared distance to any point already chosen.
        # This is updated in O(N) per sampled point rather than O(N * k) with a full
        # N x k pairwise distance matrix.
        min_dist_sq = np.sum((coords - coords[selected[0]]) ** 2, axis=1).astype(np.float32)

        for i in range(1, self.target_pts):
            score = min_dist_sq * weights
            score[chosen] = -np.inf
            candidate = int(np.argmax(score))
            selected[i] = candidate
            chosen[candidate] = True

            new_dist_sq = np.sum((coords - coords[candidate]) ** 2, axis=1).astype(np.float32)
            min_dist_sq = np.minimum(min_dist_sq, new_dist_sq)

        return points[selected].copy()

    def _temporal_delta_intensity(self, curr_points: np.ndarray, prev_points: np.ndarray) -> np.ndarray:
        """Temporal delta encoding using voxel occupancy residual.

        A point in the current frame is considered dynamic if its voxel key is not
        present in the previous frame. This is an O(N) set-difference operation on
        voxel hashes and is consistent with spiking neural network event coding.
        """
        curr_points = self._to_float32_points(curr_points)
        prev_points = self._to_float32_points(prev_points)

        if curr_points.shape[0] == 0:
            return np.zeros((0, 4), dtype=np.float32)

        curr_voxels = self._voxel_hash(curr_points[:, :3])
        prev_voxels = self._voxel_hash(prev_points[:, :3])

        # Residual occupancy is the symmetric voxel difference. Because only the
        # current frame is used for the output, we encode voxels newly occupied in
        # the current frame as dynamic events. This is stable and deterministic.
        if prev_voxels.size == 0:
            dynamic_mask = np.ones(curr_points.shape[0], dtype=bool)
        else:
            # np.setdiff1d is sorted and deterministic.
            changed_voxels = np.setdiff1d(curr_voxels, prev_voxels, assume_unique=False)
            dynamic_mask = np.isin(curr_voxels, changed_voxels)

        out = curr_points.copy()
        intensity = np.full(out.shape[0], 0.2, dtype=np.float32)
        intensity[dynamic_mask] = 1.0
        out[:, 3] = intensity
        return out

    def _process_frame(self, frame_idx: int) -> np.ndarray:
        """Process one frame using all three stages with deterministic temporal state."""
        curr_path = self.bin_filepaths[frame_idx]
        prev_idx = max(0, frame_idx - 1)
        prev_path = self.bin_filepaths[prev_idx]

        curr_raw = self._read_bin_file(curr_path)
        prev_raw = self._read_bin_file(prev_path)

        curr_down = self._voxel_downsample(curr_raw)
        prev_down = self._voxel_downsample(prev_raw)

        curr_sampled = self._range_weighted_fps(curr_down)
        prev_sampled = self._range_weighted_fps(prev_down)

        # Temporal modulation uses the prior frame to determine whether each point is
        # static or dynamic. Dynamic voxels produce a stronger intensity burst to
        # trigger SNN spikes.
        curr_delta = self._temporal_delta_intensity(curr_sampled, prev_sampled)

        # Normalize spatial coordinates symmetrically across the full signed domain
        # [-max_range, +max_range] -> [0.0, 1.0]. This preserves negative Y/Z values
        # that are common in real LiDAR sweeps instead of collapsing them to zero.
        coords = curr_delta[:, :3]
        norm_coords = np.clip((coords + self.max_range) / (2.0 * self.max_range), 0.0, 1.0)

        norm_tensor = np.zeros((4, self.target_pts), dtype=np.float32)
        norm_tensor[:3, :] = norm_coords.T
        norm_tensor[3, :] = curr_delta[:, 3]

        raw_coords = curr_delta.copy()
        return {"norm_tensor": norm_tensor, "raw_coords": raw_coords}

    def __getitem__(self, idx: int):
        """Return one frame's normalized SNN input and raw mapping coordinates.

        The method always reads both current and previous frames deterministically,
        which keeps the data path clean for multi-worker PyTorch loading.
        """
        if idx < 0 or idx >= len(self.bin_filepaths):
            raise IndexError(f"Index {idx} out of range for dataset length {len(self.bin_filepaths)}.")

        return self._process_frame(idx)


def kitti_collate_batch(batch):
    """Assemble a batch of per-frame outputs into clean torch tensors.

    Each dataset item contains:
    - norm_tensor: [4, 8192]
    - raw_coords: [8192, 4]

    The batched outputs become:
    - norm_tensor: [B, 4, 8192]
    - raw_coords: [B, 8192, 4]
    """
    if not isinstance(batch, (list, tuple)) or len(batch) == 0:
        raise ValueError("Batch must be a non-empty list of dataset samples.")

    norm_batch = torch.as_tensor(
        np.stack([sample["norm_tensor"] for sample in batch], axis=0),
        dtype=torch.float32,
    )
    raw_batch = torch.as_tensor(
        np.stack([sample["raw_coords"] for sample in batch], axis=0),
        dtype=torch.float32,
    )
    return {"norm_tensor": norm_batch, "raw_coords": raw_batch}


def get_streaming_dataloader(bin_filepaths: list, batch_size: int = 4):
    """Create a streaming DataLoader for the optimized KITTI pipeline.

    Settings:
    - num_workers=2 for parallel disk IO.
    - pin_memory=True to accelerate CPU->GPU copies.
    - prefetch_factor=2 for higher-throughput batching.
    - collate_fn=kitti_collate_batch for deterministic torch-native batches.
    """
    dataset = OptimizedKITTIPipeline(bin_filepaths=bin_filepaths)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        collate_fn=kitti_collate_batch,
    )
    return loader


def discover_and_sort_kitti_bins(root_dir: str):
    """Recursively discover KITTI-like .bin files and sort them by frame index.

    This is critical because the dataset compares each frame with the previous one:
        prev_idx = max(0, idx - 1)

    If the underlying file list is not sorted by temporal index, the residual
    encoding becomes invalid. The function extracts the numeric frame identifier from
    each filename and orders the files in ascending sequence.
    """
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"KITTI root directory does not exist: {root_dir}")

    matches = glob.glob(os.path.join(root_dir, "**", "*.bin"), recursive=True)
    matches = [p for p in matches if os.path.isfile(p)]

    def sort_key(path):
        name = os.path.basename(path)
        stem = os.path.splitext(name)[0]
        digits = [int(val) for val in re.findall(r"\d+", stem)]
        if digits:
            return (0, digits[0])
        return (1, name)

    return sorted(matches, key=sort_key)


def create_mock_bin_files(dir_path: str, count: int = 5):
    """Generate fake KITTI-style .bin files for local validation.

    Each file contains 50,000 random 3D points plus a synthetic intensity channel.
    This is a lightweight stand-in for the real sensor output and is sufficient for
    testing ingestion logic and output tensor shapes.
    """
    os.makedirs(dir_path, exist_ok=True)

    for i in range(count):
        rng = np.random.default_rng(42 + i)
        x = rng.uniform(0.0, 80.0, size=50000).astype(np.float32)
        y = rng.uniform(-40.0, 40.0, size=50000).astype(np.float32)
        z = rng.uniform(-3.0, 3.0, size=50000).astype(np.float32)
        intensity = rng.uniform(0.0, 1.0, size=50000).astype(np.float32)
        points = np.stack([x, y, z, intensity], axis=1).astype(np.float32)

        out_path = os.path.join(dir_path, f"frame_{i:05d}.bin")
        points.astype(np.float32).tofile(out_path)

    return sorted(glob.glob(os.path.join(dir_path, "*.bin")))


if __name__ == "__main__":
    mock_dir = os.path.join("mock_kitti", "velodyne")
    filepaths = create_mock_bin_files(mock_dir, count=5)

    dataset = OptimizedKITTIPipeline(bin_filepaths=filepaths)
    sample = dataset[0]

    assert sample["norm_tensor"].shape == (4, 8192), (
        f"Unexpected norm_tensor shape: {sample['norm_tensor'].shape}"
    )
    assert sample["raw_coords"].shape == (8192, 4), (
        f"Unexpected raw_coords shape: {sample['raw_coords'].shape}"
    )

    # DataLoader smoke test with batch aggregation.
    dataloader = get_streaming_dataloader(filepaths, batch_size=2)
    batch = next(iter(dataloader))

    print(f"norm_tensor batch shape: {tuple(batch['norm_tensor'].shape)}")
    print(f"raw_coords batch shape: {tuple(batch['raw_coords'].shape)}")
    print(f"single-frame norm_tensor shape: {tuple(sample['norm_tensor'].shape)}")
    print(f"single-frame raw_coords shape: {tuple(sample['raw_coords'].shape)}")
    print("KITTI pipeline validation passed.")
