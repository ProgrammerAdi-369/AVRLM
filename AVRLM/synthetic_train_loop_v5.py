"""
v5: fixes v4's overcorrection, diagnosed live on the dashboard -- 96.2% of
a real (unbalanced) scene was classified as "Static Obstacle" instead of
"Drivable". v4's class_weights=[1.0, 6.0, 20.0] combined with heavy
oversampling (minority_boost=11.0) pushed the model to over-predict
minority classes once deployed on genuinely imbalanced data (like
kitti_loader.py's raw mock point clouds), even though it looked fine on
v4's own artificially-balanced validation set.

Key fix: validate on BOTH a class-balanced set (to track rare-class
learning) AND a naturally-imbalanced set (to catch exactly this kind of
deployment-time collapse before it reaches the dashboard). Also dial back
weights/oversampling from v4's aggressive settings toward v3's more
moderate ones, since v3 (93.4% overall / 62.4% min-class) never showed
this collapse.
"""

import numpy as np
import torch
import torch.nn as nn
import time

from spiking_model import SpikingPointNet
from profiler import EdgeProfiler
from synthetic_lidar_data import (
    generate_ground_plane,
    generate_pole,
    generate_dynamic_cluster,
    generate_boundary_stress_ring,
)

CHECKPOINT_PATH = "snn_weights.pth"
NUM_POINTS = 8192
MAX_RANGE = 100.0
NUM_CLASSES = 3


def build_richer_scene(rng: np.random.Generator):
    parts = [generate_ground_plane(rng=rng)]

    n_poles = rng.integers(2, 6)
    for _ in range(n_poles):
        cx = rng.uniform(-60.0, 60.0)
        cy = rng.uniform(-60.0, 60.0)
        height = rng.uniform(1.2, 3.0)
        parts.append(generate_pole((cx, cy), height=height, n_points=180, rng=rng))

    n_dynamic = rng.integers(1, 4)
    for _ in range(n_dynamic):
        cx = rng.uniform(-40.0, 40.0)
        cy = rng.uniform(-40.0, 40.0)
        spread = rng.uniform(0.2, 0.6)
        parts.append(generate_dynamic_cluster((cx, cy), spread=spread, n_points=150, rng=rng))

    if rng.random() < 0.3:
        parts.append(generate_boundary_stress_ring(rng=rng))

    points = np.concatenate([p[0] for p in parts], axis=0)
    labels = np.concatenate([p[1] for p in parts], axis=0)
    spikes = np.concatenate([p[2] for p in parts], axis=0)
    return points, labels, spikes


def class_balanced_sample(points, labels, num_points, rng, minority_boost=7.0):
    """Dialed back from v4's 11.0 -- that was too aggressive and taught the
    model to expect minority classes far more often than they actually
    appear in real deployment data."""
    n = points.shape[0]
    class_ids, counts = np.unique(labels, return_counts=True)
    freq = {c: cnt / n for c, cnt in zip(class_ids, counts)}

    weights = np.ones(n, dtype=np.float64)
    for c in class_ids:
        mask = labels == c
        base_weight = 1.0 / max(freq[c], 1e-6)
        if c != 0:
            base_weight *= minority_boost
        weights[mask] = base_weight
    weights /= weights.sum()

    if n >= num_points:
        idx = rng.choice(n, size=num_points, replace=False, p=weights)
    else:
        idx = rng.choice(n, size=num_points, replace=True, p=weights)

    return points[idx], labels[idx]


def uniform_sample(points, labels, num_points, rng):
    """NO class balancing -- mirrors kitti_loader.py's raw mock data
    distribution. Used only for the 'realistic' validation set so we can
    catch deployment-time collapse before it reaches the dashboard."""
    n = points.shape[0]
    if n >= num_points:
        idx = rng.choice(n, size=num_points, replace=False)
    else:
        idx = rng.choice(n, size=num_points, replace=True)
    return points[idx], labels[idx]


def scene_to_tensor(pts_sampled, lbl_sampled, max_range=MAX_RANGE):
    coords = pts_sampled[:, :3]
    norm_coords = np.clip((coords + max_range) / (2.0 * max_range), 0.0, 1.0)

    norm_tensor = np.zeros((4, pts_sampled.shape[0]), dtype=np.float32)
    norm_tensor[:3, :] = norm_coords.T
    norm_tensor[3, :] = pts_sampled[:, 3]

    return norm_tensor


def get_batch(batch_size, rng, num_points=NUM_POINTS, balanced=True, minority_boost=7.0):
    inputs, targets = [], []
    for _ in range(batch_size):
        pts, lbl, _ = build_richer_scene(rng)
        if balanced:
            pts_s, lbl_s = class_balanced_sample(pts, lbl, num_points, rng, minority_boost)
        else:
            pts_s, lbl_s = uniform_sample(pts, lbl, num_points, rng)
        norm_tensor = scene_to_tensor(pts_s, lbl_s)
        inputs.append(torch.tensor(norm_tensor, dtype=torch.float32))
        targets.append(torch.tensor(lbl_s, dtype=torch.int64))
    return torch.stack(inputs), torch.stack(targets)


def weighted_spike_ce_loss(spk_rec, targets, class_weights, num_steps):
    avg_spikes = spk_rec.sum(dim=0) / num_steps
    logits = avg_spikes.permute(0, 2, 1).reshape(-1, avg_spikes.shape[1])
    targets_flat = targets.reshape(-1)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    return loss_fn(logits, targets_flat)


def per_class_accuracy(spk_rec, targets, num_classes=NUM_CLASSES):
    total_spikes = spk_rec.sum(dim=0)
    preds = torch.argmax(total_spikes, dim=1)
    correct = (preds == targets)

    report = {}
    for c in range(num_classes):
        mask = targets == c
        total = mask.sum().item()
        acc = correct[mask].sum().item() / total if total > 0 else float("nan")
        report[c] = (acc, total)
    overall = correct.float().mean().item()
    return overall, report


def evaluate(net, rng, device, n_batches, balanced, batch_size=4):
    net.eval()
    val_acc_total = 0.0
    class_totals = {0: [0, 0], 1: [0, 0], 2: [0, 0]}

    with torch.no_grad():
        for _ in range(n_batches):
            inputs, targets = get_batch(batch_size, rng, balanced=balanced)
            inputs, targets = inputs.to(device), targets.to(device)
            spk_rec = net(inputs)
            overall_acc, report = per_class_accuracy(spk_rec, targets)
            val_acc_total += overall_acc
            for c, (acc, total) in report.items():
                if total > 0:
                    class_totals[c][0] += acc * total
                    class_totals[c][1] += total

    val_acc = val_acc_total / n_batches
    class_accs = {
        c: (class_totals[c][0] / class_totals[c][1] if class_totals[c][1] > 0 else float("nan"))
        for c in range(3)
    }
    return val_acc, class_accs


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    net = SpikingPointNet(num_steps=10).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=5e-4)

    epochs = 100
    warmup_epochs = 5

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    profiler = EdgeProfiler()

    # Dialed back from v4's [1.0, 6.0, 20.0] -- closer to v3's [1.0, 8.0, 15.0],
    # which never showed the "everything is Static Obstacle" collapse.
    class_weights = torch.tensor([1.0, 7.0, 16.0], dtype=torch.float32).to(device)
    minority_boost = 7.0

    train_rng = np.random.default_rng(123)
    val_rng_balanced = np.random.default_rng(999)
    val_rng_realistic = np.random.default_rng(7777)  # disjoint seed, uniform sampling

    batches_per_epoch = 25
    batch_size = 4
    val_batches = 10

    best_realistic_min_class_acc = 0.0

    print(f"Starting training: {epochs} epochs x {batches_per_epoch} batches x {batch_size} scenes/batch")
    print("Validating on BOTH a class-balanced set AND a realistic (naturally")
    print("imbalanced) set every epoch -- checkpoint saved on the REALISTIC")
    print("set's minimum class accuracy, since that's what the dashboard sees.\n")

    for epoch in range(epochs):
        net.train()
        epoch_loss = 0.0

        for batch_idx in range(batches_per_epoch):
            inputs, targets = get_batch(batch_size, train_rng, balanced=True, minority_boost=minority_boost)
            inputs, targets = inputs.to(device), targets.to(device)

            spk_rec = net(inputs)
            loss = weighted_spike_ce_loss(spk_rec, targets, class_weights, net.num_steps)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / batches_per_epoch

        bal_acc, bal_class_accs = evaluate(net, val_rng_balanced, device, val_batches, balanced=True)
        real_acc, real_class_accs = evaluate(net, val_rng_realistic, device, val_batches, balanced=False)

        real_min_class_acc = min(v for v in real_class_accs.values() if not np.isnan(v))

        bal_str = " | ".join(f"c{c}:{a*100:.0f}%" for c, a in bal_class_accs.items())
        real_str = " | ".join(f"c{c}:{a*100:.0f}%" for c, a in real_class_accs.items())

        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | "
              f"BalancedVal: {bal_acc*100:.1f}% [{bal_str}] | "
              f"RealisticVal: {real_acc*100:.1f}% [{real_str}]")

        if real_min_class_acc > best_realistic_min_class_acc:
            best_realistic_min_class_acc = real_min_class_acc
            torch.save(net.state_dict(), CHECKPOINT_PATH)
            print(f"  New best REALISTIC min-class-acc ({real_min_class_acc*100:.1f}%) -- checkpoint saved")

    print(f"\nTraining complete.")
    print(f"Best realistic (naturally-imbalanced) minimum per-class accuracy: {best_realistic_min_class_acc*100:.1f}%")
    print(f"Final checkpoint: {CHECKPOINT_PATH}")

    inputs, targets = get_batch(4, val_rng_realistic, balanced=False)
    inputs = inputs.to(device)
    start_time = time.perf_counter()
    with torch.no_grad():
        spk_rec = net(inputs)
    process_time = time.perf_counter() - start_time
    metrics = profiler.evaluate_efficiency(spk_rec, process_time, inputs.size(0))
    profiler.print_report(metrics, "FINAL") 