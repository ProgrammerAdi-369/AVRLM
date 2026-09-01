import torch
import numpy as np
from synthetic_data import build_scene 

def get_mock_batch(batch_size=4, num_points=8192):
    """
    Bridges Member 3's synthetic NumPy physics into Member 1's PyTorch AI.
    Forces the output to strictly match the [Batch, 4, 8192] contract.
    """
    batch_inputs = []
    batch_targets = []
    
    for _ in range(batch_size):
        # 1. Generate one physical scene from Member 3's simulator
        pts_np, lbl_np, _ = build_scene(include_boundary_stress=True)
        
        # 2. Force the data to exactly 8192 points to prevent PyTorch crashes
        total_generated = pts_np.shape[0]
        if total_generated >= num_points:
            # Randomly sample down to 8192
            indices = np.random.choice(total_generated, num_points, replace=False)
        else:
            # Pad with zeros if the scene generated too few points
            indices = np.pad(np.arange(total_generated), (0, num_points - total_generated), mode='wrap')
            
        pts_sampled = pts_np[indices]
        lbl_sampled = lbl_np[indices]
        
        # 3. Transpose points to match PyTorch's [Channels, Points] format
        pts_transposed = pts_sampled.T 
        
        batch_inputs.append(torch.tensor(pts_transposed, dtype=torch.float32))
        batch_targets.append(torch.tensor(lbl_sampled, dtype=torch.int64))
        
    # Stack into final Batch tensors
    inputs = torch.stack(batch_inputs)
    targets = torch.stack(batch_targets)
    
    return inputs, targets