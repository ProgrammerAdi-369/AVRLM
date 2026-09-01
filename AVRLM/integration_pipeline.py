import torch
import numpy as np
import time
import os
import sys

# Ensure Python can read Member 3's files inside the AVRLM folder
sys.path.append(os.path.join(os.path.dirname(__file__), 'AVRLM'))

# 1. MEMBER 2 (The Supplier)
from kitti_loader import get_streaming_dataloader, create_mock_bin_files

# 2. YOU (The Brain)
from spiking_model import SpikingPointNet

# 3. MEMBER 3 (The Consumer)
from handoff import generate_2_5d_grid, memory_metrics

def run_ugv_perception():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize your SNN in evaluation mode (no gradient tracking for pure speed)
    ai_model = SpikingPointNet(num_steps=10).to(device)
    ai_model.eval() 
    
    # Generate local LiDAR files using Member 2's mock function to test the pipeline
    print("Generating mock LiDAR hardware files...")
    mock_dir = os.path.join("mock_kitti", "velodyne")
    bin_files = create_mock_bin_files(mock_dir, count=5)
    dataloader = get_streaming_dataloader(bin_files, batch_size=1) 
    
    print("UGV Perception Engine Online...")
    
    with torch.no_grad():
        for frame_idx, batch in enumerate(dataloader):
            start_time = time.perf_counter()
            
            # --- A. INGESTION (Member 2) ---
            inputs = batch["norm_tensor"].to(device) 
            raw_metrics = batch["raw_coords"].squeeze(0).cpu().numpy() 
            
            # --- B. INFERENCE (You) ---
            spk_rec = ai_model(inputs) 
            
            total_spikes_tensor = spk_rec.sum(dim=0) 
            predictions_tensor = torch.argmax(total_spikes_tensor, dim=1) 
            activity_tensor = total_spikes_tensor.sum(dim=1) 
            
            labels_np = predictions_tensor.squeeze().cpu().numpy()
            spikes_np = activity_tensor.squeeze().cpu().numpy()
            
            # --- C. SPATIAL MAPPING (Member 3) ---
            active_map = generate_2_5d_grid(
                points=raw_metrics, 
                labels=labels_np, 
                spikes=spikes_np
            )
            
            # --- D. DASHBOARD METRICS ---
            mem_stats = memory_metrics()
            
            fps = 1.0 / (time.perf_counter() - start_time)
            print(f"Frame {frame_idx} | Speed: {fps:.1f} FPS | Active Cells: {mem_stats['active_cell_count']}")

if __name__ == "__main__":
    run_ugv_perception()