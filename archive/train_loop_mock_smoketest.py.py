import torch
import snntorch.functional as SF
import time

# Import your cleanly separated modules
from spiking_model import SpikingPointNet
from mock_loader import get_mock_batch
from profiler import EdgeProfiler

# --- 3. THE TRAINING LOOP ---
if __name__ == "__main__":
    # Automatically use your ASUS TUF's GPU if installed correctly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    net = SpikingPointNet(num_steps=10).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    
    # SNN-specific Loss Function
    loss_fn = SF.ce_count_loss() 

    # Initialize your new profiler module
    profiler = EdgeProfiler()
    
    epochs = 5
    
    print("Starting Training Loop...")
    for epoch in range(epochs):
        net.train()
        
        # 1. Get Data
        inputs, targets = get_mock_batch()
        inputs, targets = inputs.to(device), targets.to(device)

        # Start the hardware timer
        start_time = time.perf_counter()

        # 2. Forward Pass
        spk_rec = net(inputs) 

        # Stop the hardware timer
        process_time = time.perf_counter() - start_time

        # Generate and print the metrics
        metrics = profiler.evaluate_efficiency(spk_rec, process_time, inputs.size(0)) # <-- Add this
        profiler.print_report(metrics, epoch + 1)
        
        # 3. Calculate Loss
        loss = loss_fn(spk_rec, targets)
        
        # 4. Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")