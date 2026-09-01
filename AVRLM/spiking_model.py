import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from snntorch import utils


# --- 1. THE ARCHITECTURE ---
class SpikingPointNet(nn.Module):
    def __init__(self, beta=0.9, num_steps=10):
        super().__init__()
        self.num_steps = num_steps
        
        # Surrogate gradient for backpropagation
        spike_grad = surrogate.fast_sigmoid() 
        
        # 1D Convolutions (These are the layers with trainable parameters!)
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=64, kernel_size=1) 
        self.lif1 = snn.Leaky(beta=beta, threshold=0.5, spike_grad=spike_grad, init_hidden=True) 
        
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1)
        self.lif2 = snn.Leaky(beta=beta, threshold=0.5, spike_grad=spike_grad, init_hidden=True)
        
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=3, kernel_size=1) 
        self.lif3 = snn.Leaky(beta=beta, threshold=0.5, spike_grad=spike_grad, init_hidden=True, output=True) 

    def forward(self, x):
        utils.reset(self) # Clear old memory before processing a new batch
        spk_rec = []
        
        # Temporal Loop: Accumulate voltage over time
        for step in range(self.num_steps):
            c1 = self.conv1(x)
            s1 = self.lif1(c1)
            
            c2 = self.conv2(s1)
            s2 = self.lif2(c2)
            
            c3 = self.conv3(s2)
            spk_out, mem_out = self.lif3(c3)
            
            spk_rec.append(spk_out)
            
        return torch.stack(spk_rec)