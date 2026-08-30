import time
import torch

class EdgeProfiler:
    def __init__(self):
        # Industry standard energy estimates in picojoules (pJ) for 32-bit math
        self.MAC_ENERGY_PJ = 4.6  
        self.AC_ENERGY_PJ = 0.9   
        
    def evaluate_efficiency(self, spk_rec, process_time, batch_size):
        """
        Calculates FPS, computational sparsity, and energy savings.
        """
        # 1. Latency & Speed
        fps = batch_size / process_time
        
        # 2. Sparsity & Compute Math
        # Total possible calculations if this were a standard AI
        total_neuron_states = spk_rec.numel() 
        
        # Actual times a neuron fired (the only times math actually occurred)
        active_spikes = spk_rec.sum().item()  
        
        # Empty space that the SNN mathematically skipped
        skipped_math = total_neuron_states - active_spikes 
        sparsity_percentage = (skipped_math / total_neuron_states) * 100
        
        # 3. Energy Calculation
        standard_energy_cost = total_neuron_states * self.MAC_ENERGY_PJ
        snn_energy_cost = active_spikes * self.AC_ENERGY_PJ
        energy_saved = standard_energy_cost - snn_energy_cost
        
        return {
            "fps": fps,
            "latency_sec": process_time,
            "sparsity_pct": sparsity_percentage,
            "ac_ops": active_spikes,
            "mac_ops_avoided": skipped_math,
            "energy_saved_pj": energy_saved
        }

    def print_report(self, metrics, epoch):
        print(f"{'='*40}")
        print(f"🚀 EDGE PROFILER REPORT | EPOCH {epoch}")
        print(f"{'='*40}")
        print(f"Speed     : {metrics['fps']:.2f} FPS | {metrics['latency_sec']:.4f}s Latency")
        print(f"Sparsity  : {metrics['sparsity_pct']:.2f}% of standard math skipped!")
        print(f"Compute   : {int(metrics['ac_ops'])} AC ops vs {int(metrics['ac_ops'] + metrics['mac_ops_avoided'])} MAC ops")
        print(f"Energy    : {metrics['energy_saved_pj']:.2f} pJ saved per frame")
        print(f"{'='*40}")