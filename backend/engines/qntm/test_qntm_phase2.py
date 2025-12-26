"""
Test Script for QNTM Phase 2
============================
Verifies the QNTM Engine's WaveFunction and Annealing logic.
We expect to see the "Collapse Metric" decrease (variance -> 0)
and the average energy decrease as the system cools.
"""

import sys
import os

# Add backend to path
current_dir = os.path.dirname(os.path.abspath(__file__)) # backend/engines/qntm
engines_dir = os.path.dirname(os.path.dirname(current_dir)) # backend
sys.path.append(engines_dir)

from engines.qntm.state import WaveFunction
from engines.qntm.annealer import QuantumAnnealer

def run_test():
    print("Initializing Quantum Multiverse (WaveFunction)...")
    # Create a population of 50 random universes
    wf = WaveFunction(population_size=50, num_steps=20)
    
    print(f"Created population of {len(wf.states)} states.")
    initial_metric = wf.collapse_metric()
    print(f"Initial Collapse Metric (Variance): {initial_metric:.4f} (Should be high ~ high entropy)")
    
    # Initialize Annealer
    annealer = QuantumAnnealer(wf, cooling_rate=0.9)
    print("Starting Quantum Annealing (Tunneling Enabled)...")
    
    # Run
    history = []
    print(f"{'Step':<5} | {'Temp':<8} | {'Avg Energy':<10} | {'Collapse (Var)':<15} | {'Ghosts':<6}")
    print("-" * 60)
    
    for stats in annealer.run(steps=50, yield_every=5):
        print(f"{stats['step']:<5} | {stats['temp']:.4f}   | {stats['avg_energy']:.4f}     | {stats['collapse_metric']:.4f}          | {stats['ghost_count']:<6}")
        history.append(stats)
        
    final_metric = wf.collapse_metric()
    print(f"\nFinal Collapse Metric: {final_metric:.4f}")
    
    # Check convergence
    if final_metric < initial_metric * 0.1:
        print("SUCCESS: WaveFunction collapsed significanly (Variance reduced by >90%).")
    else:
        print("WARNING: WaveFunction did not collapse as expected.")
        
    print(f"Total Ghost Trajectories Recorded: {len(annealer.ghost_trajectories)}")
    
    # Show the "Mean Trajectory" (Expectation Value)
    mean_traj = wf.get_mean_trajectory()
    # Just print start/end for brevity
    print(f"Mean Trajectory Start: {mean_traj[0]:.2f}")
    print(f"Mean Trajectory End:   {mean_traj[-1]:.2f}")

if __name__ == "__main__":
    run_test()
