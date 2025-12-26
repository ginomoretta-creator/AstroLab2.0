"""
Test Script for THRML Phase 1
=============================
Verifies the Heterogeneous Graph, Energy functions, and Gibbs Sampler.
"""

import sys
import os

# Add backend to path so we can import engines
current_dir = os.path.dirname(os.path.abspath(__file__)) # backend/engines/thrml
engines_dir = os.path.dirname(os.path.dirname(current_dir)) # backend
sys.path.append(engines_dir)

from engines.thrml.graph import HeterogeneousGraph
from engines.thrml.sampler import GibbsSampler
from engines.thrml.energy import compute_total_energy

def run_test():
    print("Initializing Heterogeneous Graph...")
    graph = HeterogeneousGraph()
    
    # 1. Create a chain of 20 time steps
    n_steps = 20
    nodes = []
    for i in range(n_steps):
        # Add slight bias to some nodes to encourage activation
        bias = 0.5 if 5 <= i <= 15 else -0.1
        node = graph.add_timestep_node(time_index=i, bias=bias)
        nodes.append(node)
        
    print(f"Created {len(nodes)} TimeStepNodes.")
    
    # 2. Add edges (Smoothness)
    # Connect i to i+1 with positive weight (ferromagnetic)
    for i in range(n_steps - 1):
        graph.add_edge(nodes[i], nodes[i+1], weight=1.0)
        
    print("Added smoothness edges.")
    
    # 3. Add Fuel Constraint
    # Limit to 5 units of thrust (approx 5 nodes ON)
    # High penalty
    fuel_node = graph.add_fuel_node(max_fuel=5.0, penalty_strength=2.0)
    print("Added FuelConstraintNode (Max 5.0).")
    
    # 4. Initialize state randomly
    import random
    for node in nodes:
        node.value = random.choice([0.0, 1.0])
        
    initial_energy = compute_total_energy(graph)
    initial_active = sum(n.value for n in nodes)
    print(f"Initial State: Active={initial_active}, Energy={initial_energy:.2f}")
    
    # 5. Run Sampler
    sampler = GibbsSampler(graph, beta=2.0) # High beta = low temp = greedy optimization
    
    print("\nStarting Sampling Loop...")
    history = []
    
    # Run for 100 sweeps
    for stats in sampler.run(n_steps=100, yield_every=10):
        print(f"Step {stats['step']}: Active={stats['active_nodes']}, Energy={stats['energy']:.2f}")
        history.append(stats)
        
    final_active = sum(n.value for n in nodes)
    print(f"\nFinal State: Active={final_active}")
    
    # Verify constraints
    # We asked for max 5. With beta=2.0 and penalty=2.0, it should be close to 5 or less.
    if final_active <= 7: # Allow some fuctuation
        print("SUCCESS: Fuel constraint respected (approx).")
    else:
        print(f"WARNING: Fuel constraint violated (Active={final_active}, Limit=5).")
        
    # Verify smoothness
    switches = 0
    for i in range(n_steps - 1):
        if nodes[i].value != nodes[i+1].value:
            switches += 1
            
    print(f"Smoothness Check: {switches} switches (Lower is better)")

if __name__ == "__main__":
    run_test()
