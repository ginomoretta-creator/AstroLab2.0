"""
QNTM Annealer
=============

Simulates the Quantum Annealing process.
Instead of a single particle rolling down a hill (Gradient Descent),
we simulate a population (WaveFunction) that "tunnels" through barriers.

Tunneling Logic:
- Representation: Population of states.
- Transition: Parallel updates with "Tunneling Probability" proportional to Energy differences.
- High Tunneling (High Temp) -> States jump across barriers easily (Superposition).
- Low Tunneling (Low Temp) -> States get trapped in local/global minima (Collapse).
"""

import math
import random
import copy
from typing import List, Dict, Any, Generator

from .state import WaveFunction, QuantumState

# Optimization: Import energy function from THRML if available, or define simple one
# For purity, QNTM should have its own energy evaluator that respects the same physics.
# Let's define a simple abstract energy function here or reuse thrml.energy logic concept.

def simple_energy_function(config: List[float], budget: float = 5.0, bias_field: List[float] = None) -> float:
    """
    Computes energy for a continuous configuration.
    E = Smoothness + Fuel Penalty - Bias Alignment
    """
    energy = 0.0
    
    # 1. Smoothness (Sum of squared differences)
    for i in range(len(config) - 1):
        diff = config[i+1] - config[i]
        energy += 5.0 * (diff * diff)
        
    # 2. Fuel Penalty (Global)
    # Use absolute values for braking/thrusting
    total_thrust = sum(abs(x) for x in config)
    excess = max(0.0, total_thrust - budget)
    energy += 2.0 * (excess * excess)
    
    # 3. Bias Alignment (Physics-Aware)
    # Energy is lowered when thrust aligns with positive bias (h)
    # E += - sum(h_i * s_i)
    if bias_field and len(bias_field) == len(config):
        alignment = sum(b * s for b, s in zip(bias_field, config))
        energy -= alignment * 2.0  # Weight the bias influence
    
    return energy

class QuantumAnnealer:
    def __init__(self, wavefunction: WaveFunction, cooling_rate: float = 0.95, bias_field: List[float] = None):
        self.wavefunction = wavefunction
        self.temp = 10.0 # Initial "Quantum Fluctuation" temperature
        self.cooling_rate = cooling_rate
        self.ghost_trajectories: List[List[float]] = [] # Track rejected/historical paths
        self.bias_field = bias_field
        
    def step(self):
        """
        Perform one annealing step:
        1. Perturb population (propose new locations).
        2. Accept/Reject based on Metropolis criterion (Simulated Tunneling).
        3. Cool down.
        """
        
        # Optimize: Calculate energies for current states if not already done
        for state in self.wavefunction.states:
            if state.energy == float('inf'):
                state.energy = simple_energy_function(state.configuration, bias_field=self.bias_field)
        
        # Parallel Perturbation (Simulating quantum fluctuations)
        for state in self.wavefunction.states:
            # 1. Create a mutant (Tunneling attempt)
            mutant = state.clone()
            
            # Mutate: Pick a random time step and shift it
            idx = random.randint(0, self.wavefunction.num_steps - 1)
            # Gaussian shift
            shift = random.gauss(0, 0.5 * self.temp) 
            # Clamp to [-1, 1] to allow braking (negative)
            new_val = max(-1.0, min(1.0, mutant.configuration[idx] + shift))
            mutant.configuration[idx] = new_val
            
            # Calculate new energy
            # Calculate energy
            mutant.energy = simple_energy_function(mutant.configuration, bias_field=self.bias_field)
            
            # 2. Acceptance (Metropolis-Hastings acting as Tunneling probability)
            # Delta E
            delta_E = mutant.energy - state.energy
            
            if delta_E < 0:
                # Lower energy: Always accept (Greedy / Relaxation)
                # Keep old state as "Ghost" before overwriting?
                # Maybe ghost is the *rejected* ones usually, but let's clear up what we treat as ghosts.
                # Actually, let's keep 'Ghost' as exploring paths.
                
                # Replace state
                state.configuration = mutant.configuration
                state.energy = mutant.energy
                
            else:
                # Higher energy: Tunnel probability
                # P = exp(-Delta E / T)
                tunnel_prob = math.exp(-delta_E / self.temp)
                
                if random.random() < tunnel_prob:
                    # Tunneled!
                    state.configuration = mutant.configuration
                    state.energy = mutant.energy
                else:
                    # Rejected - This attempt becomes a "Ghost Trajectory"
                    if random.random() < 0.05: # Increase ghost tracking slightly
                        self.ghost_trajectories.append(mutant.configuration)
        
        # Convergence Pressure:
        # Occasionally force some states to jump to the best known state
        # This simulates "Gravity" pulling the cloud together as it cools
        best_state = min(self.wavefunction.states, key=lambda s: s.energy)
        for state in self.wavefunction.states:
            if state.energy > best_state.energy * 1.5: # If very far from optimal
                if random.random() < 0.1: # 10% chance to collapse to best
                     state.configuration = copy.deepcopy(best_state.configuration)
                     state.energy = best_state.energy

        # 3. Cooling
        self.temp *= self.cooling_rate

    def run(self, steps: int = 100, yield_every: int = 10) -> Generator[Dict[str, Any], None, None]:
        for i in range(steps):
            self.step()
            
            if i % yield_every == 0:
                # Collect stats
                avg_energy = sum(s.energy for s in self.wavefunction.states) / len(self.wavefunction.states)
                collapse = self.wavefunction.collapse_metric()
                
                yield {
                    "step": i,
                    "temp": self.temp,
                    "avg_energy": avg_energy,
                    "collapse_metric": collapse, # Variance
                    "ghost_count": len(self.ghost_trajectories)
                }

