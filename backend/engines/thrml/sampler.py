"""
THRML Gibbs Sampler
===================

Implements the sampling logic for the Heterogeneous Graph.
Uses the Energy logic to compute effective fields and updates node states
probabilistically (Gibbs Sampling).
"""

import math
import random
from typing import List, Dict, Any, Generator

from .graph import HeterogeneousGraph, TimeStepNode
from .energy import compute_local_field_components, compute_fuel_penalty_cost, compute_total_energy

def sigmoid(x: float) -> float:
    try:
        if x < -700: # avoid underflow
            return 0.0
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0

class GibbsSampler:
    """
    Orchestrates the sampling process on the graph.
    """
    def __init__(self, graph: HeterogeneousGraph, beta: float = 1.0):
        self.graph = graph
        self.beta = beta
        
    def step(self):
        """
        Performs one full sweep (one update per node on average) over the graph.
        """
        # Get all dynamic nodes (TimeSteps)
        nodes = self.graph.timestep_nodes
        
        # Shuffle for random update order (Gibbs requirement for correctness, usually)
        # Creating a list indices is cheaper than shuffling objects if list is large, 
        # but objects are references so it's fine.
        indices = list(range(len(nodes)))
        random.shuffle(indices)
        
        for idx in indices:
            node = nodes[idx]
            
            # 1. Compute components of Energy Landscape
            # E(s) = -h_linear * s + MetricPenalty(|s|)
            # s in {-1, 0, 1}
            
            h_linear = compute_local_field_components(self.graph, node)
            fuel_cost = compute_fuel_penalty_cost(self.graph, node)
            
            # Energies of the 3 states (relative to some baseline)
            # E(0) = 0
            # E(1) = -h_linear * 1 + fuel_cost
            # E(-1) = -h_linear * (-1) + fuel_cost = h_linear + fuel_cost
            
            E_0 = 0.0
            E_plus = -h_linear + fuel_cost
            E_minus = h_linear + fuel_cost
            
            # 2. Compute Probabilities (Boltzmann / Softmax)
            # P(s) = exp(-beta * E(s)) / Z
            
            # For numerical stability, subtract min E
            min_E = min(E_0, E_plus, E_minus)
            
            w_0 = math.exp(-self.beta * (E_0 - min_E))
            w_plus = math.exp(-self.beta * (E_plus - min_E))
            w_minus = math.exp(-self.beta * (E_minus - min_E))
            
            Z = w_0 + w_plus + w_minus
            
            p_0 = w_0 / Z
            p_plus = w_plus / Z
            p_minus = w_minus / Z
            
            # 3. Sample
            r = random.random()
            if r < p_minus:
                node.value = -1.0
            elif r < p_minus + p_0:
                node.value = 0.0
            else:
                node.value = 1.0
                
    def run(self, n_steps: int, yield_every: int = 1) -> Generator[Dict[str, Any], None, None]:
        """
        Runs the sampler for n_steps sweeps.
        Yields stats periodically.
        """
        for i in range(n_steps):
            self.step()
            
            if i % yield_every == 0:
                energy = compute_total_energy(self.graph)
                
                # Calculate active thrust count
                active_count = sum(1 for n in self.graph.timestep_nodes if n.value > 0.5)
                
                yield {
                    "step": i,
                    "energy": energy,
                    "active_nodes": active_count,
                    "temperature": 1.0 / self.beta
                }
                
    def set_temperature(self, temp: float):
        if temp <= 0:
            raise ValueError("Temperature must be positive")
        self.beta = 1.0 / temp
