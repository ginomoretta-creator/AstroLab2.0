"""
QNTM State Representation
=========================

Defines the `WaveFunction` which represents the superposition of multiple 
trajectory states. Unlike the classical solver which tracks one "best" path, 
the WaveFunction maintains a population of potential realities.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
import random
import copy
import math

# Reuse the Node structure from THRML if useful, or define a lighter weight state
# For QNTM, a "State" is a full schedule configuration (a classic 'particle' in the search space).

@dataclass
class QuantumState:
    """
    A single "collapsed" reality (one specific trajectory configuration).
    """
    id: int
    configuration: List[float] # The spin values [0, 1] or continuous thrust [0.0 - 1.0]
    energy: float = float('inf')
    
    def clone(self):
        return QuantumState(
            id=random.randint(0, 1000000), 
            configuration=copy.deepcopy(self.configuration),
            energy=self.energy
        )

class WaveFunction:
    """
    Represents the probability cloud of the system.
    Maintains a population of QuantumStates (particles).
    """
    def __init__(self, population_size: int = 50, num_steps: int = 20):
        self.population_size = population_size
        self.num_steps = num_steps
        self.states: List[QuantumState] = []
        self.best_state: QuantumState = None
        
        # Initialize superposition (Random Chaos)
        self._initialize_superposition()
        
    def _initialize_superposition(self):
        """Create initial random population."""
        self.states = []
        for i in range(self.population_size):
            # Random configuration: continuous 0.0 to 1.0
            config = [random.random() for _ in range(self.num_steps)]
            state = QuantumState(id=i, configuration=config)
            self.states.append(state)
            
    def get_probability_cloud(self) -> List[List[float]]:
        """
        Returns the raw configuration data for all states in the population.
        Used for visualization (drawing the "Tube").
        """
        return [s.configuration for s in self.states]
        
    def get_mean_trajectory(self) -> List[float]:
        """Collapsed average state (Expectation Value)."""
        if not self.states:
            return []
        
        avgs = [0.0] * self.num_steps
        for s in self.states:
            for i, val in enumerate(s.configuration):
                avgs[i] += val
        
        return [x / len(self.states) for x in avgs]
    
    def collapse_metric(self) -> float:
        """
        Returns a measure of how 'collapsed' the wavefunction is.
        High variance = Superposition (High Uncertainty).
        Low variance = Collapsed (Solution Found).
        """
        if not self.states:
            return 0.0
            
        # Calculate variance across the population for each time step
        total_variance = 0.0
        mean_traj = self.get_mean_trajectory()
        
        for i in range(self.num_steps):
            step_variance = 0.0
            for s in self.states:
                diff = s.configuration[i] - mean_traj[i]
                step_variance += diff * diff
            total_variance += step_variance / len(self.states)
            
        return total_variance
