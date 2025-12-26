"""
Pure Quantum-Inspired Schedule Generation
=========================================

This module provides binary thrust schedule generation using the custom
QuantumAnnealer and WaveFunction classes, simulating a quantum tunneling
process over a population of states.

Key Features:
- Pure Python Implementation (No D-Wave deps)
- WaveFunction Collapse Logic
- Physics-Aware Bias Fields via Energy Function
- "Tunneling" through energy barriers

Author: ASL-Sandbox Team
"""

import sys
import os
import numpy as np
import jax.numpy as jnp
from typing import Dict, Any, Optional, List, Tuple

# Add core to path
current_dir = os.path.dirname(os.path.abspath(__file__))
engines_dir = os.path.dirname(os.path.dirname(current_dir)) # backend
repo_root = os.path.dirname(engines_dir) # AstroLab2.0

if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Import Pure QNTM Components
from .state import WaveFunction
from .annealer import QuantumAnnealer

# Import physics-aware components from core
try:
    from core import (
        compute_physics_bias_field,
        compute_reference_trajectory_for_bias,
        update_bias_from_elite_samples,
        filter_schedules_by_fuel_budget,
        MU, EARTH_POS, MOON_POS,
        get_initial_state_4d
    )
    CORE_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: Could not import core: {e}")
    CORE_AVAILABLE = False


class SimulatedQuantumAnnealer:
    """
    Wrapper for the Pure QuantumAnnealer to match the previous API structure.
    """
    
    def __init__(self):
        """Initialize."""
        pass
    
    def generate_thrust_schedules(
        self,
        num_steps: int,
        batch_size: int,
        coupling_strength: float = 1.0,
        bias: float = 0.0,
        physics_bias_field: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Generate thrust schedules using the WaveFunction collapse simulation.
        """
        
        # 1. Initialize WaveFunction (Population)
        wavefunction = WaveFunction(population_size=batch_size, num_steps=num_steps)
        
        # 2. Prepare Bias Field
        # Convert numpy/jax array to list for Pure Python logic
        bias_list = [bias] * num_steps
        if physics_bias_field is not None:
             bias_list = [float(x) for x in physics_bias_field]
             
        # 3. Create Annealer
        annealer = QuantumAnnealer(
            wavefunction=wavefunction, 
            cooling_rate=0.95,
            bias_field=bias_list
        )
        
        # 4. Run Simulation
        # Run for sufficient steps to allow collapse
        sim_steps = 100
        # We drain the generator to run it
        for _ in annealer.run(steps=sim_steps):
            pass
            
        # 5. Extract Results
        # Get final configurations (Probability Cloud)
        schedules = [s.configuration for s in wavefunction.states]
        energies = [s.energy for s in wavefunction.states]
        
        # Binarize/Ternarize for output
        # Map continuous [-1, 1] to {-1, 0, 1}
        # Thresholds: > 0.33 -> 1, < -0.33 -> -1, else 0
        arr = jnp.array(schedules)
        ternary_schedules = jnp.where(arr > 0.33, 1.0, jnp.where(arr < -0.33, -1.0, 0.0))
        
        return {
            "schedules": ternary_schedules.astype(jnp.float32),
            "energies": jnp.array(energies, dtype=jnp.float32),
            "h": {}, # Legacy compat
            "J": {}  # Legacy compat
        }
    
    def generate_physics_guided_schedules(
        self,
        num_steps: int,
        batch_size: int,
        coupling_strength: float = 1.0,
        initial_state: Optional[np.ndarray] = None,
        thrust_accel: float = 0.01,
        dt: float = 0.01,
        fuel_budget_fraction: float = 0.4
    ) -> Dict[str, Any]:
        """
        Generate physics-aware thrust schedules using Pure QNTM logic.
        """
        # Get initial state
        if initial_state is None:
            if CORE_AVAILABLE:
                initial_state = np.array(get_initial_state_4d(200.0))
            else:
                initial_state = np.array([0.017, 0, 0, 1.0])
        
        # Compute physics-aware bias field
        if CORE_AVAILABLE:
            ref_traj = compute_reference_trajectory_for_bias(
                num_steps, dt, thrust_accel, 
                jnp.array(initial_state[:4]), 
                fuel_budget_fraction
            )
            bias_field = compute_physics_bias_field(
                num_steps, ref_traj, fuel_budget_fraction
            )
            physics_bias = np.array(bias_field)
        else:
            physics_bias = np.linspace(0.5, -1.0, num_steps)
            physics_bias += (fuel_budget_fraction - 0.5) * 2.0
        
        # Generate with Pure Annealer
        result = self.generate_thrust_schedules(
            num_steps,
            batch_size,
            coupling_strength,
            physics_bias_field=physics_bias
        )
        
        raw_schedules = result['schedules']
        raw_energies = result['energies']
        
        # Filter by fuel budget
        if CORE_AVAILABLE:
            valid_schedules, valid_mask = filter_schedules_by_fuel_budget(
                raw_schedules,
                max_thrust_fraction=fuel_budget_fraction + 0.2,
                min_thrust_fraction=max(0.05, fuel_budget_fraction - 0.2)
            )
            valid_energies = raw_energies[valid_mask] if len(raw_energies) > 0 else []
        else:
            thrust_fractions = jnp.mean(jnp.abs(raw_schedules), axis=1)
            valid_mask = (thrust_fractions >= 0.1) & (thrust_fractions <= 0.7)
            valid_schedules = raw_schedules[valid_mask]
            valid_energies = raw_energies[valid_mask]
        
        # Pad results
        if len(valid_schedules) >= batch_size:
            output_schedules = valid_schedules[:batch_size]
            output_energies = valid_energies[:batch_size]
        else:
            n_valid = len(valid_schedules)
            if n_valid > 0:
                repeats = (batch_size // n_valid) + 1
                output_schedules = jnp.tile(valid_schedules, (repeats, 1))[:batch_size]
                output_energies = jnp.tile(valid_energies, repeats)[:batch_size]
            else:
                output_schedules = raw_schedules[:batch_size]
                output_energies = raw_energies[:batch_size]
        
        return {
            "schedules": output_schedules,
            "energies": output_energies,
            "h": result['h'],
            "J": result['J'],
            "metadata": {
                "physics_bias_field": physics_bias.tolist(),
                "coupling_strength": coupling_strength,
                "fuel_budget_fraction": fuel_budget_fraction,
                "valid_fraction": float(jnp.sum(valid_mask) / len(valid_mask)),
                "physics_guided": CORE_AVAILABLE
            }
        }


# =============================================================================
# Comparison Utilities
# =============================================================================

def generate_random_schedules(
    num_steps: int,
    batch_size: int,
    thrust_probability: float = 0.5
) -> jnp.ndarray:
    key = jax.random.PRNGKey(int(np.random.randint(0, 100000)))
    return jax.random.bernoulli(key, thrust_probability, (batch_size, num_steps)).astype(jnp.float32)


def compare_methods(
    num_steps: int,
    batch_size: int,
    coupling_strength: float = 1.0,
    fuel_budget_fraction: float = 0.4
) -> Dict[str, Dict[str, Any]]:
    annealer = SimulatedQuantumAnnealer()
    
    quantum_result = annealer.generate_physics_guided_schedules(
        num_steps, batch_size, coupling_strength,
        fuel_budget_fraction=fuel_budget_fraction
    )
    
    random_schedules = generate_random_schedules(
        num_steps, batch_size, thrust_probability=fuel_budget_fraction
    )
    
    quantum_thrust_fracs = jnp.mean(jnp.abs(quantum_result['schedules']), axis=1)
    random_thrust_fracs = jnp.mean(jnp.abs(random_schedules), axis=1)
    
    return {
        'quantum': {
            'mean_thrust_fraction': float(jnp.mean(quantum_thrust_fracs)),
            'std_thrust_fraction': float(jnp.std(quantum_thrust_fracs)),
            'mean_energy': float(jnp.mean(quantum_result['energies'])),
            'physics_guided': quantum_result['metadata']['physics_guided']
        },
        'random': {
            'mean_thrust_fraction': float(jnp.mean(random_thrust_fracs)),
            'std_thrust_fraction': float(jnp.std(random_thrust_fracs)),
            'mean_energy': None 
        }
    }


# =============================================================================
# Module Exports
# =============================================================================

import jax

__all__ = [
    'SimulatedQuantumAnnealer',
    'generate_random_schedules',
    'compare_methods',
    'CORE_AVAILABLE'
]
