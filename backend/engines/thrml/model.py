"""
Generative Model for Thrust Schedule Sampling
==============================================

This module provides physics-guided thrust schedule generation using
the "Pure" THRML engine: A Heterogeneous Graph model with explicit
constraint nodes.

Key Features:
- Heterogeneous Graph Structure (TimeStepNode + FuelConstraintNode)
- Global "Back-Pressure" from FuelConstraintNode
- Stochastic Gibbs Sampling on the Graph
- Physics-aware local bias fields

Author: ASL-Sandbox Team
"""

import sys
import os
import jax
import jax.numpy as jnp
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
engines_dir = os.path.dirname(os.path.dirname(current_dir)) # backend
repo_root = os.path.dirname(engines_dir) # AstroLab2.0

# Add core to path (at repo root)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Import Pure THRML components
from .graph import HeterogeneousGraph, TimeStepNode, FuelConstraintNode
from .sampler import GibbsSampler

# Import physics-aware energy model from core
try:
    from core import (
        compute_physics_bias_field,
        compute_reference_trajectory_for_bias,
        update_bias_from_elite_samples,
        filter_schedules_by_fuel_budget,
        PhysicsGuidedScheduleGenerator,
        MU, EARTH_POS, MOON_POS,
        get_initial_state_4d
    )
    CORE_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: Could not import core energy model: {e}")
    CORE_AVAILABLE = False


# =============================================================================
# Pure THRML Graph Construction
# =============================================================================

def create_thrml_graph(
    num_steps: int, 
    coupling_strength: float, 
    external_field: jnp.ndarray,
    fuel_budget_fraction: float = 0.4
) -> HeterogeneousGraph:
    """
    Creates a HeterogeneousGraph for the THRML engine.
    
    Structure:
    - Chain of TimeStepNodes (Ising-like backbone)
    - One Global FuelConstraintNode connected to ALL TimeStepNodes
    
    Args:
        num_steps: Number of time steps
        coupling_strength: Smoothness weight (J)
        external_field: Local bias array (h_i)
        fuel_budget_fraction: Desired max fuel usage (0.0 - 1.0)
        
    Returns:
        HeterogeneousGraph instance
    """
    graph = HeterogeneousGraph()
    
    # 1. Create TimeStep Nodes
    prev_node = None
    for i in range(num_steps):
        bias = float(external_field[i])
        node = graph.add_timestep_node(time_index=i, dt=1.0, bias=bias)
        
        # Connect to previous (Smoothness Chain)
        if prev_node is not None:
            graph.add_edge(prev_node, node, weight=coupling_strength)
        prev_node = node
        
    # 2. Add Global Fuel Constraint Node (The "Back-Pressure" Source)
    # Convert fraction to total 'thrust units' (assuming 1 unit per step)
    max_fuel_units = num_steps * fuel_budget_fraction
    
    # Penalty strength needs to be tuned relative to local biases.
    # If too high, it freezes everything. If too low, it's ignored.
    fuel_penalty_strength = 2.0 
    
    fuel_node = graph.add_fuel_node(max_fuel=max_fuel_units, penalty_strength=fuel_penalty_strength)
    
    # Note: FuelConstraintNode logic in energy.py iterates over ALL timestep_nodes internally,
    # so we don't strictly need explicit edges for the computation, 
    # BUT for graph topology visualization later, explicit edges would be good.
    # For now, the energy calculation is implicit (global sum), so we skip adding N edges here
    # to save memory/complexity, relying on energy.py's implementation.
    
    return graph


# =============================================================================
# THRML-Based Schedule Generation
# =============================================================================

def generate_thrust_schedules_thrml(
    key: jax.random.PRNGKey,
    num_steps: int,
    batch_size: int,
    coupling_strength: float = 0.5,
    bias_field: Optional[jnp.ndarray] = None,
    n_warmup: int = 10,
    steps_per_sample: int = 2,
    fuel_budget_fraction: float = 0.4
) -> jnp.ndarray:
    """
    Generate thrust schedules using the Pure THRML Gibbs Sampler.
    
    Args:
        key: JAX random key (used for compatibility, though sampler uses random)
        num_steps: Length of schedule
        batch_size: Number of schedules to generate
        coupling_strength: Smoothness
        bias_field: External field h
        n_warmup: Warmup steps
        fuel_budget_fraction: Target fuel usage
        
    Returns:
        schedules: (batch_size, num_steps) binary array {0, 1}
    """
    # Default bias field if not provided
    if bias_field is None:
        bias_field = jnp.zeros(num_steps)
    
    schedules_list = []
    
    # Note: We are running a Pure Python object-based sampler in a loop.
    # This is slower than vectorized JAX/NumPy but implements the strict 
    # "Pure Architecture" requested.
    
    for i in range(batch_size):
        # 1. Create a fresh graph for this sample
        # (Re-creating is safer than resetting to avoid state leakage)
        graph = create_thrml_graph(
            num_steps, 
            coupling_strength, 
            bias_field,
            fuel_budget_fraction
        )
        
        # 2. Initialize Sampler
        # Temperature 1.0 (beta=1.0) is standard
        sampler = GibbsSampler(graph, beta=1.0)
        
        # 3. Randomize Initial State
        # (GibbsSampler starts with 0.0, we can randomize if we want, 
        # but warmup usually handles this)
        
        # 4. Run Sampler
        # We run for n_warmup steps, then take the state
        # The 'run' method yields stats, but we just want to execute
        
        # Manually pump the steps
        for _ in range(n_warmup):
            sampler.step()
            
        # 5. Extract Schedule
        # Get values from TimeStepNodes, sorted by time index
        # (Though list append order is usually preserved, sorting is safer)
        sorted_nodes = sorted(graph.timestep_nodes, key=lambda n: n.time_index)
        # Ternary logic: values are -1.0, 0.0, 1.0
        schedule = [float(n.value) for n in sorted_nodes]
        schedules_list.append(schedule)
        
    return jnp.array(schedules_list, dtype=jnp.float32)


# =============================================================================
# Physics-Guided Schedule Generation
# =============================================================================

def generate_physics_guided_schedules(
    key: jax.random.PRNGKey,
    num_steps: int,
    batch_size: int,
    coupling_strength: float = 0.5,
    initial_state: Optional[jnp.ndarray] = None,
    thrust_accel: float = 0.01,
    dt: float = 0.01,
    fuel_budget_fraction: float = 0.4,
    method: str = "thrml"
) -> Tuple[jnp.ndarray, Dict[str, Any]]:
    """
    Generate physics-guided thrust schedules with fuel budget filtering.
    """
    k1, k2, k3 = jax.random.split(key, 3)
    
    # Get initial state
    if initial_state is None:
        initial_state = get_initial_state_4d(200.0) if CORE_AVAILABLE else jnp.array([0.017, 0, 0, 1.0])
    
    # Compute physics-aware bias field
    if CORE_AVAILABLE:
        ref_traj = compute_reference_trajectory_for_bias(
            num_steps, dt, thrust_accel, initial_state, fuel_budget_fraction
        )
        bias_field = compute_physics_bias_field(
            num_steps,
            ref_traj,
            fuel_budget_fraction=fuel_budget_fraction,
            arrival_coast_fraction=0.15,
            periapsis_boost=2.0,
            apoapsis_penalty=-1.0,
            arrival_coast_strength=-3.0
        )
    else:
        # Simple linear bias
        bias_field = jnp.linspace(0.5, -1.0, num_steps)
        bias_field = bias_field + (fuel_budget_fraction - 0.5) * 2.0
    
    # Generate schedules using Pure THRML
    if method == "thrml":
        # Pass fuel_budget_fraction to the graph for the Constraint Node
        raw_schedules = generate_thrust_schedules_thrml(
            k2, 
            num_steps, 
            batch_size, 
            coupling_strength, 
            bias_field,
            fuel_budget_fraction=fuel_budget_fraction
        )
    else:
        # Random fallback
        probs = jax.nn.sigmoid(bias_field)
        raw_schedules = jax.random.bernoulli(k2, probs, (batch_size, num_steps)).astype(jnp.float32)
    
    # Filter by fuel budget (Post-processing check)
    # Even with the FuelConstraintNode, soft constraints might be violated,
    # so we keep the hard filter for safety.
    if CORE_AVAILABLE:
        valid_schedules, valid_mask = filter_schedules_by_fuel_budget(
            raw_schedules,
            max_thrust_fraction=fuel_budget_fraction + 0.15,
            min_thrust_fraction=max(0.05, fuel_budget_fraction - 0.15)
        )
    else:
        thrust_fractions = jnp.mean(jnp.abs(raw_schedules), axis=1)
        valid_mask = (thrust_fractions >= 0.1) & (thrust_fractions <= 0.7)
        valid_schedules = raw_schedules[valid_mask]
    
    # Pad if needed
    if len(valid_schedules) >= batch_size:
        output_schedules = valid_schedules[:batch_size]
    else:
        n_valid = len(valid_schedules)
        if n_valid > 0:
            repeats = (batch_size // n_valid) + 1
            output_schedules = jnp.tile(valid_schedules, (repeats, 1))[:batch_size]
        else:
            output_schedules = raw_schedules[:batch_size]
            
    metadata = {
        'bias_field': bias_field,
        'coupling_strength': coupling_strength,
        'mean_thrust_fraction': float(jnp.mean(jnp.abs(output_schedules))),
        'method': method,
        'physics_guided': CORE_AVAILABLE
    }
    
    return output_schedules, metadata


# =============================================================================
# Legacy API 
# =============================================================================

def generate_thrust_schedules(
    key: jax.random.PRNGKey,
    num_steps: int,
    batch_size: int,
    coupling_strength: float = 0.5,
    eclipse_indices: List[int] = [],
    perigee_indices: List[int] = [],
    bias_field: Optional[jnp.ndarray] = None
) -> jnp.ndarray:
    """Legacy API wrapper."""
    if bias_field is None:
        bias_field = jnp.zeros(num_steps)
        if eclipse_indices:
            bias_field = bias_field.at[jnp.array(eclipse_indices)].set(-10.0)
        if perigee_indices:
            bias_field = bias_field.at[jnp.array(perigee_indices)].add(2.0)
            
    return generate_thrust_schedules_thrml(
        key, num_steps, batch_size, coupling_strength, bias_field
    )
