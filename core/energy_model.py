"""
Physics-Aware Energy Model for Ising-Based Trajectory Sampling
===============================================================

This module provides physics-informed bias fields for the 1D Ising model
used to generate thrust schedules. The key insight is that optimal low-thrust
trajectories have predictable structure:

- Thrust at periapsis (efficient for orbit raising)
- Coast at apoapsis (except for plane changes)
- Coast during lunar approach (for capture)
- Respect fuel budget constraints

By encoding this domain knowledge into the Ising energy function, we guide
the sampling toward physically meaningful thrust patterns.

Author: ASL-Sandbox Team
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Optional, Tuple, Dict, Any

from .constants import MU, EARTH_POS, MOON_POS
from .physics_core import (
    propagate_trajectory_4state,
    detect_periapsis_apoapsis,
    get_initial_state_4d
)


# =============================================================================
# Physics-Aware Bias Field Computation
# =============================================================================

def compute_physics_bias_field(
    num_steps: int,
    reference_trajectory: Optional[jnp.ndarray] = None,
    fuel_budget_fraction: float = 0.5,
    arrival_coast_fraction: float = 0.15,
    periapsis_boost: float = 2.0,
    apoapsis_penalty: float = -1.0,
    arrival_coast_strength: float = -3.0,
    smoothing_window: int = 5
) -> jnp.ndarray:
    """
    Computes a physics-informed bias field for the Ising model.
    
    The bias field h_i for each time step encodes our prior knowledge about
    where thrust should be applied:
    
    - h > 0: Encourage thrust (spin = +1)
    - h < 0: Discourage thrust (spin = -1, coast)
    - h = 0: Neutral
    
    Components:
    1. Periapsis bias: Positive bias near periapsis (efficient thrust)
    2. Apoapsis bias: Negative bias near apoapsis (coast)
    3. Arrival coast: Strong negative bias in final portion
    4. Global fuel constraint: Shift mean to match desired duty cycle
    
    Args:
        num_steps: Number of time steps in schedule
        reference_trajectory: (N, 4) trajectory from reference propagation
                            If None, uses a default spiral reference
        fuel_budget_fraction: Target fraction of time with thrust on (0-1)
        arrival_coast_fraction: Fraction of trajectory to coast at end
        periapsis_boost: Positive bias strength at periapsis
        apoapsis_penalty: Negative bias strength at apoapsis
        arrival_coast_strength: Negative bias for arrival coast window
        smoothing_window: Gaussian smoothing window for orbital detection
        
    Returns:
        bias_field: (num_steps,) array of bias values
    """
    field = jnp.zeros(num_steps)
    
    # 1. Orbital phase-aware biasing (if reference trajectory available)
    if reference_trajectory is not None and len(reference_trajectory) >= num_steps:
        # Detect periapsis and apoapsis
        periapsis_mask, apoapsis_mask = detect_periapsis_apoapsis(reference_trajectory[:num_steps])
        
        # Apply Gaussian smoothing to create gradual bias regions
        if smoothing_window > 1:
            kernel = jnp.exp(-jnp.linspace(-2, 2, smoothing_window)**2)
            kernel = kernel / kernel.sum()
            
            periapsis_smooth = jnp.convolve(periapsis_mask.astype(float), kernel, mode='same')
            apoapsis_smooth = jnp.convolve(apoapsis_mask.astype(float), kernel, mode='same')
        else:
            periapsis_smooth = periapsis_mask.astype(float)
            apoapsis_smooth = apoapsis_mask.astype(float)
        
        # Add orbital phase bias
        field = field + periapsis_boost * periapsis_smooth
        field = field + apoapsis_penalty * apoapsis_smooth
    
    # 2. Arrival coast window
    arrival_start = int(num_steps * (1 - arrival_coast_fraction))
    arrival_taper = jnp.linspace(0, 1, num_steps - arrival_start)  # Gradual increase
    arrival_bias = jnp.zeros(num_steps)
    arrival_bias = arrival_bias.at[arrival_start:].set(arrival_coast_strength * arrival_taper)
    field = field + arrival_bias
    
    # 3. Global fuel budget constraint (mean-field bias)
    # To achieve target thrust fraction p, we shift the field such that
    # the expected value of σ(tanh(β * h)) ≈ p
    # For symmetric Ising, h = 0 gives 50% thrust
    # To get p, we need h ≈ arctanh(2p - 1) / β
    # Simplified: linear shift
    target_shift = (fuel_budget_fraction - 0.5) * 4.0  # Heuristic
    field = field + target_shift
    
    return field


def compute_reference_trajectory_for_bias(
    num_steps: int,
    dt: float,
    thrust_accel: float,
    initial_state: Optional[jnp.ndarray] = None,
    constant_thrust_fraction: float = 0.3
) -> jnp.ndarray:
    """
    Generate a reference trajectory with constant thrust fraction.
    
    This provides orbital structure (periapsis/apoapsis locations) for
    physics-aware bias computation.
    
    Args:
        num_steps: Number of time steps
        dt: Time step
        thrust_accel: Thrust acceleration magnitude
        initial_state: Initial state (defaults to 200km LEO)
        constant_thrust_fraction: Fraction of time with thrust on
        
    Returns:
        reference_trajectory: (num_steps+1, 4) state history
    """
    if initial_state is None:
        initial_state = get_initial_state_4d(200.0)
    
    # Create a simple thrust schedule: thrust for first X% of each orbit
    # For a rough reference, just use constant thrust at reduced level
    constant_schedule = jnp.full(num_steps, constant_thrust_fraction * thrust_accel)
    
    reference_traj = propagate_trajectory_4state(
        initial_state[:4], constant_schedule, dt, num_steps
    )
    
    return reference_traj


# =============================================================================
# Adaptive Bias Field (Cross-Entropy Method Style)
# =============================================================================

def update_bias_from_elite_samples(
    elite_schedules: jnp.ndarray,
    current_bias: jnp.ndarray,
    learning_rate: float = 0.5,
    smoothing: bool = True,
    smoothing_window: int = 5
) -> jnp.ndarray:
    """
    Update bias field based on elite (best-performing) schedules.
    
    This implements a Cross-Entropy Method (CEM) style update where
    we bias the Ising model toward patterns that worked well.
    
    Args:
        elite_schedules: (K, N) binary schedules from top performers
        current_bias: (N,) current bias field
        learning_rate: How much to update (0=no update, 1=full update)
        smoothing: Whether to smooth the updated bias
        smoothing_window: Window size for Gaussian smoothing
        
    Returns:
        updated_bias: (N,) new bias field
    """
    # Compute average of elite schedules
    # Schedules are in {0, 1}, map to {-1, +1} for Ising interpretation
    elite_mean = jnp.mean(elite_schedules, axis=0)  # In [0, 1]
    
    # Convert to bias: if mean is 0.5, bias should be 0
    # If mean is 1.0, bias should be positive
    # If mean is 0.0, bias should be negative
    target_bias = (elite_mean - 0.5) * 4.0  # Scale factor
    
    # Smooth update
    updated_bias = (1 - learning_rate) * current_bias + learning_rate * target_bias
    
    # Optional smoothing to encourage coherent thrust arcs
    if smoothing and smoothing_window > 1:
        kernel = jnp.exp(-jnp.linspace(-2, 2, smoothing_window)**2)
        kernel = kernel / kernel.sum()
        updated_bias = jnp.convolve(updated_bias, kernel, mode='same')
    
    return updated_bias


# =============================================================================
# Constraint-Aware Schedule Filtering
# =============================================================================

def filter_schedules_by_fuel_budget(
    schedules: jnp.ndarray,
    max_thrust_fraction: float = 0.6,
    min_thrust_fraction: float = 0.1
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Filter schedules to those respecting fuel budget constraints.
    
    Args:
        schedules: (batch_size, num_steps) binary schedules
        max_thrust_fraction: Maximum allowed thrust duty cycle
        min_thrust_fraction: Minimum required thrust duty cycle
        
    Returns:
        valid_schedules: Filtered schedules
        valid_mask: Boolean mask of which schedules were valid
    """
    thrust_fractions = jnp.mean(jnp.abs(schedules), axis=1)
    valid_mask = (thrust_fractions >= min_thrust_fraction) & (thrust_fractions <= max_thrust_fraction)
    
    return schedules[valid_mask], valid_mask


def repair_schedule_fuel_budget(
    schedule: jnp.ndarray,
    target_thrust_fraction: float = 0.4,
    key: jax.random.PRNGKey = None
) -> jnp.ndarray:
    """
    Repair a schedule to meet fuel budget constraint.
    
    If too much thrust: randomly turn off some thrust windows
    If too little thrust: randomly turn on some coast windows
    
    Args:
        schedule: (num_steps,) binary schedule
        target_thrust_fraction: Target duty cycle
        key: Random key for stochastic repair
        
    Returns:
        repaired_schedule: Schedule meeting target thrust fraction
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    
    current_fraction = jnp.mean(schedule)
    target_thrust_steps = int(len(schedule) * target_thrust_fraction)
    current_thrust_steps = int(jnp.sum(schedule))
    
    repaired = schedule.copy()
    
    if current_thrust_steps > target_thrust_steps:
        # Need to remove some thrust
        thrust_indices = jnp.where(schedule > 0.5)[0]
        n_remove = current_thrust_steps - target_thrust_steps
        remove_indices = jax.random.choice(key, thrust_indices, (n_remove,), replace=False)
        repaired = repaired.at[remove_indices].set(0.0)
        
    elif current_thrust_steps < target_thrust_steps:
        # Need to add some thrust
        coast_indices = jnp.where(schedule < 0.5)[0]
        n_add = target_thrust_steps - current_thrust_steps
        add_indices = jax.random.choice(key, coast_indices, (min(n_add, len(coast_indices)),), replace=False)
        repaired = repaired.at[add_indices].set(1.0)
    
    return repaired


# =============================================================================
# Eclipse Constraint (For Future Use)
# =============================================================================

def compute_eclipse_windows(
    trajectory: jnp.ndarray,
    earth_shadow_cone_angle: float = 0.26  # ~15 degrees in radians
) -> jnp.ndarray:
    """
    Compute which trajectory points are in Earth's shadow.
    
    Simplified model: Shadow is a cylinder behind Earth.
    
    Args:
        trajectory: (N, 4 or 5) state history
        earth_shadow_cone_angle: Half-angle of umbra cone (radians)
        
    Returns:
        eclipse_mask: (N,) boolean mask, True if in eclipse
    """
    from .constants import R_EARTH_NORM
    
    positions = trajectory[:, :2]
    
    # Sun direction (simplified: always along +x in rotating frame)
    # This is a rough approximation; real model would use ephemerides
    sun_dir = jnp.array([1.0, 0.0])
    
    # Check if behind Earth (negative x relative to Earth)
    rel_pos = positions - EARTH_POS
    behind_earth = rel_pos[:, 0] < 0
    
    # Check if within shadow cylinder (|y| < R_Earth)
    in_cylinder = jnp.abs(rel_pos[:, 1]) < R_EARTH_NORM
    
    eclipse_mask = behind_earth & in_cylinder
    
    return eclipse_mask


def apply_eclipse_constraint(
    bias_field: jnp.ndarray,
    trajectory: jnp.ndarray,
    eclipse_penalty: float = -20.0
) -> jnp.ndarray:
    """
    Add strong negative bias during eclipse windows (no solar power for thrust).
    
    Args:
        bias_field: Current bias field
        trajectory: Reference trajectory for eclipse detection
        eclipse_penalty: Strong negative bias during eclipse
        
    Returns:
        Updated bias field with eclipse constraint
    """
    eclipse_mask = compute_eclipse_windows(trajectory)
    
    # Truncate or pad eclipse mask to match bias field length
    n_bias = len(bias_field)
    n_eclipse = len(eclipse_mask)
    
    if n_eclipse > n_bias:
        eclipse_mask = eclipse_mask[:n_bias]
    elif n_eclipse < n_bias:
        eclipse_mask = jnp.pad(eclipse_mask, (0, n_bias - n_eclipse), constant_values=False)
    
    updated_field = bias_field + eclipse_penalty * eclipse_mask.astype(float)
    
    return updated_field


# =============================================================================
# Complete Physics-Guided Sampling Pipeline
# =============================================================================

class PhysicsGuidedScheduleGenerator:
    """
    Encapsulates the physics-guided schedule generation pipeline.
    
    This class manages:
    - Reference trajectory computation
    - Physics-aware bias field
    - Iterative bias updates (CEM-style)
    - Fuel budget constraints
    """
    
    def __init__(
        self,
        num_steps: int,
        dt: float,
        thrust_accel: float,
        initial_state: jnp.ndarray,
        isp_normalized: float = 300.0,
        fuel_budget_fraction: float = 0.4,
        coupling_strength: float = 1.0
    ):
        """
        Initialize the generator.
        
        Args:
            num_steps: Number of time steps
            dt: Time step (normalized)
            thrust_accel: Thrust acceleration magnitude (normalized)
            initial_state: Initial state (4 or 5 elements)
            isp_normalized: Specific impulse (normalized)
            fuel_budget_fraction: Target thrust duty cycle
            coupling_strength: Ising coupling strength (smoothness)
        """
        self.num_steps = num_steps
        self.dt = dt
        self.thrust_accel = thrust_accel
        self.initial_state = initial_state[:4] if len(initial_state) > 4 else initial_state
        self.isp_normalized = isp_normalized
        self.fuel_budget_fraction = fuel_budget_fraction
        self.coupling_strength = coupling_strength
        
        # Compute reference trajectory
        self.reference_trajectory = compute_reference_trajectory_for_bias(
            num_steps, dt, thrust_accel, self.initial_state, fuel_budget_fraction
        )
        
        # Initialize bias field
        self.bias_field = compute_physics_bias_field(
            num_steps,
            self.reference_trajectory,
            fuel_budget_fraction
        )
    
    def get_bias_field(self) -> jnp.ndarray:
        """Get current bias field."""
        return self.bias_field
    
    def get_coupling_strength(self) -> float:
        """Get Ising coupling strength."""
        return self.coupling_strength
    
    def update_from_elites(
        self,
        elite_schedules: jnp.ndarray,
        elite_trajectories: jnp.ndarray,
        learning_rate: float = 0.3
    ):
        """
        Update bias field based on elite samples.
        
        Args:
            elite_schedules: (K, num_steps) best schedules
            elite_trajectories: (K, num_steps+1, 4) corresponding trajectories
            learning_rate: Update strength
        """
        # Update from elite schedule statistics
        self.bias_field = update_bias_from_elite_samples(
            elite_schedules,
            self.bias_field,
            learning_rate
        )
        
        # Optionally re-compute reference from best trajectory
        if len(elite_trajectories) > 0:
            best_traj = elite_trajectories[0]
            
            # Re-detect orbital phases from best trajectory
            periapsis_mask, apoapsis_mask = detect_periapsis_apoapsis(best_traj)
            
            # Add subtle orbital phase reinforcement
            self.bias_field = self.bias_field + 0.1 * (periapsis_mask.astype(float) - apoapsis_mask.astype(float))
    
    def filter_schedules(self, schedules: jnp.ndarray) -> jnp.ndarray:
        """Filter schedules by fuel budget."""
        valid, _ = filter_schedules_by_fuel_budget(
            schedules,
            max_thrust_fraction=self.fuel_budget_fraction + 0.2,
            min_thrust_fraction=max(0, self.fuel_budget_fraction - 0.2)
        )
        return valid
    
    def get_ising_parameters(self) -> Dict[str, Any]:
        """
        Get parameters for Ising model construction.
        
        Returns:
            Dictionary with 'biases', 'coupling', 'beta' for IsingEBM
        """
        return {
            'biases': self.bias_field,
            'coupling': self.coupling_strength,
            'beta': 1.0,  # Inverse temperature
            'num_steps': self.num_steps
        }


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    'compute_physics_bias_field',
    'compute_reference_trajectory_for_bias',
    'update_bias_from_elite_samples',
    'filter_schedules_by_fuel_budget',
    'repair_schedule_fuel_budget',
    'compute_eclipse_windows',
    'apply_eclipse_constraint',
    'PhysicsGuidedScheduleGenerator',
]
