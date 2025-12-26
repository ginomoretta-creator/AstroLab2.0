"""
Core Physics Engine for CR3BP Low-Thrust Trajectory Propagation
================================================================

This module provides the unified physics engine for both THRML-Sandbox and
QNTM-Sandbox projects. Key features:

- 5-state dynamics [x, y, vx, vy, m] with mass depletion
- Fuel consumption via specific impulse (Isp)
- Mass-dependent thrust acceleration
- Perigee/apoapsis detection for physics-aware sampling
- JAX-accelerated batch propagation
- CasADi-compatible NumPy implementations

Author: ASL-Sandbox Team
"""

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import Tuple, Optional, NamedTuple

from .constants import (
    MU, EPSILON, G0_NORM, G0_MS2, A_STAR_MS2, T_STAR_S, L_STAR_KM,
    EARTH_POS, MOON_POS, R_EARTH_NORM, R_MOON_NORM, LUNAR_SOI_NORM
)


# =============================================================================
# Data Structures
# =============================================================================

class TrajectoryResult(NamedTuple):
    """Result of trajectory propagation."""
    trajectory: jnp.ndarray      # (N+1, 5) states: [x, y, vx, vy, m]
    final_distance_to_moon: float
    min_distance_to_moon: float  # Added for collision check
    final_velocity_magnitude: float
    total_fuel_used: float
    is_valid: bool               # True if trajectory didn't diverge


class OrbitalElements(NamedTuple):
    """Osculating orbital elements (Earth-centered approximation)."""
    semi_major_axis: float
    eccentricity: float
    true_anomaly: float
    is_at_periapsis: bool
    is_at_apoapsis: bool


# =============================================================================
# Core CR3BP Dynamics (5-State with Mass)
# =============================================================================

@jax.jit
def equations_of_motion_with_mass(
    state: jnp.ndarray,
    thrust_mag: float,
    isp_normalized: float,
    mu: float = MU
) -> jnp.ndarray:
    """
    CR3BP Equations of Motion with mass depletion.
    
    State vector: [x, y, vx, vy, m]
    
    Dynamics:
        ẋ = vx
        ẏ = vy
        v̇x = 2*vy + x - (1-μ)(x+μ)/r1³ - μ(x-(1-μ))/r2³ + ax_thrust
        v̇y = -2*vx + y - (1-μ)y/r1³ - μy/r2³ + ay_thrust
        ṁ = -T / (Isp * g0)  [only when thrusting]
    
    Args:
        state: [x, y, vx, vy, m] - position, velocity, mass (normalized)
        thrust_mag: Thrust magnitude (normalized force, not acceleration)
        isp_normalized: Specific impulse in normalized units
        mu: Mass ratio (default: Earth-Moon)
        
    Returns:
        dstate_dt: [vx, vy, ax, ay, mdot]
    """
    x, y, vx, vy, m = state
    
    # Distances to primaries with softening
    r1_sq = (x + mu)**2 + y**2
    r2_sq = (x - (1 - mu))**2 + y**2
    r1_cubed = (r1_sq + EPSILON)**1.5
    r2_cubed = (r2_sq + EPSILON)**1.5
    
    # Gravitational accelerations
    ax_grav = 2 * vy + x - (1 - mu) * (x + mu) / r1_cubed - mu * (x - (1 - mu)) / r2_cubed
    ay_grav = -2 * vx + y - (1 - mu) * y / r1_cubed - mu * y / r2_cubed
    
    # Thrust acceleration (tangential - velocity-aligned)
    # a = T / m (force divided by current mass)
    v_mag = jnp.sqrt(vx**2 + vy**2 + 1e-10)
    thrust_accel = thrust_mag / (m + 1e-10)  # Avoid division by zero
    ax_thrust = thrust_accel * vx / v_mag
    ay_thrust = thrust_accel * vy / v_mag
    
    # Mass flow rate (only when thrusting)
    # ṁ = -|T| / (Isp * g0)
    # thrust_mag is normalized force; isp_normalized and G0_NORM are consistent
    mdot = jnp.where(
        jnp.abs(thrust_mag) > 0,
        -jnp.abs(thrust_mag) / (isp_normalized * G0_NORM + 1e-10),
        0.0
    )
    
    return jnp.array([vx, vy, ax_grav + ax_thrust, ay_grav + ay_thrust, mdot])


@jax.jit  
def equations_of_motion_4state(
    state: jnp.ndarray,
    thrust_accel: float,
    mu: float = MU
) -> jnp.ndarray:
    """
    Original 4-state CR3BP dynamics (for backwards compatibility).
    
    State vector: [x, y, vx, vy]
    thrust_accel: Acceleration magnitude (not force)
    
    Returns:
        dstate_dt: [vx, vy, ax, ay]
    """
    x, y, vx, vy = state
    
    # Distances to primaries with softening
    r1_sq = (x + mu)**2 + y**2
    r2_sq = (x - (1 - mu))**2 + y**2
    r1_cubed = (r1_sq + EPSILON)**1.5
    r2_cubed = (r2_sq + EPSILON)**1.5
    
    # Tangential thrust direction
    v_mag = jnp.sqrt(vx**2 + vy**2 + 1e-10)
    ux = vx / v_mag
    uy = vy / v_mag
    
    # Accelerations
    ax = 2 * vy + x - (1 - mu) * (x + mu) / r1_cubed - mu * (x - (1 - mu)) / r2_cubed + thrust_accel * ux
    ay = -2 * vx + y - (1 - mu) * y / r1_cubed - mu * y / r2_cubed + thrust_accel * uy
    
    return jnp.array([vx, vy, ax, ay])


# =============================================================================
# RK4 Integration
# =============================================================================

@jax.jit
def rk4_step_with_mass(
    state: jnp.ndarray,
    dt: float,
    thrust_mag: float,
    isp_normalized: float
) -> jnp.ndarray:
    """Single RK4 step for 5-state dynamics."""
    k1 = equations_of_motion_with_mass(state, thrust_mag, isp_normalized)
    k2 = equations_of_motion_with_mass(state + 0.5 * dt * k1, thrust_mag, isp_normalized)
    k3 = equations_of_motion_with_mass(state + 0.5 * dt * k2, thrust_mag, isp_normalized)
    k4 = equations_of_motion_with_mass(state + dt * k3, thrust_mag, isp_normalized)
    
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


@jax.jit
def rk4_step_4state(
    state: jnp.ndarray,
    dt: float,
    thrust_accel: float
) -> jnp.ndarray:
    """Single RK4 step for 4-state dynamics (backwards compatible)."""
    k1 = equations_of_motion_4state(state, thrust_accel)
    k2 = equations_of_motion_4state(state + 0.5 * dt * k1, thrust_accel)
    k3 = equations_of_motion_4state(state + 0.5 * dt * k2, thrust_accel)
    k4 = equations_of_motion_4state(state + dt * k3, thrust_accel)
    
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


# =============================================================================
# Trajectory Propagation
# =============================================================================

@partial(jax.jit, static_argnames=['num_steps'])
def propagate_trajectory_with_mass(
    initial_state: jnp.ndarray,
    thrust_schedule: jnp.ndarray,
    thrust_magnitude: float,
    isp_normalized: float,
    dt: float,
    num_steps: int
) -> jnp.ndarray:
    """
    Propagates a trajectory with mass depletion.
    
    Args:
        initial_state: [x, y, vx, vy, m] - initial state (5 elements)
        thrust_schedule: (num_steps,) binary array [0 or 1]
        thrust_magnitude: Maximum thrust (normalized force)
        isp_normalized: Specific impulse (normalized)
        dt: Time step (normalized)
        num_steps: Number of steps
        
    Returns:
        trajectory: (num_steps + 1, 5) array of states
    """
    def step_fn(carry, i):
        state = carry
        
        # Check termination condition (distance > 1.5 * Earth-Moon distance)
        # Position is relative to barycenter, but close enough to Earth for this check
        # Earth is at (-mu, 0), so |pos| approx distance from Earth
        dist_from_center = jnp.linalg.norm(state[:2])
        is_escaped = dist_from_center > 1.5
        
        thrust_on = thrust_schedule[i]
        current_thrust = thrust_on * thrust_magnitude
        next_state_computed = rk4_step_with_mass(state, dt, current_thrust, isp_normalized)
        
        # If escaped, freeze state (effectively stopping simulation for this particle)
        next_state = jnp.where(is_escaped, state, next_state_computed)
        
        return next_state, state
    
    final_state, trajectory_history = jax.lax.scan(
        step_fn, initial_state, jnp.arange(num_steps)
    )
    
    # Append final state
    full_trajectory = jnp.vstack([trajectory_history, final_state[None, :]])
    return full_trajectory


@partial(jax.jit, static_argnames=['num_steps'])
def propagate_trajectory_4state(
    initial_state: jnp.ndarray,
    thrust_schedule: jnp.ndarray,
    dt: float,
    num_steps: int
) -> jnp.ndarray:
    """
    Propagates a trajectory (4-state, backwards compatible).
    
    Args:
        initial_state: [x, y, vx, vy]
        thrust_schedule: (num_steps,) thrust magnitudes (acceleration)
        dt: Time step
        num_steps: Number of steps
        
    Returns:
        trajectory: (num_steps + 1, 4) array of states
    """
    def step_fn(carry, i):
        state = carry
        
        # Check termination condition
        dist_from_center = jnp.linalg.norm(state[:2])
        is_escaped = dist_from_center > 1.5
        
        thrust_accel = thrust_schedule[i]
        next_state_computed = rk4_step_4state(state, dt, thrust_accel)
        
        next_state = jnp.where(is_escaped, state, next_state_computed)
        
        return next_state, state
    
    final_state, trajectory_history = jax.lax.scan(
        step_fn, initial_state, jnp.arange(num_steps)
    )
    
    full_trajectory = jnp.vstack([trajectory_history, final_state[None, :]])
    return full_trajectory


# Batch propagation (vmap over schedule dimension)
batch_propagate_with_mass = jax.vmap(
    propagate_trajectory_with_mass,
    in_axes=(None, 0, None, None, None, None)
)

batch_propagate_4state = jax.vmap(
    propagate_trajectory_4state,
    in_axes=(None, 0, None, None)
)

# Alias for backwards compatibility
batch_propagate = batch_propagate_4state


# =============================================================================
# Orbital Mechanics Utilities
# =============================================================================

@jax.jit
def compute_distance_to_body(position: jnp.ndarray, body: str = "moon") -> float:
    """Compute distance from position to Earth or Moon."""
    if body == "moon":
        return jnp.linalg.norm(position - MOON_POS)
    else:
        return jnp.linalg.norm(position - EARTH_POS)


@jax.jit
def compute_relative_velocity(state: jnp.ndarray, body: str = "moon") -> float:
    """
    Compute velocity magnitude relative to a body.
    
    Note: In the rotating frame, both primaries are stationary,
    so this is just the velocity magnitude at the spacecraft.
    For Moon approach, what matters is the velocity in the Moon-centered frame.
    """
    vx, vy = state[2], state[3]
    return jnp.sqrt(vx**2 + vy**2)


def detect_periapsis_apoapsis(trajectory: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Detect periapsis and apoapsis points in a trajectory (Earth-centered).
    
    Args:
        trajectory: (N, 4 or 5) state history
        
    Returns:
        periapsis_mask: (N,) boolean array, True at periapsis
        apoapsis_mask: (N,) boolean array, True at apoapsis
    """
    # Compute distances to Earth
    positions = trajectory[:, :2]
    distances = jnp.linalg.norm(positions - EARTH_POS, axis=1)
    
    # Find local minima (periapsis) and maxima (apoapsis)
    # Compare each point to its neighbors
    n = len(distances)
    if n < 3:
        return jnp.zeros(n, dtype=bool), jnp.zeros(n, dtype=bool)
    
    # Pad for boundary handling
    d_padded = jnp.pad(distances, (1, 1), mode='edge')
    
    is_minimum = (d_padded[1:-1] < d_padded[:-2]) & (d_padded[1:-1] < d_padded[2:])
    is_maximum = (d_padded[1:-1] > d_padded[:-2]) & (d_padded[1:-1] > d_padded[2:])
    
    return is_minimum, is_maximum


def compute_osculating_elements(state: jnp.ndarray, mu_earth: float = 1 - MU) -> OrbitalElements:
    """
    Compute osculating orbital elements (Earth-centered 2-body approximation).
    
    This is an approximation valid near Earth, used for physics-aware biasing.
    
    Args:
        state: [x, y, vx, vy, (m)] - state vector
        mu_earth: Gravitational parameter of Earth (normalized)
        
    Returns:
        OrbitalElements named tuple
    """
    # Position and velocity relative to Earth
    r = state[:2] - EARTH_POS
    v = state[2:4]
    
    r_mag = jnp.linalg.norm(r)
    v_mag = jnp.linalg.norm(v)
    
    # Specific orbital energy: ε = v²/2 - μ/r
    energy = 0.5 * v_mag**2 - mu_earth / r_mag
    
    # Semi-major axis: a = -μ / (2ε)
    a = -mu_earth / (2 * energy + 1e-10)
    
    # Angular momentum (z-component in 2D)
    h = r[0] * v[1] - r[1] * v[0]
    
    # Eccentricity vector
    e_vec = ((v_mag**2 - mu_earth/r_mag) * r - jnp.dot(r, v) * v) / mu_earth
    e = jnp.linalg.norm(e_vec)
    
    # True anomaly
    cos_nu = jnp.dot(r, e_vec) / (r_mag * e + 1e-10)
    nu = jnp.arccos(jnp.clip(cos_nu, -1, 1))
    
    # Adjust for quadrant based on r·v sign
    if jnp.dot(r, v) < 0:
        nu = 2 * jnp.pi - nu
    
    # Periapsis/apoapsis detection (within 5 degrees)
    is_periapsis = jnp.abs(nu) < 0.1  # Near 0
    is_apoapsis = jnp.abs(nu - jnp.pi) < 0.1  # Near π
    
    return OrbitalElements(
        semi_major_axis=float(a),
        eccentricity=float(e),
        true_anomaly=float(nu),
        is_at_periapsis=bool(is_periapsis),
        is_at_apoapsis=bool(is_apoapsis)
    )


# =============================================================================
# Fuel Budget and Constraints
# =============================================================================

def check_fuel_budget(
    schedule: jnp.ndarray,
    thrust_magnitude: float,
    isp_normalized: float,
    initial_mass: float,
    min_final_mass_fraction: float = 0.3
) -> bool:
    """
    Check if a thrust schedule respects the fuel budget.
    
    Args:
        schedule: (N,) binary thrust schedule
        thrust_magnitude: Thrust magnitude (normalized force)
        isp_normalized: Specific impulse (normalized)
        initial_mass: Initial spacecraft mass (normalized)
        min_final_mass_fraction: Minimum allowed final mass as fraction of initial
        
    Returns:
        True if schedule respects budget, False otherwise
    """
    total_burn_steps = jnp.sum(schedule)
    # dt is implicit in the schedule (each step = one dt)
    # Mass flow per step: dm = T * dt / (Isp * g0)
    # For normalized units with dt embedded, this simplifies
    # We need to know dt to compute this properly
    
    # For now, use total impulse constraint
    # This is a soft approximation - real check needs dt
    thrust_fraction = total_burn_steps / len(schedule)
    
    # Rough estimate: if thrusting > 70% of time, likely exceeds budget
    return thrust_fraction <= (1 - min_final_mass_fraction)


def compute_fuel_consumed(
    schedule: jnp.ndarray,
    thrust_magnitude: float,
    isp_normalized: float,
    dt: float
) -> float:
    """
    Compute total fuel consumed for a given schedule.
    
    Args:
        schedule: (N,) binary thrust schedule
        thrust_magnitude: Thrust magnitude (normalized force)
        isp_normalized: Specific impulse (normalized)
        dt: Time step (normalized)
        
    Returns:
        Total mass consumed (normalized)
    """
    total_burn_time = jnp.sum(schedule) * dt
    mass_flow_rate = thrust_magnitude / (isp_normalized * G0_NORM)
    return mass_flow_rate * total_burn_time


# =============================================================================
# Cost Functions
# =============================================================================

@jax.jit
def compute_trajectory_cost(
    trajectory: jnp.ndarray,
    weight_distance: float = 1.0,
    weight_velocity: float = 0.1,
    weight_fuel: float = 0.0
) -> float:
    """
    Compute weighted cost for a trajectory.
    
    Cost = w_d * dist_to_moon + w_v * arrival_velocity + w_f * fuel_used
    
    Args:
        trajectory: (N, 4 or 5) state history
        weight_distance: Weight for distance to Moon
        weight_velocity: Weight for arrival velocity
        weight_fuel: Weight for fuel consumption
        
    Returns:
        Scalar cost (lower is better)
    """
    final_state = trajectory[-1]
    
    # Distance to Moon
    dist_moon = jnp.linalg.norm(final_state[:2] - MOON_POS)
    
    # Velocity magnitude
    vel_mag = jnp.sqrt(final_state[2]**2 + final_state[3]**2)
    
    # Fuel used (if mass state available)
    if trajectory.shape[1] >= 5:
        initial_mass = trajectory[0, 4]
        final_mass = trajectory[-1, 4]
        fuel_used = initial_mass - final_mass
    else:
        fuel_used = 0.0
    
    # Minimum distance to Moon (Collision Check)
    all_dists = jnp.linalg.norm(trajectory[:, :2] - MOON_POS, axis=1)
    min_dist_moon = jnp.min(all_dists)
    
    # Collision Penalty
    # If min_dist < R_MOON_NORM (surface impact), add massive penalty
    # R_MOON_NORM is approx 0.0045 (~1737 km)
    collision_penalty = jnp.where(min_dist_moon < R_MOON_NORM, 1e6, 0.0)
    
    return weight_distance * dist_moon + weight_velocity * vel_mag + weight_fuel * fuel_used + collision_penalty


batch_compute_cost = jax.vmap(compute_trajectory_cost, in_axes=(0, None, None, None))


# =============================================================================
# Validation Utilities
# =============================================================================

@jax.jit
def compute_jacobi_constant(state: jnp.ndarray, mu: float = MU) -> float:
    """
    Compute the Jacobi integral (conserved in CR3BP without thrust).
    
    C = -(vx² + vy²) + 2*Ω(x,y)
    
    where Ω = (1/2)(x² + y²) + (1-μ)/r1 + μ/r2
    """
    x, y, vx, vy = state[:4]
    
    r1 = jnp.sqrt((x + mu)**2 + y**2 + EPSILON)
    r2 = jnp.sqrt((x - (1 - mu))**2 + y**2 + EPSILON)
    
    omega = 0.5 * (x**2 + y**2) + (1 - mu) / r1 + mu / r2
    v_sq = vx**2 + vy**2
    
    return -v_sq + 2 * omega


def validate_trajectory(
    trajectory: jnp.ndarray,
    thrust_schedule: jnp.ndarray,
    max_jacobi_drift: float = 0.01,
    max_distance: float = 5.0  # Normalized units (5 * L* = 1.9 million km)
) -> dict:
    """
    Validate a propagated trajectory.
    
    Checks:
    - Jacobi constant conservation during coast phases
    - Trajectory doesn't diverge to infinity
    - Final mass is positive (if applicable)
    
    Returns:
        Dictionary with validation results
    """
    # Find coast phases
    coast_mask = thrust_schedule < 0.5
    
    # Compute Jacobi constant at each point
    jacobi = jax.vmap(compute_jacobi_constant)(trajectory)
    
    # Check conservation during coast (compare adjacent coast points)
    coast_indices = jnp.where(coast_mask)[0]
    if len(coast_indices) > 1:
        jacobi_coast = jacobi[coast_indices]
        jacobi_drift = jnp.abs(jacobi_coast[1:] - jacobi_coast[:-1]).max()
    else:
        jacobi_drift = 0.0
    
    # Check divergence
    max_dist = jnp.linalg.norm(trajectory[:, :2], axis=1).max()
    
    # Check mass (if available)
    if trajectory.shape[1] >= 5:
        final_mass = trajectory[-1, 4]
        mass_valid = final_mass > 0
    else:
        final_mass = None
        mass_valid = True
    
    return {
        'jacobi_drift': float(jacobi_drift),
        'jacobi_valid': jacobi_drift < max_jacobi_drift,
        'max_distance': float(max_dist),
        'distance_valid': max_dist < max_distance,
        'final_mass': float(final_mass) if final_mass is not None else None,
        'mass_valid': mass_valid,
        'is_valid': (jacobi_drift < max_jacobi_drift) and (max_dist < max_distance) and mass_valid
    }


# =============================================================================
# Initial State Generation
# =============================================================================

def get_parking_orbit_state(
    altitude_km: float = 200.0,
    inclination: float = 0.0  # Not used in 2D, for future 3D extension
) -> jnp.ndarray:
    """
    Generate initial state for a circular parking orbit around Earth.
    
    Args:
        altitude_km: Altitude above Earth's surface (km)
        inclination: Orbit inclination (degrees) - placeholder for 3D
        
    Returns:
        state: [x, y, vx, vy, m_normalized] initial state
        Mass is set to 1.0 (normalized) - should be overwritten by user config
    """
    from .constants import R_EARTH_KM, L_STAR_KM
    
    # Orbital radius
    r_km = R_EARTH_KM + altitude_km
    r_norm = r_km / L_STAR_KM
    
    # Position (start on x-axis, right of Earth)
    x = -MU + r_norm
    y = 0.0
    
    # Circular velocity in rotating frame
    # v_inertial = sqrt(GM/r), v_rotating = v_inertial - ω × r
    # In normalized units, GM_Earth ≈ 1-μ, ω = 1
    v_circ = jnp.sqrt((1 - MU) / r_norm)
    
    # In rotating frame: subtract frame rotation at position x
    # vy_rotating = vy_inertial - omega * x (since ω = 1 and rotation is about z)
    # Note: x = -MU + r_norm, NOT just r_norm!
    vx = 0.0
    vy = v_circ - x  # FIXED: use x (the actual position), not r_norm
    
    # Mass (normalized to 1.0)
    m = 1.0
    
    return jnp.array([x, y, vx, vy, m])


def get_initial_state_4d(altitude_km: float = 200.0) -> jnp.ndarray:
    """4-state version for backwards compatibility."""
    state_5d = get_parking_orbit_state(altitude_km)
    return state_5d[:4]


# =============================================================================
# Dimensionalization
# =============================================================================

def dimensionalize_trajectory(
    trajectory: jnp.ndarray,
    L_star: float = L_STAR_KM * 1000,  # meters
    V_star: float = None
) -> jnp.ndarray:
    """
    Convert normalized trajectory to metric units.
    
    Args:
        trajectory: (N, 4 or 5) normalized states
        L_star: Length unit (default: meters)
        V_star: Velocity unit (default: computed from L_star and T_star)
        
    Returns:
        trajectory in metric units (m, m/s, kg)
    """
    if V_star is None:
        V_star = L_star / T_STAR_S
    
    result = trajectory.copy()
    result = result.at[:, :2].set(trajectory[:, :2] * L_star)
    result = result.at[:, 2:4].set(trajectory[:, 2:4] * V_star)
    # Mass remains unchanged (user should denormalize separately if needed)
    
    return result


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Core dynamics
    'equations_of_motion_with_mass',
    'equations_of_motion_4state',
    'rk4_step_with_mass',
    'rk4_step_4state',
    
    # Propagation
    'propagate_trajectory_with_mass',
    'propagate_trajectory_4state',
    'batch_propagate_with_mass',
    'batch_propagate_4state',
    'batch_propagate',  # Alias
    
    # Orbital mechanics
    'compute_distance_to_body',
    'compute_relative_velocity',
    'detect_periapsis_apoapsis',
    'compute_osculating_elements',
    
    # Constraints
    'check_fuel_budget',
    'compute_fuel_consumed',
    
    # Cost functions
    'compute_trajectory_cost',
    'batch_compute_cost',
    
    # Validation
    'compute_jacobi_constant',
    'validate_trajectory',
    
    # Utilities
    'get_parking_orbit_state',
    'get_initial_state_4d',
    'dimensionalize_trajectory',
    
    # Data structures
    'TrajectoryResult',
    'OrbitalElements',
]
