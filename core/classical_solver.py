"""
Classical Direct Collocation Solver for Trajectory Optimization
================================================================

This module provides a classical optimal control solver using CasADi
and IPOPT for refining discrete thrust schedules into dynamically
feasible trajectories.

The warm-starting workflow:
1. Binary schedule → Smooth profile (moving average)
2. Forward propagation → Initial state guess
3. Direct collocation with IPOPT → Locally optimal trajectory

Key Features:
- CR3BP dynamics with mass depletion
- Fuel-optimal objective (maximize final mass)
- Flexible constraints (lunar capture, velocity bounds)
- Warm-start iteration counting for benchmarks

Author: ASL-Sandbox Team
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

# Import constants
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, os.path.dirname(current_dir))

from core.constants import (
    MU, EPSILON, G0_NORM, A_STAR_MS2, T_STAR_S, L_STAR_KM, L_STAR_M,
    EARTH_POS_NP, MOON_POS_NP, R_MOON_NORM, LUNAR_SOI_NORM
)

# Try to import CasADi
try:
    import casadi as ca
    CASADI_AVAILABLE = True
except ImportError:
    CASADI_AVAILABLE = False
    print("WARNING: CasADi not available. Classical solver disabled.")
    print("Install with: pip install casadi")


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class SolverResult:
    """Result from classical trajectory optimization."""
    success: bool
    trajectory: Optional[np.ndarray]  # (N+1, 5) states
    control: Optional[np.ndarray]     # (N,) thrust magnitudes
    final_mass: Optional[float]
    delta_v: Optional[float]
    iterations: int
    solve_time_seconds: float
    message: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'success': self.success,
            'trajectory': self.trajectory.tolist() if self.trajectory is not None else None,
            'control': self.control.tolist() if self.control is not None else None,
            'final_mass': self.final_mass,
            'delta_v': self.delta_v,
            'iterations': self.iterations,
            'solve_time_seconds': self.solve_time_seconds,
            'message': self.message
        }


# =============================================================================
# Schedule Smoothing
# =============================================================================

def smooth_schedule(
    binary_schedule: np.ndarray,
    window_size: int = 5,
    method: str = 'gaussian'
) -> np.ndarray:
    """
    Convert binary thrust schedule to smooth continuous control.
    
    This is necessary because IPOPT requires continuous controls,
    but our generative models produce discrete {0, 1} schedules.
    
    Args:
        binary_schedule: (N,) array of 0s and 1s
        window_size: Smoothing window size
        method: 'moving_average' or 'gaussian'
        
    Returns:
        smoothed: (N,) array of continuous values in [0, 1]
    """
    if method == 'gaussian':
        # Gaussian kernel
        sigma = window_size / 4.0
        x = np.linspace(-window_size//2, window_size//2, window_size)
        kernel = np.exp(-x**2 / (2 * sigma**2))
        kernel = kernel / kernel.sum()
    else:
        # Moving average
        kernel = np.ones(window_size) / window_size
    
    # Convolve (pad to handle edges)
    padded = np.pad(binary_schedule, (window_size//2, window_size//2), mode='edge')
    smoothed = np.convolve(padded, kernel, mode='valid')
    
    # Ensure correct length
    if len(smoothed) > len(binary_schedule):
        smoothed = smoothed[:len(binary_schedule)]
    elif len(smoothed) < len(binary_schedule):
        smoothed = np.pad(smoothed, (0, len(binary_schedule) - len(smoothed)), mode='edge')
    
    return np.clip(smoothed, 0, 1)


# =============================================================================
# Forward Propagation (NumPy for initial guess)
# =============================================================================

def propagate_trajectory_numpy(
    x0: np.ndarray,
    thrust_profile: np.ndarray,
    thrust_max: float,
    isp_normalized: float,
    dt: float
) -> np.ndarray:
    """
    Forward propagate trajectory using RK4 (NumPy implementation).
    
    Used to generate initial state guess for IPOPT.
    
    Args:
        x0: Initial state [x, y, vx, vy, m]
        thrust_profile: (N,) thrust fraction [0, 1]
        thrust_max: Maximum thrust magnitude (normalized force)
        isp_normalized: Specific impulse (normalized)
        dt: Time step
        
    Returns:
        trajectory: (N+1, 5) state history
    """
    N = len(thrust_profile)
    trajectory = np.zeros((N + 1, 5))
    trajectory[0] = x0
    
    state = x0.copy()
    
    for i in range(N):
        thrust_mag = thrust_profile[i] * thrust_max
        state = _rk4_step_numpy(state, thrust_mag, isp_normalized, dt)
        trajectory[i + 1] = state
    
    return trajectory


def _rk4_step_numpy(state: np.ndarray, thrust_mag: float, isp_normalized: float, dt: float) -> np.ndarray:
    """Single RK4 step (NumPy)."""
    k1 = _dynamics_numpy(state, thrust_mag, isp_normalized)
    k2 = _dynamics_numpy(state + 0.5 * dt * k1, thrust_mag, isp_normalized)
    k3 = _dynamics_numpy(state + 0.5 * dt * k2, thrust_mag, isp_normalized)
    k4 = _dynamics_numpy(state + dt * k3, thrust_mag, isp_normalized)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def _dynamics_numpy(state: np.ndarray, thrust_mag: float, isp_normalized: float) -> np.ndarray:
    """CR3BP dynamics with mass (NumPy)."""
    x, y, vx, vy, m = state
    
    # Distances
    r1 = np.sqrt((x + MU)**2 + y**2 + EPSILON)
    r2 = np.sqrt((x - (1 - MU))**2 + y**2 + EPSILON)
    
    # Gravity
    ax_grav = 2*vy + x - (1-MU)*(x+MU)/r1**3 - MU*(x-(1-MU))/r2**3
    ay_grav = -2*vx + y - (1-MU)*y/r1**3 - MU*y/r2**3
    
    # Thrust (tangential)
    v_mag = np.sqrt(vx**2 + vy**2 + 1e-10)
    thrust_accel = thrust_mag / (m + 1e-10)
    ax_thrust = thrust_accel * vx / v_mag
    ay_thrust = thrust_accel * vy / v_mag
    
    # Mass flow
    if thrust_mag > 0:
        mdot = -thrust_mag / (isp_normalized * G0_NORM + 1e-10)
    else:
        mdot = 0.0
    
    return np.array([vx, vy, ax_grav + ax_thrust, ay_grav + ay_thrust, mdot])


# =============================================================================
# CasADi Direct Collocation Solver
# =============================================================================

def create_collocation_problem(
    x0: np.ndarray,
    thrust_initial_guess: np.ndarray,
    num_nodes: int,
    T_total: float,
    thrust_max: float,
    isp_normalized: float,
    lunar_capture_radius: float = 0.05,  # ~19,000 km
    min_final_mass_fraction: float = 0.2,
    max_velocity: Optional[float] = None
) -> Optional[SolverResult]:
    """
    Create and solve a direct collocation optimal control problem.
    
    Objective: Maximize final mass (minimize fuel consumption)
    
    Constraints:
    - CR3BP dynamics with mass depletion
    - Initial state fixed
    - Final state within lunar capture region
    - Thrust bounds [0, thrust_max]
    - Minimum final mass
    
    Args:
        x0: Initial state [x, y, vx, vy, m]
        thrust_initial_guess: (N,) initial thrust profile [0, 1]
        num_nodes: Number of collocation nodes
        T_total: Total transfer time (normalized)
        thrust_max: Maximum thrust (normalized force)
        isp_normalized: Specific impulse (normalized)
        lunar_capture_radius: Lunar capture constraint radius
        min_final_mass_fraction: Minimum final mass as fraction of initial
        max_velocity: Optional velocity constraint at Moon
        
    Returns:
        SolverResult with trajectory, control, and solve statistics
    """
    if not CASADI_AVAILABLE:
        return SolverResult(
            success=False, trajectory=None, control=None,
            final_mass=None, delta_v=None, iterations=0,
            solve_time_seconds=0.0, message="CasADi not available"
        )
    
    import time
    start_time = time.time()
    
    try:
        opti = ca.Opti()
        
        N = num_nodes
        dt = T_total / N
        
        # Decision variables
        X = opti.variable(5, N+1)  # States: [x, y, vx, vy, m]
        U = opti.variable(1, N)     # Controls: thrust magnitude [0, thrust_max]
        
        # Extract state components for readability
        x = X[0, :]
        y = X[1, :]
        vx = X[2, :]
        vy = X[3, :]
        m = X[4, :]
        
        # Dynamics function (symbolic)
        def dynamics_casadi(X_k, U_k):
            x_k, y_k, vx_k, vy_k, m_k = X_k[0], X_k[1], X_k[2], X_k[3], X_k[4]
            thrust_k = U_k[0]
            
            # Distances with softening
            r1 = ca.sqrt((x_k + MU)**2 + y_k**2 + EPSILON)
            r2 = ca.sqrt((x_k - (1 - MU))**2 + y_k**2 + EPSILON)
            
            # Gravity
            ax_grav = 2*vy_k + x_k - (1-MU)*(x_k+MU)/r1**3 - MU*(x_k-(1-MU))/r2**3
            ay_grav = -2*vx_k + y_k - (1-MU)*y_k/r1**3 - MU*y_k/r2**3
            
            # Thrust (tangential)
            v_mag = ca.sqrt(vx_k**2 + vy_k**2 + 1e-8)
            thrust_accel = thrust_k / (m_k + 1e-8)
            ax_thrust = thrust_accel * vx_k / v_mag
            ay_thrust = thrust_accel * vy_k / v_mag
            
            # Mass flow
            mdot = -thrust_k / (isp_normalized * G0_NORM + 1e-8)
            
            return ca.vertcat(vx_k, vy_k, ax_grav + ax_thrust, ay_grav + ay_thrust, mdot)
        
        # Collocation constraints (Hermite-Simpson or RK4)
        for k in range(N):
            X_k = X[:, k]
            U_k = U[:, k]
            X_next = X[:, k+1]
            
            # RK4 integration
            k1 = dynamics_casadi(X_k, U_k)
            k2 = dynamics_casadi(X_k + dt/2 * k1, U_k)
            k3 = dynamics_casadi(X_k + dt/2 * k2, U_k)
            k4 = dynamics_casadi(X_k + dt * k3, U_k)
            X_predicted = X_k + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
            
            opti.subject_to(X_next == X_predicted)
        
        # Initial state constraint
        opti.subject_to(X[:, 0] == x0)
        
        # Final state: Lunar capture (within capture radius of Moon)
        moon_x, moon_y = MOON_POS_NP[0], MOON_POS_NP[1]
        final_dist_sq = (x[-1] - moon_x)**2 + (y[-1] - moon_y)**2
        opti.subject_to(final_dist_sq <= lunar_capture_radius**2)
        
        # Velocity constraint at Moon (optional)
        if max_velocity is not None:
            final_vel_sq = vx[-1]**2 + vy[-1]**2
            opti.subject_to(final_vel_sq <= max_velocity**2)
        
        # Control bounds
        opti.subject_to(opti.bounded(0, U, thrust_max))
        
        # Mass bounds
        opti.subject_to(m >= x0[4] * min_final_mass_fraction)
        opti.subject_to(m <= x0[4])  # Mass can only decrease
        
        # Objective: Maximize final mass (minimize fuel)
        opti.minimize(-m[-1])
        
        # Initial guess
        smooth_thrust = smooth_schedule(thrust_initial_guess, 5) * thrust_max
        opti.set_initial(U, smooth_thrust.reshape(1, -1))
        
        # State guess from forward propagation
        x_guess = propagate_trajectory_numpy(
            x0, smooth_thrust / thrust_max, thrust_max, isp_normalized, dt
        )
        opti.set_initial(X, x_guess.T)
        
        # Solver options
        p_opts = {'expand': True}
        s_opts = {
            'max_iter': 1000,
            'tol': 1e-6,
            'print_level': 0,
            'sb': 'yes'  # Suppress banner
        }
        opti.solver('ipopt', p_opts, s_opts)
        
        # Solve
        sol = opti.solve()
        
        # Extract solution
        trajectory = sol.value(X).T
        control = sol.value(U).flatten()
        final_mass = float(sol.value(m[-1]))
        
        # Compute delta-v
        delta_v = isp_normalized * G0_NORM * np.log(x0[4] / final_mass)
        
        # Get iteration count
        stats = opti.stats()
        iterations = stats.get('iter_count', 0)
        
        solve_time = time.time() - start_time
        
        return SolverResult(
            success=True,
            trajectory=trajectory,
            control=control,
            final_mass=final_mass,
            delta_v=delta_v,
            iterations=iterations,
            solve_time_seconds=solve_time,
            message="Optimal solution found"
        )
        
    except Exception as e:
        solve_time = time.time() - start_time
        
        # Try to get iteration count even on failure
        try:
            stats = opti.stats()
            iterations = stats.get('iter_count', 0)
        except:
            iterations = 0
        
        return SolverResult(
            success=False,
            trajectory=None,
            control=None,
            final_mass=None,
            delta_v=None,
            iterations=iterations,
            solve_time_seconds=solve_time,
            message=str(e)
        )


# =============================================================================
# Warm-Start Comparison
# =============================================================================

def measure_warmstart_benefit(
    x0: np.ndarray,
    structured_schedule: np.ndarray,
    random_schedule: np.ndarray,
    num_nodes: int,
    T_total: float,
    thrust_max: float,
    isp_normalized: float,
    **solver_kwargs
) -> Dict[str, SolverResult]:
    """
    Compare solver performance with structured vs random initialization.
    
    Args:
        x0: Initial state
        structured_schedule: (N,) schedule from THRML/Quantum
        random_schedule: (N,) random baseline schedule
        num_nodes: Collocation nodes
        T_total: Transfer time
        thrust_max: Max thrust
        isp_normalized: Isp
        **solver_kwargs: Additional solver parameters
        
    Returns:
        Dictionary mapping method name to SolverResult
    """
    results = {}
    
    # 1. Structured (THRML or Quantum)
    results['structured'] = create_collocation_problem(
        x0, structured_schedule, num_nodes, T_total, 
        thrust_max, isp_normalized, **solver_kwargs
    )
    
    # 2. Random baseline
    results['random'] = create_collocation_problem(
        x0, random_schedule, num_nodes, T_total,
        thrust_max, isp_normalized, **solver_kwargs
    )
    
    # 3. Cold start (zeros)
    cold_schedule = np.zeros_like(structured_schedule)
    results['cold_start'] = create_collocation_problem(
        x0, cold_schedule, num_nodes, T_total,
        thrust_max, isp_normalized, **solver_kwargs
    )
    
    # 4. All-thrust start
    hot_schedule = np.ones_like(structured_schedule)
    results['all_thrust'] = create_collocation_problem(
        x0, hot_schedule, num_nodes, T_total,
        thrust_max, isp_normalized, **solver_kwargs
    )
    
    return results


def compute_iteration_statistics(results: Dict[str, SolverResult]) -> Dict[str, Any]:
    """
    Compute comparative statistics from warm-start experiments.
    
    Returns:
        Statistics dictionary for publication
    """
    stats = {}
    
    for name, result in results.items():
        stats[name] = {
            'success': result.success,
            'iterations': result.iterations,
            'solve_time': result.solve_time_seconds,
            'final_mass': result.final_mass,
            'delta_v': result.delta_v
        }
    
    # Compute improvement ratios
    if results['structured'].success and results['random'].success:
        stats['iteration_improvement'] = (
            results['random'].iterations / max(1, results['structured'].iterations)
        )
        stats['time_improvement'] = (
            results['random'].solve_time_seconds / max(0.001, results['structured'].solve_time_seconds)
        )
    else:
        stats['iteration_improvement'] = None
        stats['time_improvement'] = None
    
    # Success rate summary
    stats['summary'] = {
        'methods_tested': len(results),
        'successful': sum(1 for r in results.values() if r.success),
        'mean_iterations': np.mean([r.iterations for r in results.values()]),
        'mean_solve_time': np.mean([r.solve_time_seconds for r in results.values()])
    }
    
    return stats


# =============================================================================
# Simplified Solver (for quick tests)
# =============================================================================

def quick_refine(
    binary_schedule: np.ndarray,
    x0: np.ndarray,
    thrust_max: float = 0.001,
    isp_normalized: float = 300.0,
    dt: float = 0.01,
    lunar_capture_radius: float = 0.1
) -> Optional[SolverResult]:
    """
    Quick trajectory refinement with default parameters.
    
    Args:
        binary_schedule: (N,) binary thrust schedule
        x0: Initial state [x, y, vx, vy, m]
        thrust_max: Thrust magnitude
        isp_normalized: Specific impulse
        dt: Time step
        lunar_capture_radius: Capture constraint
        
    Returns:
        SolverResult
    """
    N = len(binary_schedule)
    T_total = N * dt
    
    return create_collocation_problem(
        x0=x0,
        thrust_initial_guess=binary_schedule,
        num_nodes=N,
        T_total=T_total,
        thrust_max=thrust_max,
        isp_normalized=isp_normalized,
        lunar_capture_radius=lunar_capture_radius
    )


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    'smooth_schedule',
    'propagate_trajectory_numpy',
    'create_collocation_problem',
    'measure_warmstart_benefit',
    'compute_iteration_statistics',
    'quick_refine',
    'SolverResult',
    'CASADI_AVAILABLE'
]


if __name__ == "__main__":
    print(f"CasADi available: {CASADI_AVAILABLE}")
    
    if CASADI_AVAILABLE:
        # Test smoothing
        binary = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 0])
        smooth = smooth_schedule(binary, 3)
        print(f"Binary: {binary}")
        print(f"Smooth: {smooth}")
        
        # Test forward propagation
        x0 = np.array([0.017, 0.0, 0.0, 1.5, 1.0])
        thrust_profile = np.ones(100) * 0.5
        traj = propagate_trajectory_numpy(x0, thrust_profile, 0.001, 300.0, 0.01)
        print(f"Propagated trajectory shape: {traj.shape}")
        print(f"Final position: {traj[-1, :2]}")
        
        # Test solver (if CasADi available)
        print("\nTesting direct collocation solver...")
        result = quick_refine(
            binary_schedule=np.random.randint(0, 2, 100).astype(float),
            x0=x0,
            thrust_max=0.001,
            lunar_capture_radius=0.2  # Relaxed for test
        )
        print(f"Solver success: {result.success}")
        print(f"Iterations: {result.iterations}")
        print(f"Message: {result.message}")
