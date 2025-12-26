"""
Core module for physics-complete low-thrust trajectory optimization.
"""

from .constants import (
    MU, EPSILON,
    L_STAR_KM, L_STAR_M, T_STAR_S, V_STAR_KMS, V_STAR_MS, A_STAR_MS2,
    R_EARTH_KM, R_MOON_KM, R_EARTH_NORM, R_MOON_NORM,
    EARTH_POS, MOON_POS,
    G0_MS2, G0_NORM,
    LUNAR_SOI_NORM,
    normalize_length, denormalize_length,
    normalize_velocity, denormalize_velocity,
    normalize_time, denormalize_time,
    normalize_acceleration, denormalize_acceleration,
    normalize_thrust, compute_fuel_mass_rate, compute_delta_v
)

from .physics_core import (
    # Core dynamics
    equations_of_motion_with_mass,
    equations_of_motion_4state,
    rk4_step_with_mass,
    rk4_step_4state,
    
    # Propagation
    propagate_trajectory_with_mass,
    propagate_trajectory_4state,
    batch_propagate_with_mass,
    batch_propagate_4state,
    batch_propagate,
    
    # Orbital mechanics
    compute_distance_to_body,
    compute_relative_velocity,
    detect_periapsis_apoapsis,
    compute_osculating_elements,
    
    # Constraints
    check_fuel_budget,
    compute_fuel_consumed,
    
    # Cost functions
    compute_trajectory_cost,
    batch_compute_cost,
    
    # Validation
    compute_jacobi_constant,
    validate_trajectory,
    
    # Utilities
    get_parking_orbit_state,
    get_initial_state_4d,
    dimensionalize_trajectory,
    
    # Data structures
    TrajectoryResult,
    OrbitalElements,
)

from .energy_model import (
    compute_physics_bias_field,
    compute_reference_trajectory_for_bias,
    update_bias_from_elite_samples,
    filter_schedules_by_fuel_budget,
    repair_schedule_fuel_budget,
    compute_eclipse_windows,
    apply_eclipse_constraint,
    PhysicsGuidedScheduleGenerator,
)

# Optional imports (may require additional dependencies)
try:
    from .classical_solver import (
        smooth_schedule,
        propagate_trajectory_numpy,
        create_collocation_problem,
        measure_warmstart_benefit,
        compute_iteration_statistics,
        quick_refine,
        SolverResult,
        CASADI_AVAILABLE
    )
except ImportError:
    CASADI_AVAILABLE = False

try:
    from .benchmark_suite import (
        BenchmarkResult,
        BenchmarkConfig,
        ExplorationBenchmark,
        FullBenchmarkSuite,
        quick_benchmark
    )
    BENCHMARK_AVAILABLE = True
except ImportError:
    BENCHMARK_AVAILABLE = False

__all__ = [
    # Constants
    'MU', 'EPSILON',
    'L_STAR_KM', 'L_STAR_M', 'T_STAR_S', 'V_STAR_KMS', 'V_STAR_MS', 'A_STAR_MS2',
    'R_EARTH_KM', 'R_MOON_KM', 'R_EARTH_NORM', 'R_MOON_NORM',
    'EARTH_POS', 'MOON_POS',
    'G0_MS2', 'G0_NORM',
    'LUNAR_SOI_NORM',
    
    # Unit conversion
    'normalize_length', 'denormalize_length',
    'normalize_velocity', 'denormalize_velocity',
    'normalize_time', 'denormalize_time',
    'normalize_acceleration', 'denormalize_acceleration',
    'normalize_thrust', 'compute_fuel_mass_rate', 'compute_delta_v',
    
    # Physics
    'equations_of_motion_with_mass', 'equations_of_motion_4state',
    'rk4_step_with_mass', 'rk4_step_4state',
    'propagate_trajectory_with_mass', 'propagate_trajectory_4state',
    'batch_propagate_with_mass', 'batch_propagate_4state', 'batch_propagate',
    
    # Orbital mechanics
    'compute_distance_to_body', 'compute_relative_velocity',
    'detect_periapsis_apoapsis', 'compute_osculating_elements',
    
    # Constraints and cost
    'check_fuel_budget', 'compute_fuel_consumed',
    'compute_trajectory_cost', 'batch_compute_cost',
    
    # Validation
    'compute_jacobi_constant', 'validate_trajectory',
    
    # Utilities
    'get_parking_orbit_state', 'get_initial_state_4d', 'dimensionalize_trajectory',
    
    # Data structures
    'TrajectoryResult', 'OrbitalElements',
    
    # Energy model
    'compute_physics_bias_field', 'compute_reference_trajectory_for_bias',
    'update_bias_from_elite_samples', 'filter_schedules_by_fuel_budget',
    'repair_schedule_fuel_budget', 'compute_eclipse_windows',
    'apply_eclipse_constraint', 'PhysicsGuidedScheduleGenerator',
    
    # Availability flags
    'CASADI_AVAILABLE', 'BENCHMARK_AVAILABLE',
]

__version__ = "1.0.0"

