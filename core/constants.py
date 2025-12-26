"""
CR3BP Constants for Earth-Moon System
=====================================

All values are exact or high-precision. This module serves as the single
source of truth for physical constants across both THRML-Sandbox and QNTM-Sandbox.

Reference: JPL DE440/441 ephemerides, IAU 2015 resolutions
"""
import jax.numpy as jnp
import numpy as np

# =============================================================================
# Mass Parameters
# =============================================================================

# Mass ratio μ = M_Moon / (M_Earth + M_Moon)
# From JPL Horizons, consistent with DE440
MU = 0.01215058560962404

# Individual masses (kg) - for reference
M_EARTH_KG = 5.972168e24
M_MOON_KG = 7.34767309e22
M_TOTAL_KG = M_EARTH_KG + M_MOON_KG

# =============================================================================
# Characteristic Quantities (CR3BP Normalization)
# =============================================================================

# Length: Earth-Moon mean distance (km)
L_STAR_KM = 384400.0
L_STAR_M = L_STAR_KM * 1000.0

# Time: Chosen such that mean motion n = 1
# T* = sqrt(L*³ / G(M_E + M_M)) ≈ 375,200 s ≈ 4.34 days
T_STAR_S = 375190.258918  # More precise value
T_STAR_DAYS = T_STAR_S / 86400.0

# Derived quantities
V_STAR_KMS = L_STAR_KM / T_STAR_S  # ~1.024 km/s
V_STAR_MS = L_STAR_M / T_STAR_S    # m/s
A_STAR_MS2 = V_STAR_MS / T_STAR_S  # Characteristic acceleration (m/s²)
A_STAR_KMS2 = V_STAR_KMS / T_STAR_S  # km/s²

# =============================================================================
# Body Properties
# =============================================================================

# Radii (km)
R_EARTH_KM = 6378.137  # Equatorial radius (WGS84)
R_MOON_KM = 1737.4     # Mean radius

# Normalized radii
R_EARTH_NORM = R_EARTH_KM / L_STAR_KM
R_MOON_NORM = R_MOON_KM / L_STAR_KM

# Body positions in rotating frame (JAX arrays for GPU computation)
EARTH_POS = jnp.array([-MU, 0.0])
MOON_POS = jnp.array([1.0 - MU, 0.0])

# NumPy versions for CasADi compatibility
EARTH_POS_NP = np.array([-MU, 0.0])
MOON_POS_NP = np.array([1.0 - MU, 0.0])

# =============================================================================
# Physical Constants
# =============================================================================

# Standard gravity (m/s²)
G0_MS2 = 9.80665

# Normalized gravity (for Isp calculations in normalized units)
G0_NORM = G0_MS2 / A_STAR_MS2

# Gravitational constant (m³/kg/s²)
G_CONST = 6.67430e-11

# =============================================================================
# Numerical Parameters
# =============================================================================

# Softening parameter to avoid singularities in gravity calculation
EPSILON = 1e-6

# Default integration tolerances
RTOL_DEFAULT = 1e-8
ATOL_DEFAULT = 1e-10

# =============================================================================
# Mission-Relevant Defaults
# =============================================================================

# Typical LEO parking orbit
LEO_ALTITUDE_KM = 200.0
LEO_RADIUS_KM = R_EARTH_KM + LEO_ALTITUDE_KM
LEO_RADIUS_NORM = LEO_RADIUS_KM / L_STAR_KM

# Lunar capture region (approximate)
LUNAR_SOI_KM = 66100.0  # Hill sphere radius
LUNAR_SOI_NORM = LUNAR_SOI_KM / L_STAR_KM

# Typical low-thrust engine parameters
DEFAULT_ISP_S = 3000.0  # Hall thruster typical
DEFAULT_THRUST_N = 0.5  # 500 mN class

# =============================================================================
# Utility Functions
# =============================================================================

def normalize_length(value_km):
    """Convert km to normalized length."""
    return value_km / L_STAR_KM

def denormalize_length(value_norm):
    """Convert normalized length to km."""
    return value_norm * L_STAR_KM

def normalize_velocity(value_kms):
    """Convert km/s to normalized velocity."""
    return value_kms / V_STAR_KMS

def denormalize_velocity(value_norm):
    """Convert normalized velocity to km/s."""
    return value_norm * V_STAR_KMS

def normalize_time(value_s):
    """Convert seconds to normalized time."""
    return value_s / T_STAR_S

def denormalize_time(value_norm):
    """Convert normalized time to seconds."""
    return value_norm * T_STAR_S

def normalize_acceleration(value_ms2):
    """Convert m/s² to normalized acceleration."""
    return value_ms2 / A_STAR_MS2

def denormalize_acceleration(value_norm):
    """Convert normalized acceleration to m/s²."""
    return value_norm * A_STAR_MS2

def normalize_thrust(thrust_n, mass_kg):
    """
    Convert thrust (N) and mass (kg) to normalized acceleration.
    
    a_norm = (T/m) / A*
    """
    accel_ms2 = thrust_n / mass_kg
    return accel_ms2 / A_STAR_MS2

def compute_fuel_mass_rate(thrust_n, isp_s):
    """
    Compute mass flow rate (kg/s) for given thrust and Isp.
    
    ṁ = T / (Isp * g0)
    """
    return thrust_n / (isp_s * G0_MS2)

def compute_delta_v(m0_kg, mf_kg, isp_s):
    """
    Compute delta-v using Tsiolkovsky equation.
    
    Δv = Isp * g0 * ln(m0/mf)
    """
    return isp_s * G0_MS2 * np.log(m0_kg / mf_kg)

# =============================================================================
# Validation
# =============================================================================

if __name__ == "__main__":
    print("CR3BP Constants for Earth-Moon System")
    print("=" * 50)
    print(f"Mass ratio μ:           {MU:.14f}")
    print(f"L* (km):                {L_STAR_KM:.1f}")
    print(f"T* (s):                 {T_STAR_S:.2f}")
    print(f"T* (days):              {T_STAR_DAYS:.4f}")
    print(f"V* (km/s):              {V_STAR_KMS:.6f}")
    print(f"A* (m/s²):              {A_STAR_MS2:.6e}")
    print(f"Earth radius (norm):    {R_EARTH_NORM:.6f}")
    print(f"Moon radius (norm):     {R_MOON_NORM:.6f}")
    print(f"Lunar SOI (norm):       {LUNAR_SOI_NORM:.6f}")
    print(f"g0 (normalized):        {G0_NORM:.4f}")
