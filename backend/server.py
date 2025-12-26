import sys
import os
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Literal, Generator
import json
from fastapi.responses import StreamingResponse
import jax
import jax.numpy as jnp
import numpy as np

# Ensure backend is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Add project root for core imports
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import modules
from physics import batch_propagate, dimensionalize_trajectory
from engines.thrml.model import generate_thrust_schedules

# Try to import core for physics-aware sampling
try:
    from core import compute_physics_bias_field, get_initial_state_4d
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False
    print("Core module not available, using basic sampling")

app = FastAPI(title="Cislunar Trajectory Sandbox API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SimulationRequest(BaseModel):
    # Schedule/trajectory parameters
    num_steps: int = 120000  # More steps for longer trajectories (~250 days)
    batch_size: int = 100  # Increased for better exploration
    coupling_strength: float = 0.5
    
    # Physical Parameters
    mass: float = 500.0 # kg (smaller sat for realistic low-thrust)
    thrust: float = 0.5 # N (typical Hall thruster)
    isp: float = 3000.0 # s (Hall thruster Isp)
    initial_altitude: float = 200.0 # km (Altitude above Earth)
    
    # Method
    method: Literal["thrml", "quantum", "random"] = "thrml"
    
    # Advanced
    dt: float = 0.0005  # Very small steps for smooth trajectories (35000 * 0.0005 = 17.5 norm time = ~76 days)
    num_iterations: int = 30  # More iterations for convergence
    
    # Demo mode: scales thrust up for faster visualization during development
    demo_mode: bool = False  # If True, thrust is multiplied by 50x

# Constants for Normalization
L_STAR = 384400.0 * 1000 # meters (Earth-Moon Distance)
T_STAR = 375200.0 # seconds (approx 4.34 days)
M_STAR = 1000.0 # kg (Reference mass, arbitrary)
MU = 0.01215

@app.get("/")
def read_root():
    return {
        "status": "online", 
        "system": "ASL-Sandbox Backend",
        "core_available": CORE_AVAILABLE,
        "methods": ["thrml", "quantum", "random"]
    }

def get_initial_state(altitude_km):
    R_EARTH_KM = 6378.0
    L_MOON_KM = 384400.0
    
    r_norm = (R_EARTH_KM + altitude_km) / L_MOON_KM
    
    # Position in rotating frame (x-axis, right of Earth at origin)
    x = -MU + r_norm
    y = 0.0
    
    # Circular velocity in inertial frame: v_inertial = sqrt(GM_Earth / r)
    # In normalized units: GM_Earth = 1 - MU, so v_circ = sqrt((1-MU)/r)
    v_circ = np.sqrt((1 - MU) / r_norm)
    
    # Rotating frame correction: v_rotating = v_inertial - omega × r
    # omega = 1 in normalized units, cross product gives velocity in y-direction = omega * x
    # So vy_rotating = v_circ - x (NOT r_norm!)
    vx = 0.0
    vy = v_circ - x  # FIXED: use x, not r_norm
    
    return [x, y, vx, vy]

@app.post("/simulate")
def run_simulation(req: SimulationRequest):
    def simulation_generator() -> Generator[str, None, None]:
        try:
            key = jax.random.PRNGKey(int(np.random.randint(0, 100000)))
            
            # 1. Calculate Normalized Thrust Acceleration (clamped for stability)
            accel_metric = req.thrust / req.mass
            accel_norm = accel_metric * (T_STAR**2 / L_STAR)
            
            # Demo mode: scale up thrust by 50x for faster development iteration
            if req.demo_mode:
                accel_norm *= 50.0
            
            accel_norm = float(np.clip(accel_norm, 0.0, 0.5))  # Allow higher thrust in demo mode
            
            # 2. Determine Initial State (using core physics)
            if CORE_AVAILABLE:
                # Use core 5-state (x,y,vx,vy,m) or 4-state
                from core import get_initial_state_4d, compute_reference_trajectory_for_bias
                init_state_arr = jnp.array(get_initial_state_4d(req.initial_altitude))
            else:
                # Fallback to local
                init_state_arr = jnp.array(get_initial_state(req.initial_altitude))
            
            # 3. Initialize physics-aware bias if available
            if CORE_AVAILABLE:
                # Compute reference trajectory first!
                ref_traj = compute_reference_trajectory_for_bias(
                    req.num_steps, 
                    req.dt, 
                    accel_norm, 
                    init_state_arr
                )
                
                initial_bias = compute_physics_bias_field(
                    req.num_steps, 
                    ref_traj,  # Pass the reference!
                    0.4
                )
                current_bias = initial_bias
            else:
                current_bias = None
            
            # 4. Moon position for cost calculation
            moon_pos = jnp.array([1 - MU, 0])

            # 5. Success threshold (relaxed ~20% of Earth-Moon distance)
            success_threshold = 0.2
            fuel_min = 0.05
            fuel_max = 0.75
            # Clamp timestep for stability (normalized units) - allow larger dt for long simulations
            dt = float(np.clip(req.dt, 0.0005, 0.05))

            for i in range(req.num_iterations):
                # New randomness each iteration to avoid repeating schedules
                key, sample_key = jax.random.split(key)
                sample_key, sched_key = jax.random.split(sample_key)

                # Generate Schedules based on method
                if req.method == "thrml":
                    schedules = generate_thrust_schedules(
                        sample_key, 
                        req.num_steps, 
                        req.batch_size, 
                        req.coupling_strength, 
                        [], [],
                        bias_field=current_bias
                    )
                elif req.method == "quantum":
                    # Try to use quantum solver if available
                    try:
                        from engines.qntm.solver import SimulatedQuantumAnnealer
                        annealer = SimulatedQuantumAnnealer()
                        result = annealer.generate_thrust_schedules(
                            req.num_steps,
                            req.batch_size,
                            req.coupling_strength,
                            float(jnp.mean(current_bias)) if current_bias is not None else 0.0
                        )
                        schedules = jnp.array(result['schedules'])
                    except ImportError:
                        # Fallback: Use biased random with annealing-like structure
                        if current_bias is not None:
                            probs = jax.nn.sigmoid(current_bias)
                        else:
                            probs = jnp.ones(req.num_steps) * 0.4
                        schedules = jax.random.bernoulli(sample_key, probs, shape=(req.batch_size, req.num_steps)).astype(jnp.float32)
                else:  # random
                    schedules = jax.random.bernoulli(sample_key, p=0.4, shape=(req.batch_size, req.num_steps)).astype(jnp.float32)

                # Enforce reasonable fuel fractions by replacing outliers
                # Ternary logic: use abs(schedules) for fuel usage
                thrust_fractions = jnp.mean(jnp.abs(schedules), axis=1)
                valid_mask = (thrust_fractions >= fuel_min) & (thrust_fractions <= fuel_max)
                # Replacement must also be ternary or at least compliant. 
                # For simplicity, fallback replacement is binary (0/1) or we could sample ternary.
                replacement_schedules = jax.random.bernoulli(sched_key, p=0.4, shape=schedules.shape).astype(jnp.float32)
                schedules = jnp.where(valid_mask[:, None], schedules, replacement_schedules)
                thrust_fractions = jnp.mean(jnp.abs(schedules), axis=1)
                    
                # Propagate Physics
                thrust_schedules_mag = schedules * accel_norm
                trajectories = batch_propagate(init_state_arr, thrust_schedules_mag, dt, req.num_steps)

                # === IMPROVED MOON-SEEKING COST FUNCTION ===
                
                # 1. Final distance to Moon
                final_positions = trajectories[:, -1, :2]
                final_dist = jnp.linalg.norm(final_positions - moon_pos, axis=1)
                radii = jnp.linalg.norm(final_positions, axis=1)
                
                # 2. MINIMUM distance to Moon during entire trajectory (key improvement!)
                all_dists_to_moon = jnp.linalg.norm(trajectories[:, :, :2] - moon_pos, axis=2)
                min_dist_to_moon = jnp.min(all_dists_to_moon, axis=1)
                
                # 3. Approach progress: are we getting closer over time?
                initial_dist = jnp.linalg.norm(trajectories[:, 0, :2] - moon_pos, axis=1)
                approach_progress = initial_dist - final_dist  # positive = getting closer
                
                # 4. Apoapsis reward (keep this for orbit-raising behavior)
                earth_pos = jnp.array([-MU, 0])
                all_dists_from_earth = jnp.linalg.norm(trajectories[:, :, :2] - earth_pos, axis=2)
                max_apoapsis = jnp.max(all_dists_from_earth, axis=1)
                apoapsis_reward = jnp.clip(max_apoapsis, 0.0, 1.0) * 0.2

                # 5. Velocity Guidance (Prevent flybys)
                # Calculate velocity relative to Moon frame (already in rotating frame)
                velocities = trajectories[:, :, 2:4]
                vel_mags = jnp.linalg.norm(velocities, axis=2)
                
                # Identify where we are close to Moon (< 0.05 normalized ~ 19,000 km)
                # If close, penalize high velocity to encourage capture
                is_close = all_dists_to_moon < 0.05
                # Only penalize if close AND fast
                # Increased penalty weight to encourage braking
                velocity_cost = jnp.mean(jnp.where(is_close, vel_mags * 5.0, 0.0), axis=1)

                # Fuel Penalty (Soft Constraint)
                max_allowed = 0.6
                fuel_penalty = jnp.where(
                    thrust_fractions > max_allowed, 
                    10.0 * (thrust_fractions - max_allowed) * 384400.0,
                    0.0
                )
                budget_violation_penalty = jnp.where(
                    (thrust_fractions < fuel_min) | (thrust_fractions > fuel_max),
                    1e6,
                    0.0
                )
                # Radial penalty for escaping system
                radial_penalty = jnp.where(radii > 1.5, (radii - 1.5) * 5e3, 0.0)

                # Minimum distance to Moon (Collision Check)
                # R_MOON_NORM = 1737.4 / 384400.0 ~= 0.0045
                # Using 0.005 (~1922 km) as a safe collision buffer
                collision_radius = 0.005
                collision_penalty = jnp.where(min_dist_to_moon < collision_radius, 1e6, 0.0)

                # === COMBINED COST ===
                # - 50% Min distance (approach)
                # - 30% Final distance
                # - 20% Approach progress
                # - 10% Velocity control (new!)
                # - Bonus: Apoapsis reward
                total_cost = (
                    min_dist_to_moon * 0.5 +
                    final_dist * 0.3 +
                    - approach_progress * 0.2 +
                    velocity_cost * 1.5 +          # Increased weight for capture
                    - apoapsis_reward +
                    fuel_penalty + budget_violation_penalty + radial_penalty + collision_penalty
                )

                # Calculate success rate (based on minimum distance to Moon)
                successes = min_dist_to_moon < success_threshold
                success_rate = float(jnp.mean(successes))
                
                # Select Best (Top 10%) based on Total Cost
                k_best = max(1, int(req.batch_size * 0.1))
                best_indices = jnp.argsort(total_cost)[:k_best]
                best_schedules = schedules[best_indices]
                
                # Update Bias for next iteration (Cross-Entropy Method style)
                avg_schedule = jnp.mean(best_schedules, axis=0)
                learning_rate = 0.3
                if current_bias is not None:
                    current_bias = current_bias * (1 - learning_rate) + (avg_schedule - 0.5) * 4.0 * learning_rate
                else:
                    current_bias = (avg_schedule - 0.5) * 4.0
                
                # Update Key
                # Prepare response data
                traj_np = np.array(trajectories)
                min_dists_np = np.array(min_dist_to_moon)  # Use min distance to Moon for sorting
                schedules_np = np.array(schedules)
                total_cost_np = np.array(total_cost)

                # True best based on total cost (respects penalties)
                best_idx = int(np.argmin(total_cost_np))
                best_schedule = schedules_np[best_idx]
                best_thrust_fraction = float(np.mean(np.abs(best_schedule)))
                best_cost = float(total_cost_np[best_idx])
                best_distance = float(np.linalg.norm(traj_np[best_idx, -1, :2] - np.array([1 - MU, 0])))

                # Get indices to send: Top 5 by min distance + Random 5
                sorted_indices = np.argsort(min_dists_np)
                top_indices = sorted_indices[:5]
                random_indices = np.random.choice(req.batch_size, 5, replace=False)
                display_indices = np.unique(np.concatenate([top_indices, random_indices]))

                chunk_trajectories = traj_np[display_indices]
                best_trajectory = traj_np[best_idx]

                chunk_data = {
                    "iteration": i + 1,
                    "total_iterations": req.num_iterations,
                    "trajectories": chunk_trajectories.tolist(),
                    "best_cost": best_cost,
                    "mean_cost": float(np.mean(total_cost_np)),
                    "best_trajectory": best_trajectory.tolist(),
                    "success_rate": success_rate,
                    "method": req.method,
                    "best_schedule": best_schedule.tolist(),
                    "best_thrust_fraction": best_thrust_fraction,
                    "best_distance": best_distance,
                    "radial_penalty": float(np.mean(radial_penalty)),
                }
                
                yield json.dumps(chunk_data) + "\n"
                
        except Exception as e:
            import traceback
            yield json.dumps({
                "error": str(e),
                "traceback": traceback.format_exc()
            }) + "\n"

    return StreamingResponse(simulation_generator(), media_type="application/x-ndjson")

@app.post("/benchmark")
def run_benchmark(methods: List[str], num_samples: int = 100, num_steps: int = 200):
    """Run comparative benchmark across methods."""
    results = {}
    
    for method in methods:
        if method not in ["thrml", "quantum", "random"]:
            continue
            
        # Create request
        req = SimulationRequest(
            method=method,
            num_steps=num_steps,
            batch_size=num_samples,
            num_iterations=1,
        )
        
        # Run single iteration for benchmark
        key = jax.random.PRNGKey(42)
        accel_metric = req.thrust / req.mass
        accel_norm = accel_metric * (T_STAR**2 / L_STAR)
        init_state_arr = jnp.array(get_initial_state(req.initial_altitude))
        
        if method == "thrml":
            schedules = generate_thrust_schedules(
                key, num_steps, num_samples, req.coupling_strength, [], []
            )
        elif method == "quantum":
            probs = jnp.ones(num_steps) * 0.4
            schedules = jax.random.bernoulli(key, probs, shape=(num_samples, num_steps)).astype(jnp.float32)
        else:
            schedules = jax.random.bernoulli(key, p=0.4, shape=(num_samples, num_steps)).astype(jnp.float32)
        
        thrust_schedules_mag = schedules * accel_norm
        trajectories = batch_propagate(init_state_arr, thrust_schedules_mag, req.dt, num_steps)
        
        moon_pos = jnp.array([1 - MU, 0])
        final_positions = trajectories[:, -1, :2]
        dists = jnp.linalg.norm(final_positions - moon_pos, axis=1)
        
        results[method] = {
            "mean_distance": float(jnp.mean(dists)),
            "min_distance": float(jnp.min(dists)),
            "std_distance": float(jnp.std(dists)),
            "success_rate": float(jnp.mean(dists < 0.13)),
        }
    
    return results

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
