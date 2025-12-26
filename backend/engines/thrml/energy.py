"""
THRML Energy Logic
==================

Calculates the energy (Hamiltonian) of the Heterogeneous Graph.
Includes standard Ising interactions, local bias fields, and 
global constraint penalties (Fuel Node back-pressure).
"""

import jax
import jax.numpy as jnp
from typing import List, Dict, Any

from .graph import HeterogeneousGraph, TimeStepNode, FuelConstraintNode

def compute_total_energy(graph: HeterogeneousGraph) -> float:
    """
    Computes the total energy of the current graph configuration.
    E = E_interactions + E_bias + E_constraints
    """
    energy = 0.0
    
    # 1. Edge Interactions (Smoothness)
    # E_int = - sum (w_ij * s_i * s_j)
    # Assuming spin values are mapped to [-1, 1] or [0, 1]
    # Let's use 0/1 for "Thrust ON/OFF" usually, but Ising math works best with -1/1.
    # We will assume node.value is in [0, 1] for now, and convert if needed.
    # Actually, for standard Ising: E = -J * s_i * s_j
    
    for edge in graph.edges:
        s_i = edge.source.value
        s_j = edge.target.value
        w_ij = edge.weight
        
        # Smootheness usually means we want s_i == s_j.
        # If w_ij > 0 (ferromagnetic), lower energy when aligned.
        # If we use 0,1: 
        #   1,1 -> -w
        #   0,0 -> 0 (no benefit?) 
        # Better to map to spins: sigma = 2*s - 1
        
        sigma_i = 2 * s_i - 1
        sigma_j = 2 * s_j - 1
        
        energy -= w_ij * sigma_i * sigma_j

    # 2. Local Fields (Bias)
    # E_bias = - sum (h_i * s_i)
    for node in graph.timestep_nodes:
        # Bias usually acts on the '1' state (thrusting).
        # h_i > 0 encourages thrust.
        energy -= node.bias * node.value 
        
    # 3. Global Constraints (Fuel)
    # The FuelConstraintNode acts as a non-linear global term.
    # It "heats up" if usage > max_fuel.
    
    # Calculate total fuel usage (Based on ABSOLUTE thrust)
    # ternary: s in {-1, 0, 1}, so fuel usage is proportional to |s|
    total_thrust = sum(abs(n.value) * n.dt for n in graph.timestep_nodes)
    
    for node in graph.constraint_nodes:
        if isinstance(node, FuelConstraintNode):
            # Quadratic penalty if over budget
            # E_fuel = k * max(0, usage - limit)^2
            excess = max(0.0, total_thrust - node.max_fuel)
            
            # Simple soft constraint
            # Or "back pressure": a linear term that grows with total usage
            # But energy is a scalar.
            penalty = 0.5 * node.penalty_strength * (excess ** 2)
            energy += penalty
            
            # Update node state for visualization/introspection
            node.current_usage = total_thrust
            
    return energy

def compute_local_field_components(graph: HeterogeneousGraph, node: TimeStepNode) -> float:
    """
    Computes the linear component of the local field (bias + neighbor influence).
    E_linear(s) = -h_linear * s
    """
    field = node.bias
    
    # Neighbor influence
    # We need a way to quickly find neighbors. 
    # The current graph struct doesn't have an adjacency list.
    # For efficiency, we should add one, but for now we iterate (slow for large graphs, ok for prototype).
    # TODO: optimize adjacency
    
    # Warning: simple iteration over all edges is O(E).
    # For the prototype, we assume the graph is small or we will optimize later.
    
    for edge in graph.edges:
        neighbor = None
        if edge.source == node:
            neighbor = edge.target
        elif edge.target == node:
            neighbor = edge.source
            
        if neighbor:
            # Map neighbor 0/1 to -1/1 spin for interaction
            # If neighbor is already ternary (-1, 0, 1), we can use it directly
            # But wait, previous logic assumed 0->-1 map?
            # "sigma_neighbor = 2 * neighbor.value - 1" was for binary 0/1
            
            # For Ternary Logic, we assume the value IS the spin (-1, 0, 1).
            # So simple:
            sigma_neighbor = neighbor.value
            
            # Add to field (effective h)
            # Interaction term -J s_i s_j can be written as - (J s_j) s_i
            # So effective field adds J * s_j
            field += edge.weight * sigma_neighbor
            
    return field

def compute_fuel_penalty_cost(graph: HeterogeneousGraph, node: TimeStepNode) -> float:
    """
    Computes the marginal energy cost of turning ON thrust (magnitude 1),
    regardless of direction.
    Delta E_fuel = E(|s|=1) - E(|s|=0)
    """
    # Let's calculate the "pressure" from fuel nodes
    total_thrust = sum(abs(n.value) * n.dt for n in graph.timestep_nodes)
    
    total_penalty_gradient = 0.0

    for c_node in graph.constraint_nodes:
        if isinstance(c_node, FuelConstraintNode):
            # If we turn ON (add dt to usage), how much does energy increase?
            # E = 0.5 * k * (usage - limit)^2
            # dE = k * (usage - limit) * dt  (approx)
            
            # Note: We use the CURRENT usage state to estimate gradient.
            # If usage > limit, cost is positive.
            
            excess = max(0.0, total_thrust - c_node.max_fuel)
            pressure = c_node.penalty_strength * excess * node.dt
            
            total_penalty_gradient += pressure
            
    return total_penalty_gradient
