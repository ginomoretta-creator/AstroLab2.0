"""
THRML Graph Structure
=====================

Defines the Heterogeneous Graph structure for the THRML engine.
This replaces the simple 1D Ising chain with a graph containing specialized
node types for time steps, fuel constraints, and maneuver constraints.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import uuid

@dataclass
class Node:
    """Base class for all graph nodes."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    value: float = 0.0  # State of the node (e.g., spin, activation)
    bias: float = 0.0   # Local field / bias (h_i)
    
    def __hash__(self):
        return hash(self.id)

@dataclass
class TimeStepNode(Node):
    """
    Represents a specific time step in the trajectory.
    Value represents Thrust ON/OFF (or magnitude).
    """
    time_index: int = 0
    dt: float = 1.0
    
@dataclass
class FuelConstraintNode(Node):
    """
    Represents the global fuel budget.
    Connected to ALL TimeStepNodes (or a large subset).
    "Heats up" (increases cost) as total thrust increases.
    """
    max_fuel: float = 100.0
    current_usage: float = 0.0
    penalty_strength: float = 1.0

@dataclass
class ManeuverConstraintNode(Node):
    """
    Represents a specific maneuver requirement (e.g., "Must thrust at perigee").
    Connected to a subset of TimeStepNodes.
    """
    target_indices: List[int] = field(default_factory=list)
    type: str = "PERIGEE_KICK" # or "ORBIT_INSERTION", etc.

@dataclass
class Edge:
    """Connection between two nodes with a weight (coupling strength J_ij)."""
    source: Node
    target: Node
    weight: float = 1.0

class HeterogeneousGraph:
    """
    The main container for the thermodynamic graph.
    Manages nodes, edges, and structure.
    """
    def __init__(self):
        self.timestep_nodes: List[TimeStepNode] = []
        self.constraint_nodes: List[Node] = [] # Mix of Fuel and Maneuver nodes
        self.edges: List[Edge] = []
        self._node_map: Dict[str, Node] = {}
        
    def add_timestep_node(self, time_index: int, dt: float = 1.0, bias: float = 0.0) -> TimeStepNode:
        node = TimeStepNode(time_index=time_index, dt=dt, bias=bias)
        self.timestep_nodes.append(node)
        self._node_map[node.id] = node
        return node
        
    def add_fuel_node(self, max_fuel: float, penalty_strength: float = 1.0) -> FuelConstraintNode:
        node = FuelConstraintNode(max_fuel=max_fuel, penalty_strength=penalty_strength)
        self.constraint_nodes.append(node)
        self._node_map[node.id] = node
        
        # Automatically connect to existing time steps?
        # Typically yes, a global fuel constraint connects to everyone.
        # But we might want to do this explicitly in a 'build' step.
        return node
        
    def add_edge(self, node1: Node, node2: Node, weight: float):
        edge = Edge(source=node1, target=node2, weight=weight)
        self.edges.append(edge)
        
    @property
    def all_nodes(self) -> List[Node]:
        return self.timestep_nodes + self.constraint_nodes

    def get_node_by_id(self, node_id: str) -> Optional[Node]:
        return self._node_map.get(node_id)
