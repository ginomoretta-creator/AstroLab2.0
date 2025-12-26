
import unittest
import jax
import jax.numpy as jnp
import numpy as np

from backend.engines.thrml.graph import HeterogeneousGraph, TimeStepNode
from backend.engines.thrml.sampler import GibbsSampler
from backend.engines.thrml.energy import compute_local_field_components, compute_fuel_penalty_cost

class TestBrakingLogic(unittest.TestCase):
    def test_sampler_ternary_output(self):
        """Verify that the Gibbs Sampler produces -1, 0, and 1 values."""
        graph = HeterogeneousGraph()
        # Add a node with 0 bias
        node = graph.add_timestep_node(0, dt=1.0, bias=0.0)
        
        sampler = GibbsSampler(graph, beta=1.0)
        
        # Run sampler for many steps and track values
        values = []
        for _ in range(100):
            sampler.step()
            values.append(node.value)
            
        # Check if we got mix of values (random check)
        unique_values = set(values)
        print(f"Unique values observed: {unique_values}")
        
        # We expect at least 0 and likely others depending on random seed, 
        # but structurally we want to ensure keys are within {-1.0, 0.0, 1.0}
        for v in unique_values:
            self.assertIn(v, {-1.0, 0.0, 1.0})
            
    def test_bias_influence(self):
        """Verify that strong bias forces specific states."""
        # 1. Strong Positive Bias -> Expected +1
        graph_pos = HeterogeneousGraph()
        node_pos = graph_pos.add_timestep_node(0, bias=100.0) # Huge bias
        sampler_pos = GibbsSampler(graph_pos, beta=1.0)
        sampler_pos.step()
        self.assertEqual(node_pos.value, 1.0)
        
        # 2. Strong Negative Bias -> Expected -1 (wait, bias was linear: E = -h*s)
        # If h = -100, then E(1) = 100, E(-1) = -100. So s should go to -1.
        graph_neg = HeterogeneousGraph()
        node_neg = graph_neg.add_timestep_node(0, bias=-100.0)
        sampler_neg = GibbsSampler(graph_neg, beta=1.0)
        sampler_neg.step()
        self.assertEqual(node_neg.value, -1.0)
        
if __name__ == '__main__':
    unittest.main()
