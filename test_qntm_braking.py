
import unittest
import numpy as np
from backend.engines.qntm.annealer import QuantumAnnealer, simple_energy_function
from backend.engines.qntm.state import WaveFunction
from backend.engines.qntm.solver import SimulatedQuantumAnnealer

class TestQNTMBraking(unittest.TestCase):
    def test_energy_function_absolute(self):
        """Verify energy function penalizes absolute thrust."""
        # Config with negative thrust (braking)
        config_brake = [-1.0, -1.0, -1.0, 0.0]
        # Config with positive thrust
        config_thrust = [1.0, 1.0, 1.0, 0.0]
        # Config with zero thrust
        config_coast = [0.0, 0.0, 0.0, 0.0]
        
        # Budget = 0.0 -> Penalty for any thrust
        E_brake = simple_energy_function(config_brake, budget=0.0)
        E_thrust = simple_energy_function(config_thrust, budget=0.0)
        E_coast = simple_energy_function(config_coast, budget=0.0)
        
        # braking and thrusting should have same penalty if budget is 0
        # (Assuming smoothness is zero or identical)
        # Smoothness: Brake: 0, 0, 1 -> 1. Thrust: 0, 0, -1 -> 1. Identical.
        self.assertAlmostEqual(E_brake, E_thrust)
        self.assertGreater(E_brake, E_coast)
        
    def test_annealer_generates_negatives(self):
        """Verify annealer can reach negative states."""
        wf = WaveFunction(population_size=5, num_steps=10)
        # Force negative bias
        bias_field = [-10.0] * 10 
        # Energy = ... - (bias * s). If bias is -10, we want s = -1 to get +10 term?
        # distinct: simple_energy_function says: "alignment = sum(b*s); energy -= alignment"
        # So E -= (-10 * s).
        # To minimize E, we want (-10 * s) to be MAX POSITIVE.
        # -10 * (-1) = 10. -10 * (1) = -10.
        # So s = -1 makes (-10 * s) = 10, E -= 10 (Good).
        # s = 1 makes (-10 * s) = -10, E -= -10 = +10 (Bad).
        # So negative bias should encourage negative state (Braking).
        
        annealer = QuantumAnnealer(wf, cooling_rate=0.8, bias_field=bias_field)
        
        # Run
        for _ in annealer.run(steps=50):
            pass
            
        # Check if we have negative values
        best_state = min(wf.states, key=lambda s: s.energy)
        min_val = min(best_state.configuration)
        print(f"Min value in best state: {min_val}")
        self.assertLess(min_val, -0.5)

    def test_solver_ternary_output(self):
        """Verify solver produces -1, 0, 1."""
        solver = SimulatedQuantumAnnealer()
        # Use simple inputs
        result = solver.generate_thrust_schedules(
            num_steps=10,
            batch_size=5,
            coupling_strength=0.1,
            bias=-10.0 # Should produce -1
        )
        schedules = result['schedules']
        unique = np.unique(schedules)
        print(f"Solver output unique values: {unique}")
        
        # Should contain -1 due to strong negative bias
        self.assertTrue(np.any(unique == -1.0))
        
if __name__ == '__main__':
    unittest.main()
