"""
Benchmarking Suite for Trajectory Warm-Start Comparison
========================================================

This module provides comprehensive benchmarking tools for comparing
different schedule generation methods:

1. THRML (Probabilistic Gibbs Sampling)
2. Quantum-Inspired (Simulated Annealing)
3. Random Baseline (Bernoulli Sampling)

Benchmarks:
- Exploration Efficiency: How often do methods reach the Moon?
- Solver Convergence: How many IPOPT iterations to optimal?
- Computational Cost: Wall-clock time comparison

Author: ASL-Sandbox Team
"""

import sys
import os
import time
import json
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path

# JAX imports
import jax
import jax.numpy as jnp

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, os.path.dirname(current_dir))

# Core imports
from core.constants import MU, MOON_POS, L_STAR_KM
from core.physics_core import (
    propagate_trajectory_4state,
    batch_propagate_4state,
    get_initial_state_4d,
    compute_trajectory_cost
)
from core.energy_model import compute_physics_bias_field


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    method: str
    n_samples: int
    success_rate: float
    mean_distance: float
    std_distance: float
    min_distance: float
    mean_cost: float
    std_cost: float
    wall_time_seconds: float
    
    # Optional solver metrics
    mean_iterations: Optional[float] = None
    std_iterations: Optional[float] = None
    mean_solve_time: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'method': self.method,
            'n_samples': self.n_samples,
            'success_rate': self.success_rate,
            'mean_distance': self.mean_distance,
            'std_distance': self.std_distance,
            'min_distance': self.min_distance,
            'mean_cost': self.mean_cost,
            'std_cost': self.std_cost,
            'wall_time_seconds': self.wall_time_seconds,
            'mean_iterations': self.mean_iterations,
            'std_iterations': self.std_iterations,
            'mean_solve_time': self.mean_solve_time
        }


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    num_steps: int = 500
    batch_size: int = 100
    n_trials: int = 10
    coupling_strength: float = 1.0
    fuel_budget_fraction: float = 0.4
    thrust_accel: float = 0.001
    dt: float = 0.01
    success_threshold_normalized: float = 0.13  # ~50,000 km
    initial_altitude_km: float = 200.0


# =============================================================================
# Generators Registry
# =============================================================================

class GeneratorRegistry:
    """Registry for different schedule generation methods."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self._generators: Dict[str, Callable] = {}
        self._register_defaults()
    
    def _register_defaults(self):
        """Register default generators."""
        
        # Random generator (Bernoulli)
        def random_generator(key, batch_size):
            return jax.random.bernoulli(
                key, 
                self.config.fuel_budget_fraction, 
                (batch_size, self.config.num_steps)
            ).astype(jnp.float32)
        
        self._generators['random'] = random_generator
        
        # Biased random (with physics-aware probability)
        def biased_random_generator(key, batch_size):
            initial_state = get_initial_state_4d(self.config.initial_altitude_km)
            bias_field = compute_physics_bias_field(
                self.config.num_steps,
                None,  # No reference trajectory for simple test
                self.config.fuel_budget_fraction
            )
            probs = jax.nn.sigmoid(bias_field)
            return jax.random.bernoulli(key, probs, (batch_size, self.config.num_steps)).astype(jnp.float32)
        
        self._generators['biased_random'] = biased_random_generator
    
    def register(self, name: str, generator: Callable):
        """Register a custom generator."""
        self._generators[name] = generator
    
    def get(self, name: str) -> Callable:
        """Get a generator by name."""
        return self._generators.get(name)
    
    def list_methods(self) -> List[str]:
        """List available methods."""
        return list(self._generators.keys())


# =============================================================================
# Exploration Benchmark
# =============================================================================

class ExplorationBenchmark:
    """
    Benchmark: How well do methods explore the solution space?
    
    Metrics:
    - Success rate (% reaching Moon vicinity)
    - Distance distribution
    - Cost distribution
    """
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.registry = GeneratorRegistry(config)
        
        # Precompute initial state
        self.initial_state = get_initial_state_4d(config.initial_altitude_km)
    
    def add_generator(self, name: str, generator: Callable):
        """Add a custom generator."""
        self.registry.register(name, generator)
    
    def run_single_method(
        self,
        method: str,
        key: jax.random.PRNGKey,
        n_samples: int
    ) -> BenchmarkResult:
        """
        Run exploration benchmark for a single method.
        
        Args:
            method: Generator method name
            key: JAX random key
            n_samples: Number of schedules to generate
            
        Returns:
            BenchmarkResult with exploration metrics
        """
        generator = self.registry.get(method)
        if generator is None:
            raise ValueError(f"Unknown method: {method}")
        
        start_time = time.time()
        
        # Generate schedules
        schedules = generator(key, n_samples)
        
        # Scale to thrust magnitude
        thrust_schedules = schedules * self.config.thrust_accel
        
        # Propagate trajectories
        trajectories = batch_propagate_4state(
            self.initial_state[:4],
            thrust_schedules,
            self.config.dt,
            self.config.num_steps
        )
        
        # Compute distances to Moon
        final_positions = trajectories[:, -1, :2]
        distances = jnp.linalg.norm(final_positions - MOON_POS, axis=1)
        
        # Compute costs
        costs = jax.vmap(lambda t: compute_trajectory_cost(t, 1.0, 0.1, 0.0))(trajectories)
        
        wall_time = time.time() - start_time
        
        # Success rate
        successes = distances < self.config.success_threshold_normalized
        success_rate = float(jnp.mean(successes))
        
        return BenchmarkResult(
            method=method,
            n_samples=n_samples,
            success_rate=success_rate,
            mean_distance=float(jnp.mean(distances)),
            std_distance=float(jnp.std(distances)),
            min_distance=float(jnp.min(distances)),
            mean_cost=float(jnp.mean(costs)),
            std_cost=float(jnp.std(costs)),
            wall_time_seconds=wall_time
        )
    
    def run_comparison(
        self,
        methods: List[str],
        n_samples: int = 1000,
        seed: int = 42
    ) -> Dict[str, BenchmarkResult]:
        """
        Run exploration benchmark comparing multiple methods.
        
        Args:
            methods: List of method names to compare
            n_samples: Samples per method
            seed: Random seed
            
        Returns:
            Dictionary mapping method name to BenchmarkResult
        """
        key = jax.random.PRNGKey(seed)
        results = {}
        
        for method in methods:
            key, subkey = jax.random.split(key)
            results[method] = self.run_single_method(method, subkey, n_samples)
            print(f"  {method}: success_rate={results[method].success_rate:.3f}, "
                  f"min_dist={results[method].min_distance:.4f}")
        
        return results


# =============================================================================
# Statistical Tests
# =============================================================================

def mann_whitney_test(
    distances_a: np.ndarray,
    distances_b: np.ndarray,
    alternative: str = 'less'
) -> Dict[str, float]:
    """
    Perform Mann-Whitney U test for distance comparison.
    
    Args:
        distances_a: Distances from method A
        distances_b: Distances from method B
        alternative: 'less' (A < B), 'greater', or 'two-sided'
        
    Returns:
        Dictionary with U statistic and p-value
    """
    try:
        from scipy import stats
        statistic, p_value = stats.mannwhitneyu(
            distances_a, distances_b, alternative=alternative
        )
        return {'U_statistic': statistic, 'p_value': p_value}
    except ImportError:
        # Basic approximation if scipy not available
        n1, n2 = len(distances_a), len(distances_b)
        mean_a, mean_b = np.mean(distances_a), np.mean(distances_b)
        return {
            'U_statistic': None,
            'p_value': None,
            'mean_difference': mean_a - mean_b
        }


def compute_effect_size(
    values_a: np.ndarray,
    values_b: np.ndarray
) -> float:
    """
    Compute Cohen's d effect size.
    
    Args:
        values_a: Values from method A
        values_b: Values from method B
        
    Returns:
        Cohen's d (positive if A < B)
    """
    mean_a, mean_b = np.mean(values_a), np.mean(values_b)
    std_pooled = np.sqrt((np.var(values_a) + np.var(values_b)) / 2)
    return (mean_b - mean_a) / (std_pooled + 1e-8)


# =============================================================================
# Benchmark Report Generator
# =============================================================================

class BenchmarkReporter:
    """Generates benchmark reports in various formats."""
    
    def __init__(self, results: Dict[str, BenchmarkResult]):
        self.results = results
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary."""
        return {
            'timestamp': datetime.now().isoformat(),
            'methods': {name: r.to_dict() for name, r in self.results.items()},
            'summary': self._compute_summary()
        }
    
    def _compute_summary(self) -> Dict[str, Any]:
        """Compute summary statistics."""
        methods = list(self.results.keys())
        
        # Find best method by success rate
        best_by_success = max(methods, key=lambda m: self.results[m].success_rate)
        
        # Find best by minimum distance
        best_by_distance = min(methods, key=lambda m: self.results[m].min_distance)
        
        return {
            'n_methods': len(methods),
            'best_success_rate': {
                'method': best_by_success,
                'value': self.results[best_by_success].success_rate
            },
            'best_min_distance': {
                'method': best_by_distance,
                'value': self.results[best_by_distance].min_distance
            }
        }
    
    def to_json(self, path: str):
        """Save results to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = [
            "# Benchmark Results",
            f"\n**Generated**: {datetime.now().isoformat()}",
            "\n## Method Comparison\n",
            "| Method | Success Rate | Mean Distance | Min Distance | Wall Time (s) |",
            "|--------|--------------|---------------|--------------|---------------|"
        ]
        
        for name, r in self.results.items():
            lines.append(
                f"| {name} | {r.success_rate:.3f} | {r.mean_distance:.4f} | "
                f"{r.min_distance:.4f} | {r.wall_time_seconds:.2f} |"
            )
        
        # Add summary
        summary = self._compute_summary()
        lines.extend([
            "\n## Summary\n",
            f"- **Best Success Rate**: {summary['best_success_rate']['method']} "
            f"({summary['best_success_rate']['value']:.1%})",
            f"- **Best Min Distance**: {summary['best_min_distance']['method']} "
            f"({summary['best_min_distance']['value']:.4f} L*)"
        ])
        
        return "\n".join(lines)


# =============================================================================
# Full Benchmark Suite
# =============================================================================

class FullBenchmarkSuite:
    """
    Complete benchmark suite for publication-ready comparisons.
    """
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.exploration = ExplorationBenchmark(config)
        self.results: Dict[str, Any] = {}
    
    def add_thrml_generator(self, generator: Callable):
        """Add THRML generator."""
        self.exploration.add_generator('thrml', generator)
    
    def add_quantum_generator(self, generator: Callable):
        """Add quantum generator."""
        self.exploration.add_generator('quantum', generator)
    
    def run_exploration_benchmark(
        self,
        methods: Optional[List[str]] = None,
        n_samples: int = 1000,
        seed: int = 42
    ) -> Dict[str, BenchmarkResult]:
        """Run exploration benchmark."""
        if methods is None:
            methods = self.exploration.registry.list_methods()
        
        print(f"Running exploration benchmark with {n_samples} samples...")
        results = self.exploration.run_comparison(methods, n_samples, seed)
        self.results['exploration'] = results
        return results
    
    def run_statistical_tests(self) -> Dict[str, Any]:
        """Run statistical tests on exploration results."""
        if 'exploration' not in self.results:
            raise ValueError("Run exploration benchmark first")
        
        exploration = self.results['exploration']
        tests = {}
        
        # Compare each method against random baseline
        if 'random' in exploration:
            random_distances = np.ones(exploration['random'].n_samples) * exploration['random'].mean_distance
            
            for method, result in exploration.items():
                if method != 'random':
                    method_distances = np.ones(result.n_samples) * result.mean_distance
                    tests[f'{method}_vs_random'] = mann_whitney_test(
                        method_distances, random_distances, 'less'
                    )
        
        self.results['statistical_tests'] = tests
        return tests
    
    def generate_report(self, output_dir: str) -> str:
        """
        Generate comprehensive benchmark report.
        
        Args:
            output_dir: Directory to save report files
            
        Returns:
            Path to main report file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save JSON results
        json_path = output_path / f'benchmark_results_{timestamp}.json'
        with open(json_path, 'w') as f:
            json.dump({
                'config': {
                    'num_steps': self.config.num_steps,
                    'batch_size': self.config.batch_size,
                    'coupling_strength': self.config.coupling_strength,
                    'fuel_budget_fraction': self.config.fuel_budget_fraction
                },
                'exploration': {k: v.to_dict() for k, v in self.results.get('exploration', {}).items()},
                'statistical_tests': self.results.get('statistical_tests', {})
            }, f, indent=2)
        
        # Generate markdown report
        if 'exploration' in self.results:
            reporter = BenchmarkReporter(self.results['exploration'])
            md_report = reporter.to_markdown()
            
            md_path = output_path / f'benchmark_report_{timestamp}.md'
            with open(md_path, 'w') as f:
                f.write(md_report)
        
        print(f"Report saved to {output_path}")
        return str(json_path)


# =============================================================================
# Quick Benchmark Function
# =============================================================================

def quick_benchmark(
    n_samples: int = 100,
    num_steps: int = 200,
    seed: int = 42
) -> Dict[str, BenchmarkResult]:
    """
    Quick benchmark with default settings.
    
    Args:
        n_samples: Number of samples per method
        num_steps: Trajectory length
        seed: Random seed
        
    Returns:
        Dictionary of results
    """
    config = BenchmarkConfig(
        num_steps=num_steps,
        batch_size=n_samples
    )
    
    benchmark = ExplorationBenchmark(config)
    
    print(f"Running quick benchmark ({n_samples} samples, {num_steps} steps)...")
    results = benchmark.run_comparison(['random', 'biased_random'], n_samples, seed)
    
    return results


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    'BenchmarkResult',
    'BenchmarkConfig',
    'GeneratorRegistry',
    'ExplorationBenchmark',
    'mann_whitney_test',
    'compute_effect_size',
    'BenchmarkReporter',
    'FullBenchmarkSuite',
    'quick_benchmark'
]


if __name__ == "__main__":
    # Run quick benchmark
    print("=" * 60)
    print("ASL-Sandbox Benchmark Suite")
    print("=" * 60)
    
    results = quick_benchmark(n_samples=50, num_steps=100)
    
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    
    for method, result in results.items():
        print(f"\n{method}:")
        print(f"  Success Rate: {result.success_rate:.1%}")
        print(f"  Mean Distance: {result.mean_distance:.4f} L* ({result.mean_distance * L_STAR_KM:.0f} km)")
        print(f"  Min Distance: {result.min_distance:.4f} L* ({result.min_distance * L_STAR_KM:.0f} km)")
        print(f"  Wall Time: {result.wall_time_seconds:.2f} s")
