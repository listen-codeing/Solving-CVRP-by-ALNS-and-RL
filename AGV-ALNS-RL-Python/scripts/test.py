"""
Testing script for AGV-ALNS-RL

Tests trained model on new instances and compares RL vs non-RL performance.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import random
import argparse
from tqdm import tqdm

from agv_alns.ALNS_RL import AGV_ALNS_RL
from scripts.data_generator import generate_agv_instance


def test(model_path: str,
        num_instances: int = 20,
        num_tasks: int = 20,
        num_agvs: int = 5,
        num_stations: int = 3,
        max_iterations: int = 500,
        seed: int = 123):
    """
    Test AGV-ALNS-RL agent

    Args:
        model_path: Path to trained model
        num_instances: Number of test instances
        num_tasks: Number of tasks per instance
        num_agvs: Number of AGVs
        num_stations: Number of charging stations
        max_iterations: ALNS iterations per instance
        seed: Random seed
    """
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    print("=" * 80)
    print("AGV-ALNS-RL Testing")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Test instances: {num_instances}")
    print(f"Problem size: {num_tasks} tasks, {num_agvs} AGVs, {num_stations} stations")
    print(f"ALNS iterations per instance: {max_iterations}")
    print(f"Random seed: {seed}")
    print("=" * 80)

    # Generate test instances
    print("\nGenerating test instances...")
    test_instances = []
    for i in range(num_instances):
        problem = generate_agv_instance(
            num_tasks=num_tasks,
            num_agvs=num_agvs,
            num_stations=num_stations,
            seed=seed + i
        )
        test_instances.append(problem)
    print(f"Generated {len(test_instances)} instances")

    # Test with RL
    print("\n" + "=" * 80)
    print("Testing with RL...")
    print("=" * 80)

    rl_costs = []
    rl_improvements = []

    for i, problem in enumerate(tqdm(test_instances, desc="RL Testing")):
        solver = AGV_ALNS_RL(
            problem=problem,
            use_rl=True,
            max_iterations=max_iterations
        )

        # Load trained model
        if os.path.exists(model_path):
            solver.load_model(model_path)
        else:
            print(f"Warning: Model not found at {model_path}, using untrained model")

        solution = solver.solve(verbose=False)

        initial_cost = solver.iteration_costs[0] if solver.iteration_costs else solution.total_cost
        final_cost = solution.total_cost
        improvement = (initial_cost - final_cost) / initial_cost * 100

        rl_costs.append(final_cost)
        rl_improvements.append(improvement)

    # Test without RL (random operator selection)
    print("\n" + "=" * 80)
    print("Testing without RL (random baseline)...")
    print("=" * 80)

    baseline_costs = []
    baseline_improvements = []

    for i, problem in enumerate(tqdm(test_instances, desc="Baseline Testing")):
        solver = AGV_ALNS_RL(
            problem=problem,
            use_rl=False,  # Disable RL
            max_iterations=max_iterations
        )

        solution = solver.solve(verbose=False)

        initial_cost = solver.iteration_costs[0] if solver.iteration_costs else solution.total_cost
        final_cost = solution.total_cost
        improvement = (initial_cost - final_cost) / initial_cost * 100

        baseline_costs.append(final_cost)
        baseline_improvements.append(improvement)

    # Compare results
    print("\n" + "=" * 80)
    print("Results Comparison")
    print("=" * 80)

    print("\nRL Performance:")
    print(f"  Average cost: {np.mean(rl_costs):.2f}")
    print(f"  Best cost: {np.min(rl_costs):.2f}")
    print(f"  Worst cost: {np.max(rl_costs):.2f}")
    print(f"  Std dev: {np.std(rl_costs):.2f}")
    print(f"  Average improvement: {np.mean(rl_improvements):.2f}%")

    print("\nBaseline Performance (Random):")
    print(f"  Average cost: {np.mean(baseline_costs):.2f}")
    print(f"  Best cost: {np.min(baseline_costs):.2f}")
    print(f"  Worst cost: {np.max(baseline_costs):.2f}")
    print(f"  Std dev: {np.std(baseline_costs):.2f}")
    print(f"  Average improvement: {np.mean(baseline_improvements):.2f}%")

    print("\nRL vs Baseline:")
    avg_improvement = (np.mean(baseline_costs) - np.mean(rl_costs)) / np.mean(baseline_costs) * 100
    print(f"  RL improvement over baseline: {avg_improvement:.2f}%")

    if avg_improvement > 0:
        print(f"  ✓ RL outperforms baseline by {avg_improvement:.2f}%")
    else:
        print(f"  ✗ RL underperforms baseline by {-avg_improvement:.2f}%")

    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test AGV-ALNS-RL agent")
    parser.add_argument("--model", type=str, default="./checkpoints/agv_alns_rl.pth", help="Path to trained model")
    parser.add_argument("--instances", type=int, default=20, help="Number of test instances")
    parser.add_argument("--tasks", type=int, default=20, help="Number of tasks per instance")
    parser.add_argument("--agvs", type=int, default=5, help="Number of AGVs")
    parser.add_argument("--stations", type=int, default=3, help="Number of charging stations")
    parser.add_argument("--iterations", type=int, default=500, help="ALNS iterations per instance")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")

    args = parser.parse_args()

    test(
        model_path=args.model,
        num_instances=args.instances,
        num_tasks=args.tasks,
        num_agvs=args.agvs,
        num_stations=args.stations,
        max_iterations=args.iterations,
        seed=args.seed
    )
