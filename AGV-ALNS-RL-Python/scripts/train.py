"""
Training script for AGV-ALNS-RL

Trains the RL agent to select operators effectively.
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


def train(num_episodes: int = 100,
         num_tasks: int = 20,
         num_agvs: int = 5,
         num_stations: int = 3,
         max_iterations: int = 500,
         seed: int = 42,
         save_path: str = "./checkpoints/agv_alns_rl.pth"):
    """
    Train AGV-ALNS-RL agent

    Args:
        num_episodes: Number of training episodes
        num_tasks: Number of tasks per instance
        num_agvs: Number of AGVs
        num_stations: Number of charging stations
        max_iterations: ALNS iterations per episode
        seed: Random seed
        save_path: Path to save model
    """
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    print("=" * 80)
    print("AGV-ALNS-RL Training")
    print("=" * 80)
    print(f"Episodes: {num_episodes}")
    print(f"Problem size: {num_tasks} tasks, {num_agvs} AGVs, {num_stations} stations")
    print(f"ALNS iterations per episode: {max_iterations}")
    print(f"Random seed: {seed}")
    print("=" * 80)

    # Generate training instances
    print("\nGenerating training instances...")
    training_instances = []
    for i in range(num_episodes):
        problem = generate_agv_instance(
            num_tasks=num_tasks,
            num_agvs=num_agvs,
            num_stations=num_stations,
            seed=seed + i
        )
        training_instances.append(problem)
    print(f"Generated {len(training_instances)} instances")

    # Create solver (will be reused across episodes)
    problem = training_instances[0]
    solver = AGV_ALNS_RL(
        problem=problem,
        use_rl=True,
        max_iterations=max_iterations,
        T0=100.0,
        alpha=0.995
    )

    # Training loop
    episode_costs = []
    episode_improvements = []

    print("\nStarting training...\n")
    for episode in tqdm(range(num_episodes), desc="Training"):
        # Get problem instance
        problem = training_instances[episode]

        # Update solver's problem
        solver.problem = problem
        solver.graph_converter.problem = problem

        # Solve
        solution = solver.solve(verbose=False)

        # Record statistics
        initial_cost = solver.iteration_costs[0] if solver.iteration_costs else solution.total_cost
        final_cost = solution.total_cost
        improvement = (initial_cost - final_cost) / initial_cost * 100

        episode_costs.append(final_cost)
        episode_improvements.append(improvement)

        # Print progress every 10 episodes
        if (episode + 1) % 10 == 0:
            avg_cost = np.mean(episode_costs[-10:])
            avg_improvement = np.mean(episode_improvements[-10:])
            print(f"\nEpisode {episode + 1}/{num_episodes}:")
            print(f"  Avg cost (last 10): {avg_cost:.2f}")
            print(f"  Avg improvement (last 10): {avg_improvement:.2f}%")

    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    solver.save_model(save_path)

    # Print final statistics
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Average cost: {np.mean(episode_costs):.2f}")
    print(f"Best cost: {np.min(episode_costs):.2f}")
    print(f"Worst cost: {np.max(episode_costs):.2f}")
    print(f"Average improvement: {np.mean(episode_improvements):.2f}%")
    print(f"Model saved to: {save_path}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AGV-ALNS-RL agent")
    parser.add_argument("--episodes", type=int, default=100, help="Number of training episodes")
    parser.add_argument("--tasks", type=int, default=20, help="Number of tasks per instance")
    parser.add_argument("--agvs", type=int, default=5, help="Number of AGVs")
    parser.add_argument("--stations", type=int, default=3, help="Number of charging stations")
    parser.add_argument("--iterations", type=int, default=500, help="ALNS iterations per episode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save", type=str, default="./checkpoints/agv_alns_rl.pth", help="Model save path")

    args = parser.parse_args()

    train(
        num_episodes=args.episodes,
        num_tasks=args.tasks,
        num_agvs=args.agvs,
        num_stations=args.stations,
        max_iterations=args.iterations,
        seed=args.seed,
        save_path=args.save
    )
