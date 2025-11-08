"""
ALNS_RL: Main solver integrating ALNS with Reinforcement Learning

This solver uses Actor-Critic networks to intelligently select destroy and repair
operators instead of traditional adaptive weight management.
"""

import torch
import random
import math
import numpy as np
from copy import deepcopy
from typing import List, Tuple, Optional

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agv_model import Solution, ProblemData, AGVPath, Task
from agv_alns.GraphConverter import AGVGraphConverter
from agv_alns.DestroyOperators import (
    RandomTaskRemoval, CriticalTaskRemoval, WorstTaskRemoval,
    BusiestStationRemoval, RandomStationRemoval, StationRelatedRemoval
)
from agv_alns.RepairOperators import (
    GreedyOptimalInsert, GreedyRandomInsert, GreedyTimeInsert,
    RandomOptimalInsert, RandomRandomInsert, RandomTimeInsert,
    AdaptiveOptimalInsert, AdaptiveRandomInsert, AdaptiveTimeInsert
)
from rl_components import actor, critic


class AGV_ALNS_RL:
    """
    AGV ALNS solver with RL-based operator selection

    This class implements the main ALNS loop with RL components replacing
    traditional adaptive weight management.
    """

    def __init__(self, problem: ProblemData, use_rl: bool = True,
                 max_iterations: int = 1000, T0: float = 100.0, alpha: float = 0.995):
        """
        Initialize ALNS_RL solver

        Args:
            problem: Problem instance data
            use_rl: Whether to use RL for operator selection (if False, uses random)
            max_iterations: Maximum number of ALNS iterations
            T0: Initial temperature for simulated annealing
            alpha: Cooling rate for temperature
        """
        self.problem = problem
        self.use_rl = use_rl

        # ALNS parameters
        self.max_iterations = max_iterations
        self.T0 = T0
        self.alpha = alpha

        # Initialize destroy operators (6 total)
        self.destroy_ops = [
            RandomTaskRemoval(),
            CriticalTaskRemoval(),
            WorstTaskRemoval(),
            BusiestStationRemoval(),
            RandomStationRemoval(),
            StationRelatedRemoval()
        ]

        # Initialize repair operators (9 total)
        self.repair_ops = [
            GreedyOptimalInsert(),
            GreedyRandomInsert(),
            GreedyTimeInsert(),
            RandomOptimalInsert(),
            RandomRandomInsert(),
            RandomTimeInsert(),
            AdaptiveOptimalInsert(),
            AdaptiveRandomInsert(),
            AdaptiveTimeInsert()
        ]

        # RL components
        if use_rl:
            self.graph_converter = AGVGraphConverter(problem)

            feature_dim = 15  # 15-dimensional features for AGV problem
            hidden_dim = 128
            num_gnn_layers = 2
            num_mlp_layers = 3
            num_actions = len(self.destroy_ops) + len(self.repair_ops)  # 15 total

            self.actor = actor(feature_dim, hidden_dim, num_gnn_layers, num_mlp_layers, num_actions)
            self.critic = critic(feature_dim, hidden_dim, num_gnn_layers, num_mlp_layers)

            self.optimizer = torch.optim.Adam(
                list(self.actor.parameters()) + list(self.critic.parameters()),
                lr=1e-5
            )

            # Training records
            self.log_probs = []
            self.state_values = []
            self.rewards = []

        # Solution tracking
        self.best_solution = None
        self.current_solution = None
        self.best_cost = float('inf')

        # Statistics
        self.iteration_costs = []
        self.best_costs = []
        self.operator_stats = {
            'destroy': [0] * len(self.destroy_ops),
            'repair': [0] * len(self.repair_ops)
        }

    def solve(self, initial_solution: Optional[Solution] = None,
             verbose: bool = True) -> Solution:
        """
        Main ALNS solving loop

        Args:
            initial_solution: Initial solution (if None, generates one)
            verbose: Whether to print progress

        Returns:
            Best solution found
        """
        # Initialize solution
        if initial_solution is None:
            self.current_solution = self._generate_initial_solution()
        else:
            self.current_solution = initial_solution.copy()

        self.current_solution.compute_cost()
        self.best_solution = self.current_solution.copy()
        self.best_cost = self.current_solution.total_cost

        T = self.T0

        if verbose:
            print(f"Initial cost: {self.best_cost:.2f}")
            print(f"Starting ALNS with {len(self.destroy_ops)} destroy + {len(self.repair_ops)} repair operators")
            print(f"Use RL: {self.use_rl}, Max iterations: {self.max_iterations}")

        for iteration in range(self.max_iterations):
            # Select operators
            if self.use_rl:
                destroy_idx, repair_idx = self._select_operators_rl()
            else:
                destroy_idx = random.randint(0, len(self.destroy_ops) - 1)
                repair_idx = random.randint(0, len(self.repair_ops) - 1)

            # Track operator usage
            self.operator_stats['destroy'][destroy_idx] += 1
            self.operator_stats['repair'][repair_idx] += 1

            # Destroy
            num_remove = random.randint(3, min(10, self.problem.num_tasks // 2))
            temp_solution = self.current_solution.copy()
            temp_solution = self.destroy_ops[destroy_idx].remove_tasks(temp_solution, num_remove)

            # Repair
            temp_solution = self.repair_ops[repair_idx].insert_tasks(temp_solution)

            # Evaluate
            new_cost = temp_solution.compute_cost()

            # Acceptance criterion (simulated annealing)
            delta = new_cost - self.current_solution.total_cost
            accept = False

            if new_cost < self.best_cost:
                # New best solution
                self.best_solution = temp_solution.copy()
                self.best_cost = new_cost
                self.current_solution = temp_solution
                accept = True
                if verbose and iteration % 50 == 0:
                    print(f"Iter {iteration}: New best cost = {self.best_cost:.2f}")
            elif new_cost < self.current_solution.total_cost:
                # Improving solution
                self.current_solution = temp_solution
                accept = True
            elif random.random() < math.exp(-delta / T):
                # Accept with probability (simulated annealing)
                self.current_solution = temp_solution
                accept = True

            # RL update
            if self.use_rl:
                # Reward: improvement in cost
                reward = max(0, self.current_solution.total_cost - new_cost)
                self.rewards.append(reward)

                # Update RL every 100 iterations
                if iteration % 100 == 0 and iteration > 0:
                    self._update_rl()

            # Cool down temperature
            T *= self.alpha

            # Track statistics
            self.iteration_costs.append(new_cost)
            self.best_costs.append(self.best_cost)

            # Early stopping if temperature too low
            if T < 0.01:
                if verbose:
                    print(f"Temperature too low ({T:.6f}), stopping early at iteration {iteration}")
                break

        if verbose:
            print(f"\nALNS completed. Best cost: {self.best_cost:.2f}")
            print(f"Operator usage:")
            print(f"  Destroy: {self.operator_stats['destroy']}")
            print(f"  Repair: {self.operator_stats['repair']}")

        return self.best_solution

    def _select_operators_rl(self) -> Tuple[int, int]:
        """
        Use RL to select destroy and repair operators

        Returns:
            (destroy_operator_index, repair_operator_index)
        """
        # Convert current solution to graph
        graph = self.graph_converter.solution_to_graph(self.current_solution)

        # Select destroy operator
        mask_destroy = torch.tensor([True] * len(self.destroy_ops) + [False] * len(self.repair_ops))
        action_d, log_prob_d = self.actor(graph, mask_destroy, greedy=False)
        destroy_idx = action_d.item()

        # Select repair operator
        mask_repair = torch.tensor([False] * len(self.destroy_ops) + [True] * len(self.repair_ops))
        action_r, log_prob_r = self.actor(graph, mask_repair, greedy=False)
        repair_idx = action_r.item() - len(self.destroy_ops)

        # Get state value
        value = self.critic(graph)

        # Record for training
        self.log_probs.extend([log_prob_d, log_prob_r])
        self.state_values.extend([value, value])

        return destroy_idx, repair_idx

    def _update_rl(self):
        """Update RL networks using REINFORCE algorithm"""
        if not self.rewards:
            return

        # Calculate returns (rewards-to-go)
        returns = []
        R = 0
        for r in reversed(self.rewards):
            R = r + 0.99 * R  # Discount factor = 0.99
            returns.insert(0, R)

        # Normalize returns
        returns = torch.tensor(returns, dtype=torch.float)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Calculate losses
        policy_losses = []
        value_losses = []

        for log_prob, value, R in zip(self.log_probs, self.state_values, returns):
            advantage = R - value.item()

            # Policy loss (REINFORCE)
            policy_losses.append(-log_prob * advantage)

            # Value loss (MSE)
            value_losses.append(torch.nn.functional.mse_loss(value, torch.tensor([R])))

        # Backpropagation
        self.optimizer.zero_grad()
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
        loss.backward()
        self.optimizer.step()

        # Clear records
        self.log_probs = []
        self.state_values = []
        self.rewards = []

    def _generate_initial_solution(self) -> Solution:
        """
        Generate initial solution using greedy heuristic

        Strategy:
        1. Sort tasks by earliest start time
        2. Assign tasks to AGVs greedily (nearest available AGV)
        3. Insert charging when needed

        Returns:
            Initial solution
        """
        solution = Solution(self.problem)

        # Create initial AGV paths
        for i in range(self.problem.num_agvs):
            agv_path = AGVPath(
                agv_id=i,
                initial_soc=1.0,
                capacity=self.problem.battery_capacity,
                initial_location=(0, 0),
                depot_location=(0, 0)
            )
            solution.add_agv_path(agv_path)

        # Sort tasks by earliest start time
        sorted_tasks = sorted(self.problem.tasks, key=lambda t: t.earliest_start)

        # Assign tasks to AGVs greedily
        for task in sorted_tasks:
            # Find AGV with minimum makespan
            min_makespan = float('inf')
            best_agv_idx = 0

            for agv_idx, agv in enumerate(solution.agv_paths):
                if agv.makespan < min_makespan:
                    min_makespan = agv.makespan
                    best_agv_idx = agv_idx

            # Insert task to best AGV
            solution.unassigned_tasks = [task]
            solution = GreedyOptimalInsert().insert_tasks(solution)

        solution.compute_cost()
        return solution

    def save_model(self, path: str):
        """Save RL model parameters"""
        if self.use_rl:
            torch.save({
                'actor': self.actor.state_dict(),
                'critic': self.critic.state_dict(),
                'optimizer': self.optimizer.state_dict()
            }, path)
            print(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load RL model parameters"""
        if self.use_rl:
            checkpoint = torch.load(path)
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic.load_state_dict(checkpoint['critic'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"Model loaded from {path}")
