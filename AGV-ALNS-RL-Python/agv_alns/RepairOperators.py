"""
Repair operators for AGV ALNS

These operators insert unassigned tasks back into the solution.
Follows a 3x3 matrix design:
- Selection: Greedy, Random, Adaptive
- Optimization: Optimal, Random, Time
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict
import random
import numpy as np
from copy import deepcopy

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agv_model import Solution, Event, EventType, Task, AGVPath


class RepairOperator(ABC):
    """Base class for repair operators"""

    @abstractmethod
    def insert_tasks(self, solution: Solution) -> Solution:
        """
        Insert unassigned tasks into solution

        Args:
            solution: Current solution with unassigned_tasks

        Returns:
            New solution with all tasks assigned
        """
        pass

    def _find_best_insertion(self, solution: Solution, task: Task,
                            optimization_goal: str = 'optimal') -> Tuple[float, Optional[int], Optional[int]]:
        """
        Find best insertion position for a task

        Args:
            solution: Current solution
            task: Task to insert
            optimization_goal: 'optimal', 'random', or 'time'

        Returns:
            (cost, agv_idx, position) or (inf, None, None) if not feasible
        """
        best_cost = float('inf')
        best_agv_idx = None
        best_pos = None

        # Try all AGVs
        for agv_idx, agv in enumerate(solution.agv_paths):
            # Try all positions in this AGV's route
            for pos in range(len(agv.events) + 1):
                # Calculate insertion cost based on optimization goal
                if optimization_goal == 'optimal':
                    cost = self._calculate_optimal_cost(solution, agv_idx, task, pos)
                elif optimization_goal == 'random':
                    cost = random.random()  # Random cost
                elif optimization_goal == 'time':
                    cost = self._calculate_time_cost(solution, agv_idx, task, pos)
                else:
                    cost = self._calculate_optimal_cost(solution, agv_idx, task, pos)

                # Check feasibility
                if self._is_insertion_feasible(solution, agv_idx, task, pos):
                    if cost < best_cost:
                        best_cost = cost
                        best_agv_idx = agv_idx
                        best_pos = pos

        return best_cost, best_agv_idx, best_pos

    def _calculate_optimal_cost(self, solution: Solution, agv_idx: int,
                               task: Task, pos: int) -> float:
        """Calculate insertion cost for optimal objective (minimize total distance)"""
        agv = solution.agv_paths[agv_idx]

        # Get previous and next locations
        if pos == 0:
            prev_loc = agv.initial_location
        else:
            prev_event = agv.events[pos - 1]
            prev_loc = self._get_event_end_location(solution, prev_event)

        if pos >= len(agv.events):
            next_loc = agv.depot_location
        else:
            next_event = agv.events[pos]
            next_loc = self._get_event_start_location(solution, next_event)

        # Calculate distance change
        task_start = task.start_location
        task_end = task.end_location

        # Old distance (without task)
        old_dist = self._euclidean_distance(prev_loc, next_loc)

        # New distance (with task)
        new_dist = (self._euclidean_distance(prev_loc, task_start) +
                   self._euclidean_distance(task_start, task_end) +
                   self._euclidean_distance(task_end, next_loc))

        return new_dist - old_dist

    def _calculate_time_cost(self, solution: Solution, agv_idx: int,
                            task: Task, pos: int) -> float:
        """Calculate insertion cost for time objective (minimize makespan)"""
        agv = solution.agv_paths[agv_idx]

        # Estimate completion time if task is inserted at position pos
        # Simplified: use current makespan + estimated time for this task
        estimated_task_time = task.service_time + task.distance

        if pos == len(agv.events):
            # Inserted at end
            if agv.events:
                return agv.makespan + estimated_task_time
            else:
                return estimated_task_time
        else:
            # Inserted in middle - may delay subsequent tasks
            return estimated_task_time

    def _is_insertion_feasible(self, solution: Solution, agv_idx: int,
                              task: Task, pos: int) -> bool:
        """
        Check if insertion is feasible

        Currently simplified - assumes always feasible.
        In full implementation, should check:
        - SOC constraints
        - Time window constraints
        - Capacity constraints
        """
        # TODO: Implement full feasibility check
        # For now, return True to allow insertions
        return True

    def _get_event_start_location(self, solution: Solution, event: Event) -> Tuple[float, float]:
        """Get start location of an event"""
        if event.is_task():
            task = solution.problem.get_task(event.event_id)
            return task.start_location
        else:
            station = solution.problem.get_station(event.event_id)
            return station.location

    def _get_event_end_location(self, solution: Solution, event: Event) -> Tuple[float, float]:
        """Get end location of an event"""
        if event.is_task():
            task = solution.problem.get_task(event.event_id)
            return task.end_location
        else:
            station = solution.problem.get_station(event.event_id)
            return station.location

    def _euclidean_distance(self, loc1: Tuple[float, float], loc2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two locations"""
        return np.sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)

    def _insert_task_at_position(self, solution: Solution, task: Task,
                                 agv_idx: int, pos: int):
        """
        Insert task at specified position

        Args:
            solution: Solution to modify
            task: Task to insert
            agv_idx: Index of AGV
            pos: Position in event sequence
        """
        agv = solution.agv_paths[agv_idx]

        # Create task event
        # Simplified timing - in full implementation, should calculate exact times
        if pos == 0:
            start_time = task.earliest_start
        elif pos < len(agv.events):
            start_time = max(task.earliest_start, agv.events[pos - 1].end_time)
        else:
            start_time = max(task.earliest_start,
                           agv.events[-1].end_time if agv.events else 0.0)

        end_time = start_time + task.service_time

        event = Event(
            event_type=EventType.TASK,
            event_id=task.task_id,
            start_time=start_time,
            end_time=end_time,
            waiting_time=0.0
        )

        # Simplified SOC calculation - assume sufficient battery
        soc_before = 0.5  # Placeholder
        soc_after = 0.4   # Placeholder

        # Insert event
        agv.events.insert(pos, event)
        agv.soc_before_events.insert(pos, soc_before)
        agv.soc_after_events.insert(pos, soc_after)

        # Update AGV statistics
        agv.total_travel_time += task.service_time
        if end_time > agv.makespan:
            agv.makespan = end_time


# ============================================================================
# 3x3 Matrix of Repair Operators
# ============================================================================

# Row 1: Greedy Selection
class GreedyOptimalInsert(RepairOperator):
    """Greedy selection + Optimal objective (minimize distance)"""

    def insert_tasks(self, solution: Solution) -> Solution:
        solution = solution.copy()

        while solution.unassigned_tasks:
            # Calculate best insertion for all tasks
            best_overall_cost = float('inf')
            best_task = None
            best_agv_idx = None
            best_pos = None

            for task in solution.unassigned_tasks:
                cost, agv_idx, pos = self._find_best_insertion(solution, task, 'optimal')
                if cost < best_overall_cost:
                    best_overall_cost = cost
                    best_task = task
                    best_agv_idx = agv_idx
                    best_pos = pos

            if best_task is None:
                break  # Cannot insert any more tasks

            # Insert best task
            self._insert_task_at_position(solution, best_task, best_agv_idx, best_pos)
            solution.unassigned_tasks.remove(best_task)

        return solution


class GreedyRandomInsert(RepairOperator):
    """Greedy selection + Random objective"""

    def insert_tasks(self, solution: Solution) -> Solution:
        solution = solution.copy()

        while solution.unassigned_tasks:
            best_overall_cost = float('inf')
            best_task = None
            best_agv_idx = None
            best_pos = None

            for task in solution.unassigned_tasks:
                cost, agv_idx, pos = self._find_best_insertion(solution, task, 'random')
                if cost < best_overall_cost:
                    best_overall_cost = cost
                    best_task = task
                    best_agv_idx = agv_idx
                    best_pos = pos

            if best_task is None:
                break

            self._insert_task_at_position(solution, best_task, best_agv_idx, best_pos)
            solution.unassigned_tasks.remove(best_task)

        return solution


class GreedyTimeInsert(RepairOperator):
    """Greedy selection + Time objective (minimize makespan)"""

    def insert_tasks(self, solution: Solution) -> Solution:
        solution = solution.copy()

        while solution.unassigned_tasks:
            best_overall_cost = float('inf')
            best_task = None
            best_agv_idx = None
            best_pos = None

            for task in solution.unassigned_tasks:
                cost, agv_idx, pos = self._find_best_insertion(solution, task, 'time')
                if cost < best_overall_cost:
                    best_overall_cost = cost
                    best_task = task
                    best_agv_idx = agv_idx
                    best_pos = pos

            if best_task is None:
                break

            self._insert_task_at_position(solution, best_task, best_agv_idx, best_pos)
            solution.unassigned_tasks.remove(best_task)

        return solution


# Row 2: Random Selection
class RandomOptimalInsert(RepairOperator):
    """Random selection + Optimal objective"""

    def insert_tasks(self, solution: Solution) -> Solution:
        solution = solution.copy()

        # Shuffle unassigned tasks
        random.shuffle(solution.unassigned_tasks)

        while solution.unassigned_tasks:
            # Randomly select a task
            task = solution.unassigned_tasks[0]

            # Find best insertion for this task
            cost, agv_idx, pos = self._find_best_insertion(solution, task, 'optimal')

            if agv_idx is None:
                break  # Cannot insert this task

            self._insert_task_at_position(solution, task, agv_idx, pos)
            solution.unassigned_tasks.remove(task)

        return solution


class RandomRandomInsert(RepairOperator):
    """Random selection + Random objective"""

    def insert_tasks(self, solution: Solution) -> Solution:
        solution = solution.copy()

        while solution.unassigned_tasks:
            # Randomly select a task
            task = random.choice(solution.unassigned_tasks)

            # Find random insertion
            cost, agv_idx, pos = self._find_best_insertion(solution, task, 'random')

            if agv_idx is None:
                # Try random AGV and position
                if solution.agv_paths:
                    agv_idx = random.randint(0, len(solution.agv_paths) - 1)
                    pos = random.randint(0, len(solution.agv_paths[agv_idx].events))
                else:
                    break

            self._insert_task_at_position(solution, task, agv_idx, pos)
            solution.unassigned_tasks.remove(task)

        return solution


class RandomTimeInsert(RepairOperator):
    """Random selection + Time objective"""

    def insert_tasks(self, solution: Solution) -> Solution:
        solution = solution.copy()

        random.shuffle(solution.unassigned_tasks)

        while solution.unassigned_tasks:
            task = solution.unassigned_tasks[0]

            cost, agv_idx, pos = self._find_best_insertion(solution, task, 'time')

            if agv_idx is None:
                break

            self._insert_task_at_position(solution, task, agv_idx, pos)
            solution.unassigned_tasks.remove(task)

        return solution


# Row 3: Adaptive Selection
class AdaptiveOptimalInsert(RepairOperator):
    """Adaptive selection (regret-based) + Optimal objective"""

    def insert_tasks(self, solution: Solution) -> Solution:
        solution = solution.copy()

        while solution.unassigned_tasks:
            # Calculate regret for each task (difference between best and second-best insertion)
            task_regrets = []

            for task in solution.unassigned_tasks:
                # Find best and second-best insertions
                insertion_costs = []
                for agv_idx in range(len(solution.agv_paths)):
                    for pos in range(len(solution.agv_paths[agv_idx].events) + 1):
                        if self._is_insertion_feasible(solution, agv_idx, task, pos):
                            cost = self._calculate_optimal_cost(solution, agv_idx, task, pos)
                            insertion_costs.append((cost, agv_idx, pos))

                if len(insertion_costs) == 0:
                    continue
                elif len(insertion_costs) == 1:
                    regret = insertion_costs[0][0]  # Only one option
                    best = insertion_costs[0]
                else:
                    insertion_costs.sort(key=lambda x: x[0])
                    regret = insertion_costs[1][0] - insertion_costs[0][0]
                    best = insertion_costs[0]

                task_regrets.append((regret, task, best))

            if not task_regrets:
                break

            # Select task with highest regret
            task_regrets.sort(key=lambda x: x[0], reverse=True)
            regret, task, (cost, agv_idx, pos) = task_regrets[0]

            self._insert_task_at_position(solution, task, agv_idx, pos)
            solution.unassigned_tasks.remove(task)

        return solution


class AdaptiveRandomInsert(RepairOperator):
    """Adaptive selection + Random objective"""

    def insert_tasks(self, solution: Solution) -> Solution:
        solution = solution.copy()

        while solution.unassigned_tasks:
            task_regrets = []

            for task in solution.unassigned_tasks:
                insertion_costs = []
                for agv_idx in range(len(solution.agv_paths)):
                    for pos in range(len(solution.agv_paths[agv_idx].events) + 1):
                        if self._is_insertion_feasible(solution, agv_idx, task, pos):
                            cost = random.random()
                            insertion_costs.append((cost, agv_idx, pos))

                if len(insertion_costs) == 0:
                    continue
                elif len(insertion_costs) == 1:
                    regret = insertion_costs[0][0]
                    best = insertion_costs[0]
                else:
                    insertion_costs.sort(key=lambda x: x[0])
                    regret = insertion_costs[1][0] - insertion_costs[0][0]
                    best = insertion_costs[0]

                task_regrets.append((regret, task, best))

            if not task_regrets:
                break

            task_regrets.sort(key=lambda x: x[0], reverse=True)
            regret, task, (cost, agv_idx, pos) = task_regrets[0]

            self._insert_task_at_position(solution, task, agv_idx, pos)
            solution.unassigned_tasks.remove(task)

        return solution


class AdaptiveTimeInsert(RepairOperator):
    """Adaptive selection + Time objective"""

    def insert_tasks(self, solution: Solution) -> Solution:
        solution = solution.copy()

        while solution.unassigned_tasks:
            task_regrets = []

            for task in solution.unassigned_tasks:
                insertion_costs = []
                for agv_idx in range(len(solution.agv_paths)):
                    for pos in range(len(solution.agv_paths[agv_idx].events) + 1):
                        if self._is_insertion_feasible(solution, agv_idx, task, pos):
                            cost = self._calculate_time_cost(solution, agv_idx, task, pos)
                            insertion_costs.append((cost, agv_idx, pos))

                if len(insertion_costs) == 0:
                    continue
                elif len(insertion_costs) == 1:
                    regret = insertion_costs[0][0]
                    best = insertion_costs[0]
                else:
                    insertion_costs.sort(key=lambda x: x[0])
                    regret = insertion_costs[1][0] - insertion_costs[0][0]
                    best = insertion_costs[0]

                task_regrets.append((regret, task, best))

            if not task_regrets:
                break

            task_regrets.sort(key=lambda x: x[0], reverse=True)
            regret, task, (cost, agv_idx, pos) = task_regrets[0]

            self._insert_task_at_position(solution, task, agv_idx, pos)
            solution.unassigned_tasks.remove(task)

        return solution
