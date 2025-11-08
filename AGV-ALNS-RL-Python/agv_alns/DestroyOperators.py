"""
Destroy operators for AGV ALNS

These operators remove tasks from the current solution to create neighborhoods.
"""

from abc import ABC, abstractmethod
from typing import List
import random
import numpy as np
from copy import deepcopy

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agv_model import Solution, Event, EventType, Task


class DestroyOperator(ABC):
    """Base class for destroy operators"""

    @abstractmethod
    def remove_tasks(self, solution: Solution, num_remove: int) -> Solution:
        """
        Remove tasks from solution

        Args:
            solution: Current solution
            num_remove: Number of tasks to remove

        Returns:
            New solution with tasks in unassigned_tasks list
        """
        pass

    def _collect_all_tasks(self, solution: Solution) -> List[Tuple[int, int, Event]]:
        """
        Collect all task events from solution

        Returns:
            List of (agv_index, event_index, event)
        """
        all_tasks = []
        for agv_idx, agv in enumerate(solution.agv_paths):
            for event_idx, event in enumerate(agv.events):
                if event.is_task():
                    all_tasks.append((agv_idx, event_idx, event))
        return all_tasks

    def _remove_tasks_from_solution(self, solution: Solution,
                                   tasks_to_remove: List[Tuple[int, int, Event]]) -> Solution:
        """
        Remove specified tasks from solution

        Args:
            solution: Solution to modify
            tasks_to_remove: List of (agv_idx, event_idx, event)

        Returns:
            Modified solution
        """
        # Group by AGV and sort by event index (descending) to avoid index shifts
        tasks_by_agv = {}
        for agv_idx, event_idx, event in tasks_to_remove:
            if agv_idx not in tasks_by_agv:
                tasks_by_agv[agv_idx] = []
            tasks_by_agv[agv_idx].append((event_idx, event))

        # Remove tasks from each AGV
        for agv_idx, tasks in tasks_by_agv.items():
            # Sort by event_idx descending to avoid index issues
            tasks.sort(key=lambda x: x[0], reverse=True)
            agv = solution.agv_paths[agv_idx]

            for event_idx, event in tasks:
                # Add to unassigned list
                task = solution.problem.get_task(event.event_id)
                solution.unassigned_tasks.append(task)
                # Remove from AGV
                agv.remove_event(event_idx)

        return solution


class RandomTaskRemoval(DestroyOperator):
    """Random task removal (corresponds to C++ random removal)"""

    def remove_tasks(self, solution: Solution, num_remove: int) -> Solution:
        """Randomly remove tasks"""
        solution = solution.copy()

        all_tasks = self._collect_all_tasks(solution)

        if len(all_tasks) == 0:
            return solution

        # Randomly select tasks
        num_to_remove = min(num_remove, len(all_tasks))
        tasks_to_remove = random.sample(all_tasks, num_to_remove)

        # Remove tasks
        solution = self._remove_tasks_from_solution(solution, tasks_to_remove)

        return solution


class CriticalTaskRemoval(DestroyOperator):
    """
    Critical task removal

    Removes tasks on the critical path (those with tight time windows
    or high impact on makespan)
    """

    def remove_tasks(self, solution: Solution, num_remove: int) -> Solution:
        """Remove critical tasks"""
        solution = solution.copy()

        all_tasks = self._collect_all_tasks(solution)

        if len(all_tasks) == 0:
            return solution

        # Calculate criticality score for each task
        task_scores = []
        for agv_idx, event_idx, event in all_tasks:
            task = solution.problem.get_task(event.event_id)
            # Criticality = priority + time window tightness
            time_window_slack = task.latest_start - task.earliest_start
            if time_window_slack == float('inf'):
                time_window_slack = solution.problem.time_horizon

            criticality = task.priority / max(time_window_slack, 1.0)
            task_scores.append((agv_idx, event_idx, event, criticality))

        # Sort by criticality (descending)
        task_scores.sort(key=lambda x: x[3], reverse=True)

        # Select top critical tasks
        num_to_remove = min(num_remove, len(task_scores))
        tasks_to_remove = [(agv_idx, event_idx, event)
                          for agv_idx, event_idx, event, _ in task_scores[:num_to_remove]]

        # Remove tasks
        solution = self._remove_tasks_from_solution(solution, tasks_to_remove)

        return solution


class WorstTaskRemoval(DestroyOperator):
    """
    Worst task removal (similar to EVSP's GreedyRemoval)

    Removes tasks with highest individual cost contribution
    """

    def remove_tasks(self, solution: Solution, num_remove: int) -> Solution:
        """Remove worst (highest cost) tasks"""
        solution = solution.copy()

        all_tasks = self._collect_all_tasks(solution)

        if len(all_tasks) == 0:
            return solution

        # Calculate cost contribution for each task
        task_costs = []
        for agv_idx, event_idx, event in all_tasks:
            cost = self._calculate_task_cost(solution, agv_idx, event_idx, event)
            task_costs.append((agv_idx, event_idx, event, cost))

        # Sort by cost (descending)
        task_costs.sort(key=lambda x: x[3], reverse=True)

        # Select worst tasks
        num_to_remove = min(num_remove, len(task_costs))
        tasks_to_remove = [(agv_idx, event_idx, event)
                          for agv_idx, event_idx, event, _ in task_costs[:num_to_remove]]

        # Remove tasks
        solution = self._remove_tasks_from_solution(solution, tasks_to_remove)

        return solution

    def _calculate_task_cost(self, solution: Solution, agv_idx: int,
                           event_idx: int, event: Event) -> float:
        """Calculate the cost contribution of a task"""
        agv = solution.agv_paths[agv_idx]
        task = solution.problem.get_task(event.event_id)

        # Simplified cost: travel distance + waiting time
        cost = task.distance + event.waiting_time
        return cost


class BusiestStationRemoval(DestroyOperator):
    """
    Busiest charging station removal (C++ specific operator)

    Strategy:
    1. Find the most heavily used charging station
    2. Remove tasks from AGVs that charge at this station
    3. Encourages rescheduling to reduce station pressure
    """

    def remove_tasks(self, solution: Solution, num_remove: int) -> Solution:
        """Remove tasks related to busiest charging station"""
        solution = solution.copy()

        # Update charging station usage
        solution.update_charging_station_usage()

        # Find busiest station
        if not solution.charging_station_usage:
            # No charging events, fallback to random removal
            return RandomTaskRemoval().remove_tasks(solution, num_remove)

        station_usage_counts = {
            station_id: len(sessions)
            for station_id, sessions in solution.charging_station_usage.items()
        }

        busiest_station = max(station_usage_counts, key=station_usage_counts.get)

        # Collect AGVs that charge at busiest station
        agv_ids_at_station = set()
        for agv_id, start, end in solution.charging_station_usage[busiest_station]:
            agv_ids_at_station.add(agv_id)

        # Collect tasks from these AGVs
        tasks_to_remove = []
        for agv_idx, agv in enumerate(solution.agv_paths):
            if agv.agv_id in agv_ids_at_station:
                # Get all task events from this AGV
                task_events = [(agv_idx, event_idx, event)
                              for event_idx, event in enumerate(agv.events)
                              if event.is_task()]

                # Remove some tasks from this AGV
                if task_events:
                    num_from_agv = min(len(task_events), max(1, num_remove // len(agv_ids_at_station)))
                    tasks_to_remove.extend(random.sample(task_events, num_from_agv))

            if len(tasks_to_remove) >= num_remove:
                break

        # Limit to num_remove
        tasks_to_remove = tasks_to_remove[:num_remove]

        if not tasks_to_remove:
            # Fallback to random removal
            return RandomTaskRemoval().remove_tasks(solution, num_remove)

        # Remove tasks
        solution = self._remove_tasks_from_solution(solution, tasks_to_remove)

        return solution


class RandomStationRemoval(DestroyOperator):
    """
    Random charging station removal

    Randomly selects a charging station and removes tasks from AGVs using it
    """

    def remove_tasks(self, solution: Solution, num_remove: int) -> Solution:
        """Remove tasks related to a randomly selected charging station"""
        solution = solution.copy()

        # Update charging station usage
        solution.update_charging_station_usage()

        if not solution.charging_station_usage:
            # No charging events, fallback to random removal
            return RandomTaskRemoval().remove_tasks(solution, num_remove)

        # Randomly select a charging station
        station_id = random.choice(list(solution.charging_station_usage.keys()))

        # Collect AGVs that charge at this station
        agv_ids_at_station = set()
        for agv_id, start, end in solution.charging_station_usage[station_id]:
            agv_ids_at_station.add(agv_id)

        # Collect tasks from these AGVs
        tasks_to_remove = []
        for agv_idx, agv in enumerate(solution.agv_paths):
            if agv.agv_id in agv_ids_at_station:
                task_events = [(agv_idx, event_idx, event)
                              for event_idx, event in enumerate(agv.events)
                              if event.is_task()]

                if task_events:
                    tasks_to_remove.extend(task_events)

        # Randomly select from collected tasks
        if tasks_to_remove:
            num_to_remove = min(num_remove, len(tasks_to_remove))
            tasks_to_remove = random.sample(tasks_to_remove, num_to_remove)
        else:
            # Fallback to random removal
            return RandomTaskRemoval().remove_tasks(solution, num_remove)

        # Remove tasks
        solution = self._remove_tasks_from_solution(solution, tasks_to_remove)

        return solution


class StationRelatedRemoval(DestroyOperator):
    """
    Station-related removal

    Removes tasks that are spatially close to a randomly selected charging station
    """

    def remove_tasks(self, solution: Solution, num_remove: int) -> Solution:
        """Remove tasks related to a charging station by spatial proximity"""
        solution = solution.copy()

        if not solution.problem.charging_stations:
            # No stations, fallback to random removal
            return RandomTaskRemoval().remove_tasks(solution, num_remove)

        # Randomly select a charging station
        station = random.choice(solution.problem.charging_stations)
        station_loc = station.location

        # Collect all tasks with their distances to the station
        all_tasks = self._collect_all_tasks(solution)
        task_distances = []

        for agv_idx, event_idx, event in all_tasks:
            task = solution.problem.get_task(event.event_id)
            # Calculate average distance to station
            dist_start = np.sqrt((task.start_location[0] - station_loc[0])**2 +
                                (task.start_location[1] - station_loc[1])**2)
            dist_end = np.sqrt((task.end_location[0] - station_loc[0])**2 +
                              (task.end_location[1] - station_loc[1])**2)
            avg_dist = (dist_start + dist_end) / 2

            task_distances.append((agv_idx, event_idx, event, avg_dist))

        if not task_distances:
            return solution

        # Sort by distance (ascending - closest first)
        task_distances.sort(key=lambda x: x[3])

        # Select closest tasks
        num_to_remove = min(num_remove, len(task_distances))
        tasks_to_remove = [(agv_idx, event_idx, event)
                          for agv_idx, event_idx, event, _ in task_distances[:num_to_remove]]

        # Remove tasks
        solution = self._remove_tasks_from_solution(solution, tasks_to_remove)

        return solution
