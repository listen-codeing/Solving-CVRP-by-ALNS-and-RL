"""
Solution: Overall AGV scheduling solution

Similar to EVSP's ScheduleClass but adapted for AGV charging constraints.
"""

from typing import List, Dict, Tuple
from copy import deepcopy
from .AGVPath import AGVPath
from .Event import Event, EventType
from .ProblemData import ProblemData, Task


class Solution:
    """
    Overall scheduling solution (similar to EVSP's ScheduleClass)

    Improvements over C++ version:
    1. Clear AGV list management
    2. Charging station state tracking
    3. Separated constraint checking
    """

    def __init__(self, problem_data: ProblemData):
        """
        Initialize solution

        Args:
            problem_data: Problem instance data
        """
        self.problem = problem_data

        # AGV scheduling list
        self.agv_paths: List[AGVPath] = []

        # Unassigned tasks (for ALNS destroy operations)
        self.unassigned_tasks: List[Task] = []

        # Charging station usage tracking
        # {station_id: [(agv_id, start_time, end_time), ...]}
        self.charging_station_usage: Dict[int, List[Tuple[int, float, float]]] = {}

        # Cost
        self.total_cost = 0.0

    def add_agv_path(self, agv_path: AGVPath):
        """Add AGV path to solution"""
        self.agv_paths.append(agv_path)

    def validate_feasibility(self) -> Tuple[bool, str]:
        """
        Validate overall feasibility

        Improvement: Separated constraint checking

        Returns:
            (is_feasible, message)
        """
        # 1. SOC constraints
        for agv in self.agv_paths:
            feasible, msg = agv.validate_soc_feasibility()
            if not feasible:
                return False, f"AGV {agv.agv_id}: {msg}"

        # 2. Charging station capacity constraints
        if not self._check_charging_station_capacity():
            return False, "Charging station capacity exceeded"

        # 3. Task coverage
        if len(self.unassigned_tasks) > 0:
            return False, f"{len(self.unassigned_tasks)} tasks unassigned"

        # 4. No duplicate tasks
        if not self._check_no_duplicate_tasks():
            return False, "Duplicate task assignments detected"

        return True, "OK"

    def _check_charging_station_capacity(self) -> bool:
        """Check charging station capacity (simultaneous charging AGVs)"""
        for station_id, sessions in self.charging_station_usage.items():
            # Check maximum concurrent charging at any time
            max_concurrent = self._get_max_concurrent_charging(sessions)
            station_capacity = self.problem.get_station_capacity(station_id)
            if max_concurrent > station_capacity:
                return False
        return True

    def _get_max_concurrent_charging(self, sessions: List[Tuple[int, float, float]]) -> int:
        """
        Calculate maximum concurrent charging count

        Uses sweep line algorithm

        Args:
            sessions: List of (agv_id, start_time, end_time)

        Returns:
            Maximum number of AGVs charging simultaneously
        """
        if not sessions:
            return 0

        events = []
        for agv_id, start, end in sessions:
            events.append((start, 1))   # Start charging
            events.append((end, -1))    # End charging

        events.sort()
        current = 0
        max_concurrent = 0

        for time, delta in events:
            current += delta
            max_concurrent = max(max_concurrent, current)

        return max_concurrent

    def _check_no_duplicate_tasks(self) -> bool:
        """Check that no task is assigned multiple times"""
        task_ids = set()
        for agv in self.agv_paths:
            for event in agv.get_task_sequence():
                if event.event_id in task_ids:
                    return False
                task_ids.add(event.event_id)
        return True

    def compute_cost(self) -> float:
        """
        Compute total cost

        Returns:
            Total cost of the solution
        """
        cost_weights = {
            'travel': self.problem.cost_travel,
            'charging': self.problem.cost_charging,
            'waiting': self.problem.cost_waiting
        }

        self.total_cost = sum(
            agv.compute_cost(cost_weights) for agv in self.agv_paths
        )
        return self.total_cost

    def update_charging_station_usage(self):
        """Update charging station usage from AGV paths"""
        self.charging_station_usage.clear()

        for agv in self.agv_paths:
            for event in agv.get_charging_sequence():
                station_id = event.event_id
                if station_id not in self.charging_station_usage:
                    self.charging_station_usage[station_id] = []

                self.charging_station_usage[station_id].append(
                    (agv.agv_id, event.start_time, event.end_time)
                )

    def get_makespan(self) -> float:
        """Get overall makespan (maximum completion time)"""
        if not self.agv_paths:
            return 0.0
        return max(agv.makespan for agv in self.agv_paths)

    def get_total_tasks(self) -> int:
        """Get total number of tasks assigned"""
        return sum(len(agv.get_task_sequence()) for agv in self.agv_paths)

    def get_all_assigned_task_ids(self) -> List[int]:
        """Get list of all assigned task IDs"""
        task_ids = []
        for agv in self.agv_paths:
            for event in agv.get_task_sequence():
                task_ids.append(event.event_id)
        return task_ids

    def copy(self) -> 'Solution':
        """Create a deep copy of the solution"""
        return deepcopy(self)

    def __repr__(self) -> str:
        return f"Solution(AGVs={len(self.agv_paths)}, " \
               f"tasks={self.get_total_tasks()}/{len(self.problem.tasks)}, " \
               f"unassigned={len(self.unassigned_tasks)}, " \
               f"cost={self.total_cost:.2f})"
