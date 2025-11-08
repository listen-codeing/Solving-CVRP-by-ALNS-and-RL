"""
AGVPath: Single AGV scheduling path

Similar to EVSP's DutyClass but adapted for AGV charging constraints.
"""

from typing import List, Tuple, Dict
from .Event import Event, EventType


class AGVPath:
    """
    Single AGV's scheduling path (similar to EVSP's DutyClass)

    Improvements over C++ version:
    1. Uses Event to unify task and charging representation
    2. Clearer state tracking (SOC changes)
    3. Independent validation logic
    """

    def __init__(self, agv_id: int, initial_soc: float, capacity: float,
                 initial_location: Tuple[float, float] = (0, 0),
                 depot_location: Tuple[float, float] = (0, 0)):
        """
        Initialize AGV path

        Args:
            agv_id: Unique AGV identifier
            initial_soc: Initial state of charge (0.0-1.0)
            capacity: Battery capacity
            initial_location: Starting location (x, y)
            depot_location: Depot location (x, y)
        """
        self.agv_id = agv_id
        self.initial_soc = initial_soc
        self.capacity = capacity
        self.initial_location = initial_location
        self.depot_location = depot_location

        # Event sequence (mixed tasks and charging)
        self.events: List[Event] = []

        # State tracking
        self.soc_before_events: List[float] = []
        self.soc_after_events: List[float] = []

        # Statistics
        self.total_travel_time = 0.0
        self.total_charging_time = 0.0
        self.total_waiting_time = 0.0
        self.makespan = 0.0  # Completion time

    def add_event(self, event: Event, soc_before: float, soc_after: float):
        """
        Add event to sequence

        Args:
            event: Event to add
            soc_before: SOC before this event
            soc_after: SOC after this event
        """
        self.events.append(event)
        self.soc_before_events.append(soc_before)
        self.soc_after_events.append(soc_after)

        # Update statistics
        if event.is_charging():
            self.total_charging_time += event.duration
        else:
            self.total_travel_time += event.duration
        self.total_waiting_time += event.waiting_time

        if self.events:
            self.makespan = max(self.makespan, event.end_time)

    def remove_event(self, event_index: int):
        """Remove event at given index"""
        if 0 <= event_index < len(self.events):
            removed_event = self.events.pop(event_index)
            self.soc_before_events.pop(event_index)
            self.soc_after_events.pop(event_index)

            # Update statistics
            if removed_event.is_charging():
                self.total_charging_time -= removed_event.duration
            else:
                self.total_travel_time -= removed_event.duration
            self.total_waiting_time -= removed_event.waiting_time

    def validate_soc_feasibility(self, soc_lower_bound: float = 0.2) -> Tuple[bool, str]:
        """
        Validate SOC feasibility

        Improvement: Independent validation method with clear constraint checking

        Args:
            soc_lower_bound: Minimum allowed SOC

        Returns:
            (is_feasible, message)
        """
        for i, soc in enumerate(self.soc_after_events):
            if soc < soc_lower_bound:
                return False, f"Event {i}: SOC={soc:.2f} < {soc_lower_bound}"
        return True, "OK"

    def get_task_sequence(self) -> List[Event]:
        """Extract task sequence (ignoring charging events)"""
        return [e for e in self.events if e.is_task()]

    def get_charging_sequence(self) -> List[Event]:
        """Extract charging sequence"""
        return [e for e in self.events if e.is_charging()]

    def compute_cost(self, cost_weights: Dict[str, float]) -> float:
        """
        Compute cost

        Improvement: Configurable cost weights

        Args:
            cost_weights: Dictionary with 'travel', 'charging', 'waiting' weights

        Returns:
            Total cost
        """
        travel_cost = self.total_travel_time * cost_weights.get('travel', 1.0)
        charging_cost = self.total_charging_time * cost_weights.get('charging', 1.0)
        waiting_cost = self.total_waiting_time * cost_weights.get('waiting', 1.0)

        return travel_cost + charging_cost + waiting_cost

    def get_workload(self) -> int:
        """Get number of tasks assigned to this AGV"""
        return len(self.get_task_sequence())

    def __repr__(self) -> str:
        return f"AGV{self.agv_id}(tasks={len(self.get_task_sequence())}, " \
               f"charges={len(self.get_charging_sequence())}, makespan={self.makespan:.2f})"
