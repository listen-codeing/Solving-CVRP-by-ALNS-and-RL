"""
Problem data structures for AGV scheduling problem.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np


@dataclass
class Task:
    """
    AGV transport task

    Attributes:
        task_id: Unique task identifier
        start_location: Pickup location (x, y)
        end_location: Delivery location (x, y)
        earliest_start: Earliest start time (time window)
        latest_start: Latest start time (time window)
        service_time: Time required to complete the task
        priority: Task priority (higher = more important)
    """
    task_id: int
    start_location: Tuple[float, float]
    end_location: Tuple[float, float]
    earliest_start: float = 0.0
    latest_start: float = float('inf')
    service_time: float = 1.0
    priority: int = 1

    @property
    def distance(self) -> float:
        """Euclidean distance of the task"""
        dx = self.end_location[0] - self.start_location[0]
        dy = self.end_location[1] - self.start_location[1]
        return np.sqrt(dx**2 + dy**2)


@dataclass
class ChargingStation:
    """
    Charging station

    Attributes:
        station_id: Unique station identifier
        location: Station location (x, y)
        capacity: Maximum number of AGVs that can charge simultaneously
        charging_rate: Charging rate (SOC per time unit)
    """
    station_id: int
    location: Tuple[float, float]
    capacity: int = 1
    charging_rate: float = 0.1  # 10% per time unit


class ProblemData:
    """
    Complete problem data for AGV scheduling

    Contains all problem-specific information including tasks, charging stations,
    map dimensions, and cost parameters.
    """

    def __init__(self,
                 tasks: List[Task],
                 charging_stations: List[ChargingStation],
                 num_agvs: int,
                 battery_capacity: float = 100.0,
                 energy_consumption_rate: float = 0.001,  # SOC per distance unit
                 map_width: float = 100.0,
                 map_height: float = 100.0):
        """
        Initialize problem data

        Args:
            tasks: List of tasks to complete
            charging_stations: List of charging stations
            num_agvs: Number of available AGVs
            battery_capacity: Battery capacity (in energy units)
            energy_consumption_rate: Energy consumed per distance unit
            map_width: Map width for normalization
            map_height: Map height for normalization
        """
        self.tasks = tasks
        self.charging_stations = charging_stations
        self.num_agvs = num_agvs
        self.battery_capacity = battery_capacity
        self.energy_consumption_rate = energy_consumption_rate
        self.map_width = map_width
        self.map_height = map_height

        # Build distance matrix
        self.distance_matrix = self._build_distance_matrix()

        # Cost weights
        self.cost_travel = 1.0
        self.cost_charging = 0.5
        self.cost_waiting = 0.3

        # Time horizon for normalization
        self.time_horizon = self._estimate_time_horizon()
        self.max_task_duration = max([t.service_time for t in tasks]) if tasks else 1.0
        self.max_waiting_time = 10.0
        self.max_distance = np.max(self.distance_matrix) if len(self.distance_matrix) > 0 else 1.0
        self.max_charging_time = 1.0 / charging_stations[0].charging_rate if charging_stations else 10.0

    def _build_distance_matrix(self) -> np.ndarray:
        """
        Build distance matrix for all locations

        Includes:
        - Task start and end locations
        - Charging station locations
        - Depot (assumed at (0, 0))
        """
        # Collect all locations
        locations = [(0, 0)]  # Depot

        # Task locations
        for task in self.tasks:
            locations.append(task.start_location)
            locations.append(task.end_location)

        # Charging station locations
        for station in self.charging_stations:
            locations.append(station.location)

        # Build distance matrix
        n = len(locations)
        dist_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i != j:
                    dx = locations[i][0] - locations[j][0]
                    dy = locations[i][1] - locations[j][1]
                    dist_matrix[i][j] = np.sqrt(dx**2 + dy**2)

        return dist_matrix

    def _estimate_time_horizon(self) -> float:
        """Estimate problem time horizon"""
        if not self.tasks:
            return 100.0

        # Use latest deadline as time horizon
        max_deadline = max([t.latest_start for t in self.tasks if t.latest_start != float('inf')])
        if max_deadline == float('-inf') or max_deadline == float('inf'):
            return 100.0

        return max_deadline * 1.2  # Add 20% buffer

    def get_station_capacity(self, station_id: int) -> int:
        """Get capacity of a charging station"""
        for station in self.charging_stations:
            if station.station_id == station_id:
                return station.capacity
        return 1  # Default

    def get_task(self, task_id: int) -> Task:
        """Get task by ID"""
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        raise ValueError(f"Task {task_id} not found")

    def get_station(self, station_id: int) -> ChargingStation:
        """Get charging station by ID"""
        for station in self.charging_stations:
            if station.station_id == station_id:
                return station
        raise ValueError(f"Station {station_id} not found")

    def calculate_energy_consumption(self, distance: float) -> float:
        """Calculate energy consumption for a given distance"""
        return distance * self.energy_consumption_rate

    def __repr__(self) -> str:
        return f"ProblemData(tasks={len(self.tasks)}, " \
               f"stations={len(self.charging_stations)}, " \
               f"agvs={self.num_agvs})"
