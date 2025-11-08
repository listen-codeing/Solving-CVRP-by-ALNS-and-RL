"""
Data generator for AGV scheduling problem instances
"""

import numpy as np
import random
from typing import Tuple, Optional

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agv_model import ProblemData, Task, ChargingStation


def generate_agv_instance(num_tasks: int = 20,
                         num_agvs: int = 5,
                         num_stations: int = 3,
                         map_width: float = 100.0,
                         map_height: float = 100.0,
                         seed: Optional[int] = None) -> ProblemData:
    """
    Generate random AGV scheduling problem instance

    Args:
        num_tasks: Number of transport tasks
        num_agvs: Number of AGVs
        num_stations: Number of charging stations
        map_width: Map width
        map_height: Map height
        seed: Random seed for reproducibility

    Returns:
        ProblemData instance
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Generate tasks
    tasks = []
    for i in range(num_tasks):
        # Random start and end locations
        start_x = random.uniform(0, map_width)
        start_y = random.uniform(0, map_height)
        end_x = random.uniform(0, map_width)
        end_y = random.uniform(0, map_height)

        # Time window
        earliest_start = random.uniform(0, 50)
        time_window_size = random.uniform(20, 50)
        latest_start = earliest_start + time_window_size

        # Service time proportional to distance
        distance = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
        service_time = distance / 10.0  # Assume speed of 10 units/time

        # Priority
        priority = random.randint(1, 5)

        task = Task(
            task_id=i,
            start_location=(start_x, start_y),
            end_location=(end_x, end_y),
            earliest_start=earliest_start,
            latest_start=latest_start,
            service_time=service_time,
            priority=priority
        )
        tasks.append(task)

    # Generate charging stations (distributed across map)
    stations = []
    for i in range(num_stations):
        # Distribute stations evenly
        if num_stations == 1:
            loc_x, loc_y = map_width / 2, map_height / 2
        elif num_stations == 2:
            loc_x = (i + 0.5) * map_width / num_stations
            loc_y = map_height / 2
        elif num_stations == 3:
            # Triangle layout
            if i == 0:
                loc_x, loc_y = map_width / 2, map_height / 6
            elif i == 1:
                loc_x, loc_y = map_width / 6, map_height * 5 / 6
            else:
                loc_x, loc_y = map_width * 5 / 6, map_height * 5 / 6
        else:
            # Random but spread out
            loc_x = (i % int(np.sqrt(num_stations)) + 0.5) * map_width / int(np.sqrt(num_stations))
            loc_y = (i // int(np.sqrt(num_stations)) + 0.5) * map_height / int(np.sqrt(num_stations))

        station = ChargingStation(
            station_id=i,
            location=(loc_x, loc_y),
            capacity=max(2, num_agvs // num_stations),  # Capacity based on AGVs
            charging_rate=0.2  # 20% charge per time unit
        )
        stations.append(station)

    # Create problem data
    problem = ProblemData(
        tasks=tasks,
        charging_stations=stations,
        num_agvs=num_agvs,
        battery_capacity=100.0,
        energy_consumption_rate=0.01,  # 1% per distance unit
        map_width=map_width,
        map_height=map_height
    )

    return problem


if __name__ == "__main__":
    # Test data generation
    problem = generate_agv_instance(num_tasks=20, num_agvs=5, num_stations=3, seed=42)
    print(problem)
    print(f"Generated {len(problem.tasks)} tasks")
    print(f"Generated {len(problem.charging_stations)} charging stations")
    print(f"Distance matrix shape: {problem.distance_matrix.shape}")
