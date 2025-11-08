"""
GraphConverter: Transform AGV solutions to graph representations for GNN

Converts Solution objects to PyTorch Geometric Data objects with 15-dimensional
node features tailored for AGV charging scheduling problems.
"""

import torch
from torch_geometric.data import Data
from typing import List, Tuple, Dict
import numpy as np

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agv_model import Solution, Event, EventType, AGVPath, ProblemData


class AGVGraphConverter:
    """
    Convert AGV scheduling solutions to graph representations

    Node types:
    1. Task event nodes
    2. Charging event nodes
    3. Center node (global state aggregation)

    Edge types:
    1. Sequential edges (within AGV route)
    2. Center edges (all nodes to center)
    """

    def __init__(self, problem_data: ProblemData):
        """
        Initialize converter

        Args:
            problem_data: Problem instance data
        """
        self.problem = problem_data
        self.num_tasks = len(problem_data.tasks)
        self.num_stations = len(problem_data.charging_stations)

    def solution_to_graph(self, solution: Solution) -> Data:
        """
        Convert Solution to PyG Data object

        Args:
            solution: AGV scheduling solution

        Returns:
            PyG Data with 15-dimensional node features
        """
        features = []
        node_map = {}  # (agv_id, event_idx) -> node_id
        node_id = 0

        # 1. Create nodes for each event in each AGV
        for agv in solution.agv_paths:
            for event_idx, event in enumerate(agv.events):
                feature = self._extract_event_feature(agv, event_idx, event, solution)
                features.append(feature)
                node_map[(agv.agv_id, event_idx)] = node_id
                node_id += 1

        # 2. Create center node
        center_feature = self._extract_center_feature(solution)
        features.append(center_feature)
        center_node_id = node_id

        # 3. Build edges
        edges = []

        # 3a. Sequential edges within AGV routes
        for agv in solution.agv_paths:
            for i in range(len(agv.events) - 1):
                node1 = node_map[(agv.agv_id, i)]
                node2 = node_map[(agv.agv_id, i + 1)]
                edges.append([node1, node2])
                edges.append([node2, node1])  # Bidirectional

        # 3b. Center node edges (all nodes connect to center)
        for nid in range(center_node_id):
            edges.append([nid, center_node_id])
            edges.append([center_node_id, nid])

        # 4. Handle edge case: no events (empty solution)
        if not edges:
            edges = [[0, 0]]  # Self-loop to avoid empty edge_index

        # 5. Construct PyG Data
        graph = Data(
            x=torch.tensor(features, dtype=torch.float),
            edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous(),
            center_node_index=torch.tensor([center_node_id], dtype=torch.long),
            state=solution,
            cost=torch.tensor([solution.total_cost], dtype=torch.float)
        )

        return graph

    def _extract_event_feature(self, agv: AGVPath, event_idx: int,
                               event: Event, solution: Solution) -> List[float]:
        """
        Extract 15-dimensional feature vector for an event

        Feature dimensions:
        1-3:   Structural features (position, AGV ID, event type)
        4-6:   Temporal features (start time, duration, waiting time)
        7-9:   Energy features (SOC before, after, change)
        10-12: Spatial features (X coord, Y coord, distance/travel)
        13-15: Charging station features (usage, is_charging, related_distance)

        Args:
            agv: AGV path containing the event
            event_idx: Index of event in AGV's event sequence
            event: The event
            solution: Complete solution (for global context)

        Returns:
            15-dimensional feature vector
        """
        features = []

        # === Structural Features (3 dims) ===
        # F1: Position in route (normalized)
        features.append(event_idx / max(len(agv.events), 1))

        # F2: AGV ID (normalized)
        features.append(agv.agv_id / max(len(solution.agv_paths), 1))

        # F3: Event type (1 if charging, 0 if task)
        features.append(1.0 if event.is_charging() else 0.0)

        # === Temporal Features (3 dims) ===
        # F4: Start time (normalized)
        features.append(event.start_time / max(self.problem.time_horizon, 1.0))

        # F5: Duration (normalized)
        features.append(event.duration / max(self.problem.max_task_duration, 1.0))

        # F6: Waiting time (normalized)
        features.append(event.waiting_time / max(self.problem.max_waiting_time, 1.0))

        # === Energy Features (3 dims) ===
        # F7: SOC before event
        soc_before = agv.soc_before_events[event_idx] if event_idx < len(agv.soc_before_events) else 0.5
        features.append(soc_before)

        # F8: SOC after event
        soc_after = agv.soc_after_events[event_idx] if event_idx < len(agv.soc_after_events) else 0.5
        features.append(soc_after)

        # F9: SOC change
        features.append(soc_before - soc_after)

        # === Spatial Features (3 dims) ===
        if event.is_task():
            task = self.problem.get_task(event.event_id)
            # F10: X coordinate (normalized by map width)
            features.append((task.start_location[0] + task.end_location[0]) / 2 / max(self.problem.map_width, 1.0))
            # F11: Y coordinate (normalized by map height)
            features.append((task.start_location[1] + task.end_location[1]) / 2 / max(self.problem.map_height, 1.0))
            # F12: Task distance (normalized)
            features.append(task.distance / max(self.problem.max_distance, 1.0))
        else:
            # Charging event
            station = self.problem.get_station(event.event_id)
            # F10: X coordinate
            features.append(station.location[0] / max(self.problem.map_width, 1.0))
            # F11: Y coordinate
            features.append(station.location[1] / max(self.problem.map_height, 1.0))
            # F12: No travel distance for charging
            features.append(0.0)

        # === Charging Station Features (3 dims) ===
        if event.is_charging():
            # F13: Station usage ratio
            station_id = event.event_id
            solution.update_charging_station_usage()
            usage = len(solution.charging_station_usage.get(station_id, []))
            capacity = self.problem.get_station_capacity(station_id)
            features.append(usage / max(capacity, 1))

            # F14: Is charging event
            features.append(1.0)

            # F15: Charging duration (normalized)
            features.append(event.duration / max(self.problem.max_charging_time, 1.0))
        else:
            # Task event
            # F13: Distance to nearest charging station (normalized)
            task = self.problem.get_task(event.event_id)
            nearest_dist = self._get_nearest_station_distance(task)
            features.append(nearest_dist / max(self.problem.max_distance, 1.0))

            # F14: Not a charging event
            features.append(0.0)

            # F15: Energy urgency (inverse of SOC after)
            features.append(1.0 - soc_after)

        return features

    def _extract_center_feature(self, solution: Solution) -> List[float]:
        """
        Extract 15-dimensional feature vector for center node (global state)

        Args:
            solution: Complete solution

        Returns:
            15-dimensional feature vector representing global state
        """
        features = []

        # === Structural Features (3 dims) ===
        # F1: Number of AGVs (normalized)
        features.append(len(solution.agv_paths) / max(self.problem.num_agvs, 1))

        # F2: Total number of events (normalized)
        total_events = sum(len(agv.events) for agv in solution.agv_paths)
        features.append(total_events / max(self.num_tasks * 2, 1))  # *2 for tasks + charging

        # F3: Unassigned tasks ratio
        features.append(len(solution.unassigned_tasks) / max(self.num_tasks, 1))

        # === Temporal Features (3 dims) ===
        # F4: Average makespan (normalized)
        avg_makespan = solution.get_makespan() / len(solution.agv_paths) if solution.agv_paths else 0.0
        features.append(avg_makespan / max(self.problem.time_horizon, 1.0))

        # F5: Makespan variance
        if solution.agv_paths:
            makespans = [agv.makespan for agv in solution.agv_paths]
            variance = np.var(makespans)
            features.append(min(variance / max(self.problem.time_horizon, 1.0), 1.0))
        else:
            features.append(0.0)

        # F6: Average waiting time (normalized)
        total_waiting = sum(agv.total_waiting_time for agv in solution.agv_paths)
        avg_waiting = total_waiting / len(solution.agv_paths) if solution.agv_paths else 0.0
        features.append(avg_waiting / max(self.problem.max_waiting_time, 1.0))

        # === Energy Features (3 dims) ===
        # F7: Average SOC (across all AGVs at end of routes)
        if solution.agv_paths:
            avg_soc = np.mean([agv.soc_after_events[-1] if agv.soc_after_events else 0.5
                              for agv in solution.agv_paths])
            features.append(avg_soc)
        else:
            features.append(0.5)

        # F8: Min SOC (lowest SOC across all events)
        if solution.agv_paths:
            all_socs = []
            for agv in solution.agv_paths:
                all_socs.extend(agv.soc_after_events)
            min_soc = min(all_socs) if all_socs else 0.5
            features.append(min_soc)
        else:
            features.append(0.5)

        # F9: Total charging events
        total_charging = sum(len(agv.get_charging_sequence()) for agv in solution.agv_paths)
        features.append(total_charging / max(len(solution.agv_paths) * 2, 1))

        # === Spatial Features (3 dims) ===
        # F10-F11: Average AGV location (X, Y)
        if solution.agv_paths and any(agv.events for agv in solution.agv_paths):
            avg_x = avg_y = 0.0
            count = 0
            for agv in solution.agv_paths:
                if agv.events:
                    last_event = agv.events[-1]
                    if last_event.is_task():
                        task = self.problem.get_task(last_event.event_id)
                        avg_x += task.end_location[0]
                        avg_y += task.end_location[1]
                    else:
                        station = self.problem.get_station(last_event.event_id)
                        avg_x += station.location[0]
                        avg_y += station.location[1]
                    count += 1
            if count > 0:
                features.append(avg_x / count / max(self.problem.map_width, 1.0))
                features.append(avg_y / count / max(self.problem.map_height, 1.0))
            else:
                features.append(0.5)
                features.append(0.5)
        else:
            features.append(0.5)
            features.append(0.5)

        # F12: Average distance traveled
        total_dist = sum(agv.total_travel_time for agv in solution.agv_paths)
        avg_dist = total_dist / len(solution.agv_paths) if solution.agv_paths else 0.0
        features.append(avg_dist / max(self.problem.max_distance * self.num_tasks, 1.0))

        # === Charging Station Features (3 dims) ===
        # F13: Average station utilization
        solution.update_charging_station_usage()
        if solution.charging_station_usage:
            total_usage = sum(len(sessions) for sessions in solution.charging_station_usage.values())
            total_capacity = sum(self.problem.get_station_capacity(s.station_id)
                               for s in self.problem.charging_stations)
            avg_utilization = total_usage / max(total_capacity, 1)
            features.append(min(avg_utilization, 1.0))
        else:
            features.append(0.0)

        # F14: Number of active charging stations
        active_stations = len(solution.charging_station_usage)
        features.append(active_stations / max(self.num_stations, 1))

        # F15: Solution feasibility indicator (0 if feasible, 1 if has violations)
        is_feasible, _ = solution.validate_feasibility()
        features.append(0.0 if is_feasible else 1.0)

        return features

    def _get_nearest_station_distance(self, task) -> float:
        """
        Get distance from task to nearest charging station

        Args:
            task: Task object

        Returns:
            Distance to nearest charging station
        """
        if not self.problem.charging_stations:
            return 0.0

        min_dist = float('inf')
        task_center = (
            (task.start_location[0] + task.end_location[0]) / 2,
            (task.start_location[1] + task.end_location[1]) / 2
        )

        for station in self.problem.charging_stations:
            dist = np.sqrt((task_center[0] - station.location[0])**2 +
                          (task_center[1] - station.location[1])**2)
            min_dist = min(min_dist, dist)

        return min_dist if min_dist != float('inf') else 0.0
