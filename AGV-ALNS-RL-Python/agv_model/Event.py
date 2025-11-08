"""
Event representation for AGV scheduling.

An Event represents either a task execution or a charging activity.
This unified representation simplifies the handling of AGV schedules.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional


class EventType(Enum):
    """Event type enumeration"""
    TASK = "task"           # Task execution
    CHARGING = "charging"   # Charging activity


@dataclass
class Event:
    """
    Unified event record (task or charging)

    Design principles:
    - Uses dataclass for code simplification (cleaner than C++ constructors)
    - Type safety with EventType enum
    - Clear semantics (event_type instead of flags)

    Attributes:
        event_type: Type of event (TASK or CHARGING)
        event_id: Task ID or charging station ID
        start_time: Event start time
        end_time: Event end time
        waiting_time: Waiting time before event starts (default 0.0)
    """
    event_type: EventType
    event_id: int
    start_time: float
    end_time: float
    waiting_time: float = 0.0

    @property
    def duration(self) -> float:
        """Event duration"""
        return self.end_time - self.start_time

    def is_task(self) -> bool:
        """Check if this is a task event"""
        return self.event_type == EventType.TASK

    def is_charging(self) -> bool:
        """Check if this is a charging event"""
        return self.event_type == EventType.CHARGING

    def __repr__(self) -> str:
        return f"{self.event_type.value}_{self.event_id}@{self.start_time:.2f}"
