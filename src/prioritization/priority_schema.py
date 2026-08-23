"""Shared types for the prioritization engine."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PriorityWeights:
    alpha: float = 0.4  # severity
    beta: float = 0.3  # road-type
    gamma: float = 0.2  # traffic proxy
    delta: float = 0.1  # recurrence

    def __post_init__(self):
        total = self.alpha + self.beta + self.gamma + self.delta
        if total <= 0:
            raise ValueError("weights must sum to a positive value")


@dataclass
class PriorityInputs:
    severity_score: float  # S, 0-100 (from src/severity/fusion.py)
    road_type_weight: float  # 0-1 (from road_type.py)
    traffic_proxy: float  # 0-1 (from traffic_recurrence.py)
    recurrence_factor: float  # 0-1 (from traffic_recurrence.py)
