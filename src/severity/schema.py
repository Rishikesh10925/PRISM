"""Shared types for severity cue modules."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SeverityCues:
    """The three raw cues fusion.py combines into a Severity Score, plus which depth
    source produced `depth` (useful for the Task 3 fallback-vs-MiDaS agreement check)."""

    area_ratio: float  # geometric.py — a
    depth: float  # depth_proxy.py (MiDaS) or shadow_heuristic.py fallback — d
    irregularity: float  # irregularity.py — i
    depth_source: str  # "midas" or "shadow_heuristic"
