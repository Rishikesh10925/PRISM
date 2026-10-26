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
    # True (default): d is the dimensionless scale-normalized value (MiDaS differential /
    # road inverse-depth spread, or Weber shadow contrast). False: legacy raw units (MiDaS
    # inverse-depth units, or 0-255 gray levels). Either way it is RELATIVE depth evidence,
    # not a physical depth in cm.
    depth_normalized: bool = True
