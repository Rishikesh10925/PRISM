"""Unified class list every source dataset's annotations get mapped onto.

Classes 0-3 are road-damage classes (used in detection training and severity scoring).
Classes 4-5 are auxiliary road-surface classes, kept only to estimate the road-surface
area a pothole's area ratio is normalized against (see blueprint Section 5.2, cue `a`).
"""

CLASSES = [
    "pothole",
    "crack_longitudinal",
    "crack_transverse",
    "crack_alligator",
    "road_surface",
    "footpath",
]

NAME_TO_ID = {name: idx for idx, name in enumerate(CLASSES)}

# Per-source label -> unified class name. Extend this as new sources are added.
SOURCE_LABEL_MAPS = {
    "rdd2022": {
        "D00": "crack_longitudinal",
        "D10": "crack_transverse",
        "D20": "crack_alligator",
        "D40": "pothole",
        # RDD2022 also defines a handful of rarer codes (D01, D11, D43, D44, D50 ...)
        # depending on country; unmapped codes are dropped with a warning rather than
        # silently miscategorized (see convert_voc.py).
    },
    "indian_roads": {
        "pothole": "pothole",
        "road": "road_surface",
        "footpath": "footpath",
        # "shallow" (shallow water/puddle) has no equivalent in our class set yet and
        # is dropped for now — flagged in docs/phase2/02_annotation_standardization.md.
    },
    "roboflow_pothole": {
        "pothole": "pothole",
    },
}


def map_label(source: str, raw_label: str) -> str | None:
    """Return the unified class name for a raw per-source label, or None if unmapped."""
    return SOURCE_LABEL_MAPS.get(source, {}).get(raw_label)
