from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple


@dataclass
class Pose:
    position_m: Tuple[float, float, float]
    orientation_wxyz: Tuple[float, float, float, float]


@dataclass
class DetectedObject:
    id: str
    label: str
    color: Optional[str]
    pose: Pose
    bbox_xywh: Optional[Tuple[float, float, float, float]]
    confidence: float


def to_json(obj: Any) -> Dict[str, Any]:
    """Convert dataclass (possibly nested) to a plain JSON-serializable dict."""
    return asdict(obj)


