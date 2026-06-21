# rules/base_rule.py
"""
Abstract base class for all traffic violation rules.
Each rule must implement `evaluate()` and return a list of violation dicts.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np

class BaseRule(ABC):
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

    @abstractmethod
    def evaluate(self, frame: np.ndarray, detections: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
        """
        Args:
            frame: the original image (H, W, 3) in BGR or RGB.
            detections: dict with keys 'boxes' (N,4), 'classes' (N,), 'confidence' (N,)
                         bounding boxes in xyxy format, classes as integers,
                         confidence scores as floats.
        Returns:
            list of violation dicts, each containing at minimum:
                - 'type': str, e.g. 'helmet_violation'
                - 'confidence': float
                - 'bbox': [x1,y1,x2,y2] of the violating object
                - 'evidence': optional cropped image region
        """
        pass