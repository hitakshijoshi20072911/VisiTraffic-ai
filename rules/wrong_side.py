# rules/wrong_side.py
"""
Wrong‑side driving rule.
For demo, flags vehicles whose center lies inside a manually defined forbidden zone.
"""

import numpy as np
from typing import List, Dict, Any
from rules.base_rule import BaseRule

class WrongSideRule(BaseRule):
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        # Define a region where vehicles should not be (e.g., left side of a one-way)
        self.forbidden_zone = [0, 0, 200, 640]   # x1,y1,x2,y2 (left strip)
        self.vehicle_classes = [0, 1, 3]

    def evaluate(self, frame, detections):
        violations = []
        boxes = detections.get('boxes', [])
        classes = detections.get('classes', [])

        for i, box in enumerate(boxes):
            cls = int(classes[i])
            if cls not in self.vehicle_classes:
                continue
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            if (self.forbidden_zone[0] <= cx <= self.forbidden_zone[2] and
                self.forbidden_zone[1] <= cy <= self.forbidden_zone[3]):
                violations.append({
                    'type': 'wrong_side_driving',
                    'confidence': 0.9,
                    'bbox': box.tolist(),
                    'detail': 'Vehicle on wrong side'
                })
        return violations