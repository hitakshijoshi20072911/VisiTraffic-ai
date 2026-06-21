# rules/illegal_parking.py
"""
Illegal parking rule.
Flags a vehicle that is stationary inside a no‑parking zone.
For demo, we use a static zone and assume the vehicle hasn't moved (simplified).
"""

import numpy as np
from typing import List, Dict, Any
from rules.base_rule import BaseRule

class IllegalParkingRule(BaseRule):
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        # Example no‑parking polygon (list of (x,y) points)
        self.no_parking_zone = [(300, 300), (500, 300), (500, 500), (300, 500)]
        self.vehicle_classes = [1, 3]  # cars, trucks

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
            if self._point_in_polygon(cx, cy, self.no_parking_zone):
                violations.append({
                    'type': 'illegal_parking',
                    'confidence': 0.7,
                    'bbox': box.tolist(),
                    'detail': 'Vehicle parked illegally'
                })
        return violations
        
    def _point_in_polygon(self, x, y, poly):
        n = len(poly)
        inside = False
        p1x, p1y = poly[0]
        for i in range(n+1):
            p2x, p2y = poly[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside