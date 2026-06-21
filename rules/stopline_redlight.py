# rules/stopline_redlight.py
"""
Stop line / red light violation rule.
Checks if a vehicle's bounding box overlaps with a predefined stop‑line zone.
"""

import numpy as np
from typing import List, Dict, Any
from rules.base_rule import BaseRule

class StoplineRedlightRule(BaseRule):
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        # Define a mock stop‑line rectangle (in normalized coordinates or pixel coords)
        # For demo, use a fixed pixel zone (x1,y1,x2,y2) for a 640x640 frame.
        self.stop_line_zone = [100, 500, 540, 540]   # example: horizontal line region
        self.vehicle_classes = [0, 1, 3]   # motorcycle, car, truck

    def evaluate(self, frame, detections):
        violations = []
        boxes = detections.get('boxes', [])
        classes = detections.get('classes', [])

        for i, box in enumerate(boxes):
            cls = int(classes[i])
            if cls not in self.vehicle_classes:
                continue
            iou = self._iou(box, self.stop_line_zone)
            if iou > 0.3:
                violations.append({
                    'type': 'stop_line_violation',
                    'confidence': round(iou, 2),
                    'bbox': box.tolist(),
                    'detail': 'Vehicle crossed stop line'
                })
        return violations

    def _iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou