# rules/seatbelt_rule.py
"""
Seatbelt non‑compliance rule.
Uses MobileNetV3 seatbelt classifier.
"""

import numpy as np
from typing import List, Dict, Any
from rules.base_rule import BaseRule
from models.classifier import SeatbeltClassifier

class SeatbeltRule(BaseRule):
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.classifier = SeatbeltClassifier()
        self.car_class_ids = [1]        # example: car
        self.confidence_threshold = 0.5

    def evaluate(self, frame, detections):
        violations = []
        boxes = detections.get('boxes', [])
        classes = detections.get('classes', [])
        scores = detections.get('confidence', [])

        for i, box in enumerate(boxes):
            cls = int(classes[i])
            conf = float(scores[i])
            if cls not in self.car_class_ids or conf < self.confidence_threshold:
                continue

            x1, y1, x2, y2 = map(int, box)
            driver_crop = frame[y1:y2, x1:x1+(x2-x1)//3]
            seatbelt_prob = self.classifier.predict(driver_crop)
            if seatbelt_prob < 0.5:
                violations.append({
                    'type': 'seatbelt_violation',
                    'confidence': round(1.0 - seatbelt_prob, 3),
                    'bbox': [x1, y1, x2, y2],
                    'detail': 'Driver not wearing seatbelt'
                })
        return violations