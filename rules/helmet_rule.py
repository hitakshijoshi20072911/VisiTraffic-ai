# rules/helmet_rule.py
"""
Helmet non‑compliance rule.
Uses the MobileNetV3 helmet classifier from models.classifier.
"""

import numpy as np
from typing import List, Dict, Any
from rules.base_rule import BaseRule
from models.classifier import HelmetClassifier

class HelmetRule(BaseRule):
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        # Load the helmet classifier (dummy for now; real trained model later)
        self.classifier = HelmetClassifier()
        # Mapping: which class IDs represent two‑wheelers / riders
        self.rider_class_ids = [0, 2]  # example: motorcycle and rider
        self.confidence_threshold = 0.5

    def evaluate(self, frame, detections):
        violations = []
        boxes = detections.get('boxes', [])
        classes = detections.get('classes', [])
        scores = detections.get('confidence', [])

        for i, box in enumerate(boxes):
            cls = int(classes[i])              # scalar
            conf = float(scores[i])
            if cls not in self.rider_class_ids or conf < self.confidence_threshold:
                continue

            x1, y1, x2, y2 = map(int, box)
            head_crop = frame[max(0, y1): y2//2, x1:x2]
            helmet_prob = self.classifier.predict(head_crop)
            if helmet_prob < 0.5:
                violations.append({
                    'type': 'helmet_violation',
                    'confidence': round(1.0 - helmet_prob, 3),
                    'bbox': [x1, y1, x2, y2],
                    'detail': 'Rider without helmet'
                })
        return violations