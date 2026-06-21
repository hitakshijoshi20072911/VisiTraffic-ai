# rules/triple_riding.py
"""
Triple riding rule: a two‑wheeler should not have more than 2 riders.
"""

import numpy as np
from typing import List, Dict, Any
from rules.base_rule import BaseRule

class TripleRidingRule(BaseRule):
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.two_wheeler_class = 0      # motorcycle
        self.max_riders = 2

    def evaluate(self, frame, detections):
        violations = []
        boxes = detections.get('boxes', [])
        classes = detections.get('classes', [])

        for i, box in enumerate(boxes):
            cls = int(classes[i])
            if cls == self.two_wheeler_class:
                violations.append({
                    'type': 'triple_riding',
                    'confidence': 0.8,
                    'bbox': box.tolist(),
                    'detail': 'Possible triple riding detected'
                })
                break
        return violations