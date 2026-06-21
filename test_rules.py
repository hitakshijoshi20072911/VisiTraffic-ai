"""Quick test for all rule files."""
import numpy as np
import cv2
import sys
sys.path.append('.')   # ensure project root is in path

from rules.helmet_rule import HelmetRule
from rules.seatbelt_rule import SeatbeltRule
from rules.triple_riding import TripleRidingRule
from rules.stopline_redlight import StoplineRedlightRule
from rules.wrong_side import WrongSideRule
from rules.illegal_parking import IllegalParkingRule

# Create a dummy frame (640x640 BGR)
frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

# Create dummy detections: one motorcycle (cls 0) and one car (cls 1)
detections = {
    'boxes': np.array([[100, 100, 300, 400], [350, 200, 600, 550]], dtype=np.float32),
    'classes': np.array([0, 1]),
    'confidence': np.array([0.9, 0.8])
}

rules = [
    HelmetRule(),
    SeatbeltRule(),
    TripleRidingRule(),
    StoplineRedlightRule(),
    WrongSideRule(),
    IllegalParkingRule()
]

for rule in rules:
    violations = rule.evaluate(frame, detections)
    print(f"{rule.__class__.__name__}: found {len(violations)} violation(s)")
    for v in violations:
        print(f"   - {v['type']} (conf {v['confidence']})")

print("All rules executed successfully.")