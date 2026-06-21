import numpy as np
import cv2
import time
import sys
import os
import tensorflow as tf
from typing import Dict, Any, List, Optional

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from models.enhancer import AdaptiveZeroDCE
from models.detector import TrafficDetector
from models.classifier import HelmetClassifier, SeatbeltClassifier
from models.alpr import ALPR

from rules.helmet_rule import HelmetRule
from rules.seatbelt_rule import SeatbeltRule
from rules.triple_riding import TripleRidingRule
from rules.stopline_redlight import StoplineRedlightRule
from rules.wrong_side import WrongSideRule
from rules.illegal_parking import IllegalParkingRule

sys.path.insert(0, os.path.dirname(__file__))


class VisiTrafficPipeline:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.enhancer = AdaptiveZeroDCE()
        self.detector = TrafficDetector(num_classes=5)
        self.helmet_classifier = HelmetClassifier()
        self.seatbelt_classifier = SeatbeltClassifier()

        self.rules = [
            HelmetRule({"classifier": self.helmet_classifier}),
            SeatbeltRule({"classifier": self.seatbelt_classifier}),
            TripleRidingRule(),
            StoplineRedlightRule(),
            WrongSideRule(),
            IllegalParkingRule(),
        ]
        self.alpr = ALPR()

    def preprocess(self, image_bgr: np.ndarray, enhance: bool = True) -> np.ndarray:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        if enhance:
            tensor = tf.convert_to_tensor(rgb, dtype=tf.float32) / 255.0
            tensor = tf.expand_dims(tensor, 0)
            enhanced = self.enhancer(tensor)[0]
            enhanced_np = (enhanced.numpy() * 255).astype(np.uint8)
            return enhanced_np
        return rgb

    def detect_objects(self, image_rgb: np.ndarray) -> Dict[str, np.ndarray]:
        return self.detector.detect(image_rgb)

    def apply_rules(self, frame_bgr: np.ndarray, detections: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
        all_violations = []
        for rule in self.rules:
            try:
                violations = rule.evaluate(frame_bgr, detections)
                all_violations.extend(violations)
            except Exception as e:
                print(f"Rule {rule.__class__.__name__} failed: {e}")
        return all_violations

    def read_plates(self, frame_bgr: np.ndarray, violations: List[Dict[str, Any]]):
        for v in violations:
            try:
                x1, y1, x2, y2 = [int(c) for c in v['bbox']]
                vehicle_crop = frame_bgr[max(0, y1):y2, max(0, x1):x2]
                if vehicle_crop.size > 0:
                    v['license_plate'] = self.alpr.extract(vehicle_crop)
                else:
                    v['license_plate'] = ""
            except Exception:
                v['license_plate'] = ""
        return violations

    def annotate_frame(self, frame_bgr: np.ndarray, violations: List[Dict[str, Any]]) -> np.ndarray:
        annotated = frame_bgr.copy()
        for v in violations:
            x1, y1, x2, y2 = [int(c) for c in v['bbox']]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"{v['type']} ({v['confidence']:.2f})"
            if v.get('license_plate'):
                label += f" [{v['license_plate']}]"
            cv2.putText(annotated, label, (x1, max(y1 - 10, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        return annotated

    def process_frame(self, image_bgr: np.ndarray, camera_id: str = "cam_01",
                      timestamp: Optional[str] = None) -> Dict[str, Any]:
        if timestamp is None:
            timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")

        enhanced_rgb = self.preprocess(image_bgr, enhance=True)
        detections = self.detect_objects(enhanced_rgb)
        # model.predict() already returns decoded (N,) detections – no flatten needed
        violations = self.apply_rules(image_bgr, detections)
        violations = self.read_plates(image_bgr, violations)
        annotated = self.annotate_frame(image_bgr, violations)

        return {
            "camera_id": camera_id,
            "timestamp": timestamp,
            "num_violations": len(violations),
            "violations": violations,
            "annotated_frame": annotated,
        }


if __name__ == "__main__":
    print("Initialising VisiTraffic pipeline (this may load weights)...")
    pipe = VisiTrafficPipeline()
    dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    print("Processing dummy frame...")
    result = pipe.process_frame(dummy_img)
    print(f"Done. Found {result['num_violations']} violation(s).")
    for v in result['violations']:
        print(f"  - {v['type']} (conf {v['confidence']:.2f})")
    print("Pipeline test completed successfully.")