import numpy as np
import keras_cv
import tensorflow as tf
from typing import Dict

class TrafficDetector:
    def __init__(self, num_classes: int = 5):
        self.num_classes = num_classes
        backbone = keras_cv.models.YOLOV8Backbone.from_preset("yolo_v8_m_backbone_coco")
        self.model = keras_cv.models.YOLOV8Detector(
            num_classes=num_classes,
            bounding_box_format="xyxy",
            backbone=backbone,
            fpn_depth=1,
        )

    def detect(self, images: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Run decoded inference on a single image (H,W,3) uint8.
        Returns flat numpy arrays: boxes (N,4), classes (N,), confidence (N,).
        """
        if images.dtype == np.uint8:
            images = images.astype(np.float32) / 255.0
        if images.ndim == 3:
            images = np.expand_dims(images, 0)

        # model.predict() returns decoded boxes + class probabilities (N, num_classes)
        preds = self.model.predict(images, verbose=0)

        boxes = preds["boxes"]          # shape (1, N, 4)  or (N, 4)
        class_probs = preds["classes"]  # shape (1, N, num_classes) or (N, num_classes)

        # Remove batch dimension if present
        if boxes.ndim == 3:
            boxes = boxes[0]
            class_probs = class_probs[0]

        # Convert class probabilities → integer class IDs and confidence
        if class_probs.size > 0:
            class_ids = np.argmax(class_probs, axis=1).astype(np.int32)
            conf = np.max(class_probs, axis=1).astype(np.float32)
        else:
            class_ids = np.array([], dtype=np.int32)
            conf = np.array([], dtype=np.float32)

        return {
            "boxes": boxes.astype(np.float32),
            "classes": class_ids,
            "confidence": conf,
        }