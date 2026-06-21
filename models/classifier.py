import numpy as np
import tensorflow as tf
from tensorflow import keras

class HelmetClassifier:
    def __init__(self, model_path: str = None):
        if model_path is None:
            base = keras.applications.MobileNetV3Small(
                input_shape=(96, 96, 3),
                include_top=False,
                weights='imagenet'
            )
            x = keras.layers.GlobalAveragePooling2D()(base.output)
            x = keras.layers.Dense(128, activation='relu')(x)
            output = keras.layers.Dense(1, activation='sigmoid')(x)
            self.model = keras.Model(base.input, output)
        else:
            self.model = keras.models.load_model(model_path)

    def predict(self, crop: np.ndarray) -> float:
        if crop.ndim == 3:
            crop = np.expand_dims(crop, 0)          # (1, H, W, 3)
        crop = tf.image.resize(crop, (96, 96)) / 255.0
        prob = self.model(crop, training=False)
        # Ensure prob is a scalar float
        return float(prob.numpy().squeeze())


class SeatbeltClassifier:
    def __init__(self, model_path: str = None):
        base = keras.applications.MobileNetV3Small(
            input_shape=(96, 96, 3),
            include_top=False,
            weights='imagenet'
        )
        x = keras.layers.GlobalAveragePooling2D()(base.output)
        x = keras.layers.Dense(128, activation='relu')(x)
        output = keras.layers.Dense(1, activation='sigmoid')(x)
        self.model = keras.Model(base.input, output)

    def predict(self, crop: np.ndarray) -> float:
        if crop.ndim == 3:
            crop = np.expand_dims(crop, 0)
        crop = tf.image.resize(crop, (96, 96)) / 255.0
        prob = self.model(crop, training=False)
        return float(prob.numpy().squeeze())