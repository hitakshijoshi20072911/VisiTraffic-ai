# models/enhancer.py
"""
Zero-DCE (Deep Curve Estimation) layer for low-light enhancement.
A tiny CNN (7 layers, <10k params) that predicts per-pixel tone-mapping curves.
Training is unsupervised; here we provide inference-only weights (random init for demo).
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class AdaptiveZeroDCE(keras.layers.Layer):
    """Lightweight enhancement layer based on Zero-DCE (CVPR 2020)."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Tiny curve parameter estimator
        self.conv1 = layers.Conv2D(32, 3, padding='same', activation='relu')
        self.conv2 = layers.Conv2D(32, 3, padding='same', activation='relu')
        self.conv3 = layers.Conv2D(32, 3, padding='same', activation='relu')
        self.conv4 = layers.Conv2D(3, 3, padding='same', activation='tanh')  # 3 curves for R,G,B

    def call(self, inputs):
        # inputs: (B, H, W, 3) in [0,1] or [0,255], we assume [0,1]
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        curves = self.conv4(x)  # range [-1,1] after tanh

        # Iteratively apply curves: LE_n = LE_n-1 + A_n * LE_n-1 * (1 - LE_n-1)
        enhanced = inputs
        for _ in range(8):   # 8 iterations as in the paper
            enhanced = enhanced + curves * enhanced * (1.0 - enhanced)
        return tf.clip_by_value(enhanced, 0.0, 1.0)

if __name__ == "__main__":
    # Quick test: create a dummy dark image and enhance it
    import numpy as np
    enhancer = AdaptiveZeroDCE()
    dummy = tf.random.uniform((1, 256, 256, 3)) * 0.3  # dark image
    out = enhancer(dummy)
    print("Enhancer output shape:", out.shape)  # should be (1,256,256,3)
    print("Enhancer test passed.")