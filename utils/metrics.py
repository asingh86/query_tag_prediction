import tensorflow as tf
import numpy as np


def categorical_accuracy(y_test, y_pred) -> float:
    m = tf.keras.metrics.CategoricalAccuracy()
    m.update_state(y_test, y_pred)
    return m.result().numpy()
