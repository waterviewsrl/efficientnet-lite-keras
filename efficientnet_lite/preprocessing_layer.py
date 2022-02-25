import tensorflow as tf


def get_preprocessing_layer():
    """Return preprocessing layer for EfficientNet Lite variants."""

    def f(x):
        return (x/127.5) - 1.0

    return f
