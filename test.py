import tensorflow as tf

# Print TensorFlow version
print("TensorFlow version:", tf.__version__)

# Check if GPU is available
print("GPU devices:", tf.config.list_physical_devices('GPU'))

# Simple TensorFlow computation to test GPU
a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
c = tf.matmul(a, b)
print("Matrix multiplication result:\n", c)
