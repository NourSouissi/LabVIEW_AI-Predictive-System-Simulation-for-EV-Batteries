import tensorflow as tf
print(tf.__version__)
print("TensorFlow is installed successfully!")

# Check for GPU support
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT available")