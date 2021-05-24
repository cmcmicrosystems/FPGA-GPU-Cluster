
# Check Tensorflow and GPU availability
import sys
import tensorflow.keras
import pandas as pd
import sklearn as sk
import tensorflow as tf

print(f"- TensorFlow: {tf.__version__}")
print(f"- Keras: {tensorflow.keras.__version__}")
print(f"- Python {sys.version}")
print(f"- Pandas {pd.__version__}")
print(f"- Scikit-Learn {sk.__version__}")
gpu = len(tf.config.list_physical_devices('GPU'))>0
print("- GPU is", "available" if gpu else "NOT AVAILABLE")
