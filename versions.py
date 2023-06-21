import tensorflow as tf
import keras
import wandb
import skimage
import scipy
import numpy as np
import pandas as pd
import matplotlib
import cv2
import PIL
import pydicom
import urllib3

# Check GPU support in TensorFlow
print("GPU support in TensorFlow:", tf.test.is_gpu_available())

print("TensorFlow version:", tf.__version__)
print("Keras version:", keras.__version__)
print("wandb version:", wandb.__version__)
print("scikit-image version:", skimage.__version__)
print("scipy version:", scipy.__version__)
print("numpy version:", np.__version__)
print("pandas version:", pd.__version__)
print("matplotlib version:", matplotlib.__version__)
print("OpenCV version:", cv2.__version__)
print("Pillow version:", PIL.__version__)
print("pydicom version:", pydicom.__version__)
print("urllib3 version:", urllib3.__version__)