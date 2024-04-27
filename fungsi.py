import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout,LeakyReLU
import cvnn.layers as complex_layers
from tensorflow.keras import datasets, layers, models
from keras.preprocessing.image import ImageDataGenerator
import cvnn.layers as complex_layers
import numpy as np

def make_model():
  model = models.Sequential([
    complex_layers.ComplexConv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3), dtype=np.float32),
    complex_layers.ComplexMaxPooling2D((2,2), dtype=np.float32),
    complex_layers.ComplexConv2D(64, (3,3), activation='relu', dtype=np.float32),
    complex_layers.ComplexMaxPooling2D((2,2), dtype=np.float32),
    complex_layers.ComplexConv2D(128, (3,3), activation='relu', dtype=np.float32),
    complex_layers.ComplexMaxPooling2D((2,2), dtype=np.float32),
    complex_layers.ComplexConv2D(128, (3,3), activation='relu', dtype=np.float32),
    complex_layers.ComplexMaxPooling2D((2,2), dtype=np.float32),
    complex_layers.ComplexFlatten(),
    complex_layers.ComplexDense(512, activation='relu', dtype=np.float32),
    complex_layers.ComplexDense(4, activation='softmax', dtype=np.float32)
  ])

  return model