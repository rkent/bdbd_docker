# Loads a model's weights into the cache
import tensorflow
import tensorflow.keras as keras

from tensorflow.keras.applications.resnet_v2 import ResNet50V2
model = ResNet50V2(weights="imagenet")
