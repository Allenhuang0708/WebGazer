import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Model
from keras.layers import Flatten, Concatenate
from torch import eye

eyes_model_group1 = keras.Sequential(
    [
        keras.Input(shape=(224, 224, 3)),
        layers.Conv2D(96, kernel_size=11, strides=4, padding='valid', activation='relu'),
        layers.MaxPooling2D(pool_size=3, strides=2)
     ]
)
eyes_model_group2 = keras.Sequential(
    [
        keras.Input(shape=(26, 26, 48)),
        layers.Conv2D(128, kernel_size=5, strides=1, padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=3, strides=2)
    ]
)
eyes_model_group3 = keras.Sequential(
    [
        keras.Input(shape=(12, 12, 256)),
        layers.Conv2D(384, kernel_size=3, strides=1, padding='same', activation='relu'),
        layers.Conv2D(64, kernel_size=1, strides=1, padding='valid', activation='relu')  
    ]
)

face_model_group1 = keras.Sequential(
    [
        keras.Input(shape=(224, 224, 3)),
        layers.Conv2D(96, kernel_size=11, strides=4, padding='valid', activation='relu'),
        layers.MaxPooling2D(pool_size=3, strides=2)
     ]
)
face_model_group2 = keras.Sequential(
    [
        keras.Input(shape=(26, 26, 48)),
        layers.Conv2D(128, kernel_size=5, strides=1, padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=3, strides=2)
    ]
)
face_model_group3 = keras.Sequential(
    [
        keras.Input(shape=(12, 12, 256)),
        layers.Conv2D(384, kernel_size=3, strides=1, padding='same', activation='relu'),
        layers.Conv2D(64, kernel_size=1, strides=1, padding='valid', activation='relu'),
        Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu')
    ]
)

face_grid_model = keras.Sequential(
    [
        layers.Dense(256, input_shape=(625, ), activation='relu'),
        layers.Dense(128, activation='relu')
    ]
)

# concatenate each eye model's output
eye_conect_model = keras.Sequential(
    [
        layers.Dense(128, input_shape=(2*12*12*64, ), activation='relu')
    ]
)

# concatenate all model's output as last model's input
full_connect_model = keras.Sequential(
    [
        layers.Dense(128, input_shape=(128+64+128, ), activation='relu')
    ]
)

dense_model = keras.Sequential(
    [
        layers.Dense(2, input_shape=(128,))
    ]
)
