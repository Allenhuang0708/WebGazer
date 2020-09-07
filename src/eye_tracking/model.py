import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Model
from keras.layers import Flatten, Concatenate
from torch import eye


def keras_lrn(input):
    return tf.nn.local_response_normalization(input, depth_radius=5, bias=1, alpha=0.0001, beta=0.75)

eyes_model = keras.Sequential(
    [
        keras.Input(shape=(224, 224, 3)),
        layers.Conv2D(96, kernel_size=11, strides=4, padding='valid', activation='relu'),
        layers.MaxPooling2D(pool_size=3, strides=2),
        layers.Lambda(keras_lrn),
        layers.Conv2D(256, kernel_size=5, strides=1, padding='same', groups=2, activation='relu'),
        layers.MaxPooling2D(pool_size=3, strides=2),
        layers.Lambda(keras_lrn),
        layers.Conv2D(384, kernel_size=3, strides=1, padding='same', activation='relu'),
        layers.Conv2D(64, kernel_size=1, strides=1, padding='valid', activation='relu')  
     ]
)

face_model = keras.Sequential(
    [
        keras.Input(shape=(224, 224, 3)),
        layers.Conv2D(96, kernel_size=11, strides=4, padding='valid', activation='relu'),
        layers.MaxPooling2D(pool_size=3, strides=2),
        layers.Lambda(keras_lrn),
        layers.Conv2D(256, kernel_size=5, strides=1, padding='same', groups=2, activation='relu'),
        layers.MaxPooling2D(pool_size=3, strides=2),
        layers.Lambda(keras_lrn),
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
