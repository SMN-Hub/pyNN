import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D

from mlscanner.splitchar.char_trainer import set_random_seed
from mlscanner.yolo.font_yolo_generator import YoloConfiguration


# https://github.com/zzh8829/yolov3-tf2
class YoloTrainer:
    def __init__(self, conf: YoloConfiguration, seed):
        set_random_seed(seed)
        self.conf = conf
        kernel = conf.step
        self.model = tf.keras.Sequential([
            # Adds a Convolutions layer followed by max pool:
            Conv2D(16, kernel, padding='same', activation='relu', input_shape=conf.input_shape),  # size=(18, 18*18, 1) # 18*6 cells
            MaxPooling2D(pool_size=(3, 3)),  # size=(6, 6*18, 16)
            Conv2D(32, kernel, padding='same', activation='relu'),
            MaxPooling2D(pool_size=(3, 3), strides=(3, 1)),  # size=(2, 6*18, 32)
            Conv2D(64, kernel, padding='same', activation='relu'),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 1)),  # size=(1, 6*18, 64)
            # Convolution implementation of sliding window:
            Conv2D(conf.feature_size, 1, padding='same', activation='relu'),
            Conv2D(conf.feature_size, 1, padding='same', activation='relu'),
        ])
