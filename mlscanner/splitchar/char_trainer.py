import random as rn

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

from mlscanner.font_generator import chars_dataset_generator

features = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,;:!?'\"-+*/"
model_file = '../datasets/chars_model.h5'
print('Tensorflow version:', tf.__version__)


def set_random_seed(seed):
    np.random.seed(seed)
    rn.seed(seed)
    tf.random.set_seed(seed)


class CharTrainer:
    def __init__(self, seed):
        set_random_seed(seed)
        window = 3
        self.model = tf.keras.Sequential([
            # Adds a Convolutions layer followed by max pool:
            Conv2D(16, window, padding='same', activation='relu', input_shape=(18, 18, 1)),
            MaxPooling2D(),  # size=(9, 9, 16)
            Conv2D(32, window, padding='same', activation='relu'),
            MaxPooling2D(),  # size=(4, 4, 32)
            Conv2D(64, window, padding='same', activation='relu'),
            MaxPooling2D(),  # size=(2, 2, 64)
            # Flatten:
            Flatten(),  # size=(256, )
            # Adds a densely-connected layers:
            Dense(120, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            Dense(84, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            # Add a sigmoid layer with 1 output unit:
            Dense(len(features), activation='sigmoid')  # size=(76, )
        ])

    def fit(self):
        data = tf.data.Dataset.from_generator(chars_dataset_generator(features), (tf.int32, tf.int32), (tf.TensorShape([18,18,1]), tf.TensorShape([len(features)])))
        # Data augmentation
        # Split train/dev/test sets
        test_ratio_percent = 5
        data = data.shuffle(10, reshuffle_each_iteration=False)
        is_test = lambda idx, dt: idx % 100 < test_ratio_percent
        is_train = lambda idx, dt: idx % 100 >= test_ratio_percent
        recover = lambda idx, dt: dt
        test_dataset = data.enumerate().filter(is_test).map(recover)
        train_dataset = data.enumerate().filter(is_train).map(recover)
        # Train model
        model = self.model
        # compile
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.summary()
        # train
        model.fit(train_dataset.batch(64), epochs=50)
        # evaluate on test set
        print("Evaluating test set")
        outs = model.evaluate(test_dataset.batch(64))
        print("loss:", outs[0])
        print("acc:", outs[1])
        # Save model + parameters
        model.save(model_file)


def main():
    train = CharTrainer(2)
    train.fit()


if __name__ == "__main__":
    main()
