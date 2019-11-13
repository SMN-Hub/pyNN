import numpy as np
import random as rn
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

from cat_training import load_data


def set_random_seed(seed):
    np.random.seed(seed)
    rn.seed(seed)
    tf.set_random_seed(seed)


def main():
    # Fix random seed
    set_random_seed(2)
    print('Tensorflow version:', tf.__version__)
    train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
    print('Training set shape:', train_x_orig.shape)
    # Standardize data to have feature values between 0 and 1.
    train_x = train_x_orig / 255.
    test_x = test_x_orig / 255.
    window = 3

    model = tf.keras.Sequential([
        # Adds a densely-connected layer with 20 units to the model:
        Conv2D(16, window, padding='same', activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D(),  # size=(32, 32, 6)
        # Dropout(0.2),
        # Add another:
        Conv2D(32, window, padding='same', activation='relu'),
        MaxPooling2D(),  # size=(16, 16, 16)
        Conv2D(64, window, padding='same', activation='relu'),
        MaxPooling2D(),  # size=(8, 8, 32)
        # Flatten:
        Flatten(),  # size=(2048, )
        # Adds a densely-connected layers:
        Dense(120, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        Dense(84, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        # Add a sigmoid layer with 1 output unit:
        Dense(1, activation='sigmoid')
        ])
    # compile
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # train
    model.fit(train_x, train_y.T, epochs=100)  # 0.856
    # evaluate on test set
    print("Evaluating test set")
    outs = model.evaluate(test_x, test_y.T)  # 0.8
    print("loss:", outs[0])
    print("acc:", outs[1])
    # C_20 + DO + C7  +DO + H5 + H1 => E20 0.8
    # C_64 + DO + C12 +DO + H20 + H20 + H1 => E20 0.7 / E10 0.9


if __name__ == "__main__":
    main()
