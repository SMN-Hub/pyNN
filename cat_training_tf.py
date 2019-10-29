import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from cat_training import load_data


def main():
    print(tf.__version__)
    train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
    print(train_x_orig.shape)
    # Standardize data to have feature values between 0 and 1.
    train_x = train_x_orig / 255.
    test_x = test_x_orig / 255.

    model = tf.keras.Sequential([
        # Adds a densely-connected layer with 20 units to the model:
        Conv2D(64, 3, padding='same', activation='relu', input_shape=(64, 64, 3)),
        Dropout(0.2),
        # Add another:
        Conv2D(12, 3, padding='same', activation='relu'),
        Dropout(0.2),
        # Add another:
        Flatten(),
        Dense(20, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        Dense(20, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        # Add a sigmoid layer with 1 output unit:
        Dense(1, activation='sigmoid')
        ])
    # compile
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # train
    model.fit(train_x, train_y.T, epochs=10)  # 0.856
    # evaluate on test set
    model.evaluate(test_x, test_y.T)  # 0.8
    # C_20 + DO + C7  +DO + H5 + H1 => E20 0.8
    # C_64 + DO + C12 +DO + H20 + H20 + H1 => E20 0.7 / E10 0.9


if __name__ == "__main__":
    main()