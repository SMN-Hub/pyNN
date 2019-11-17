import numpy as np
import tensorflow as tf
from mlscanner.char_trainer import features, model_file


class CharInterpreter:
    def __init__(self):
        self.model = tf.keras.models.load_model(model_file)

    def predict(self, x):
        predictions = self.model.predict(x)
        print('prediction', predictions)
        n = np.argmax(predictions)
        letter = features[n]
        return letter
