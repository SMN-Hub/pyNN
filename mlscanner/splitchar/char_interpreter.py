import numpy as np
import tensorflow as tf

from mlscanner.splitchar.char_trainer import FEATURES, model_file


class CharInterpreter:
    def __init__(self):
        self.model = tf.keras.models.load_model(model_file)

    def predict(self, x):
        predictions = self.model.predict(x, 32)
        # print('prediction', predictions)
        letters = ""
        scores = []
        for prediction in predictions:
            n = np.argmax(prediction)
            letters += FEATURES[n]
            scores.append(FEATURES[n] + ": " + str(prediction[n]))
        print(scores)
        return letters
