import numpy as np
import tensorflow as tf

from mlscanner.char_trainer import features, model_file


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
            letters += features[n]
            scores.append("\"" + features[n] + "\": " + str(prediction[n]))
        print(scores)
        return letters
