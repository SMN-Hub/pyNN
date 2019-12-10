import numpy as np
import tensorflow as tf

from mlscanner.font_generator import FULL_SIZE
from mlscanner.splitchar.char_trainer import FEATURES
from mlscanner.yolo.yolo_configuration import YoloConfiguration, decode_conf, encode_conf
from mlscanner.yolo.yolo_trainer import MODEL_FILE


class YoloCharacter:
    def __init__(self, cls, score, p, w):
        self.cls = cls
        self.score = score
        self.pos = p
        self.width = w

    def iou(self, other):
        box1p = self.pos
        box1w = self.width
        box2p = other.pos
        box2w = other.width
        # box positions
        box11 = box1p - box1w // 2
        box12 = box11 + box1w
        box21 = box2p - box2w // 2
        box22 = box21 + box2w
        # intersection
        xi1 = max(box11, box21)
        xi2 = min(box12, box22)
        iw = xi2 - xi1 if xi2 > xi1 else 0
        # intersection over union
        union = box1w + box2w - iw
        iou = iw / union
        return iou

    def compare(self, other, iou_threshold):
        iou = self.iou(other)
        if iou >= iou_threshold:
            return self.score - other.score
        else:
            return 0


class YoloInterpreter:
    def __init__(self):
        self.model = tf.keras.models.load_model(MODEL_FILE)
        self.conf: YoloConfiguration = decode_conf()

    def predict(self, x, score_threshold, iou_threshold):
        predictions = self.model.predict(x, 32)
        # predictions shape: (batch_size, gridx, gridy, (pc, ...cls, x, w)) = N x 1 x 108 x 77
        batch_size = predictions.shape[0]
        # squeeze gridx dimension
        predictions = predictions.reshape(batch_size, self.conf.grid_count, self.conf.feature_size)
        pc, cls, pos, width = np.split(predictions, (1, self.conf.classes, 1, 1), axis=-1)
        # filter scores lower than score_threshold
        cls_score = cls * pc
        batch_letters = []
        for batch in range(batch_size):
            letters = []
            # filter non max boxes
            for box in range(self.conf.grid_count):
                n = np.argmax(cls_score[batch][box])
                prob = cls_score[batch][box][n]
                if prob >= score_threshold:
                    character = YoloCharacter(n, prob, pos[batch][box], width[batch][box])
                    if len(letters) > 0:
                        # compare with previous
                        cmp = character.compare(letters[-1], iou_threshold)
                        if cmp == 0:
                            letters.append(character)
                        elif cmp > 0:  # replace
                            letters[-1] = character
            batch_letters.append(letters)
        return batch_letters


if __name__ == "__main__":
    conf = YoloConfiguration(FEATURES, (FULL_SIZE, 18 * FULL_SIZE, 1), 3)
    encode_conf(conf)
    conf2 = decode_conf()
    print(conf2)
