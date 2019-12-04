import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D

from mlscanner.font_generator import FULL_SIZE, list_fonts
from mlscanner.splitchar.char_trainer import set_random_seed, FEATURES
from mlscanner.yolo.font_yolo_generator import YoloConfiguration, YoloDatasetGenerator

MODEL_FILE = '../datasets/yolo_model.h5'
TRAINING_FILE = '../datasets/words_manual.txt'
print('Tensorflow version:', tf.__version__)


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

    def get_loss_fn(self, score_thresh=0.5):
        cce = tf.keras.losses.CategoricalCrossentropy()
        bce = tf.keras.losses.BinaryCrossentropy()

        def yolo_loss(y_true, y_pred):
            # y_pred: (batch_size, grid, grid, (pc, ...cls, x, w)) = N x 1 x 105 x 77
            pc_pred, cls_pred, pos_pred, width_pred = tf.split(y_pred, (1, self.conf.classes, 1, 1), axis=-1)
            pc_true, cls_true, pos_true, width_true = tf.split(y_true, (1, self.conf.classes, 1, 1), axis=-1)
            # calculate all masks
            # obj_mask = tf.squeeze(pc_true, -1)
            # calculate all losses
            pc_loss = bce(pc_true, pc_pred)
            cls_loss = cce(cls_true, cls_pred)
            # cls_loss = obj_mask * cce(cls_true, cls_pred)
            pos_loss = bce(pos_true, pos_pred)
            width_loss = bce(width_true, width_pred)
            # sum over (batch, gridx, gridy, anchors) => (batch, 1)
            # pc_loss = tf.reduce_sum(pc_loss, axis=(1, 2, 3))
            # cls_loss = tf.reduce_sum(cls_loss, axis=(1, 2, 3))
            # pos_loss = tf.reduce_sum(pos_loss, axis=(1, 2, 3))
            # width_loss = tf.reduce_sum(width_loss, axis=(1, 2, 3))
            return pc_loss + cls_loss + pos_loss + width_loss

        return yolo_loss

    def fit(self):
        with open(TRAINING_FILE, 'r') as f:
            training_phrases = f.read()
        gen = YoloDatasetGenerator(self.conf)
        data = tf.data.Dataset.from_generator(gen.get_dataset_generator(training_phrases, list_fonts(), augment=False), (tf.int32, tf.int32), (tf.TensorShape(list(self.conf.input_shape)), tf.TensorShape([self.conf.feature_size])))
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
        model.compile(optimizer='adam', loss=self.get_loss_fn(), metrics=['accuracy'])
        model.summary()
        # train
        model.fit(train_dataset.batch(64), epochs=50)
        # evaluate on test set
        print("Evaluating test set")
        outs = model.evaluate(test_dataset.batch(64))
        print("loss:", outs[0])
        print("acc:", outs[1])
        # Save model + parameters
        model.save(MODEL_FILE)


def main():
    conf = YoloConfiguration(FEATURES, (FULL_SIZE, 18 * FULL_SIZE, 1), 3)
    train = YoloTrainer(conf, 2)
    train.fit()


if __name__ == "__main__":
    main()
