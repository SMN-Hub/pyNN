import os.path
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Conv2D, Input, LeakyReLU, MaxPooling2D, ZeroPadding2D

from mlscanner.font_generator import FULL_SIZE, list_fonts
from mlscanner.splitchar.char_trainer import set_random_seed, FEATURES
from mlscanner.yolo.font_yolo_generator import YoloDatasetGenerator
from mlscanner.yolo.yolo_configuration import YoloConfiguration, decode_conf, encode_conf

MODEL_FILE = '../datasets/yolo_model.h5'
TRAINING_FILE = '../datasets/words_manual.txt'


def normalize_iterator(value):
    if isinstance(value, (list, tuple)):
        return value
    elif isinstance(value, int):
        return tuple((value, ))
    else:
        raise ValueError('The argument must be an integer or a tuple/list of integers. Received: ' + str(value))


class YoloTrainer:
    def __init__(self, conf: YoloConfiguration, seed, resume):
        set_random_seed(seed)
        self.conf = conf
        if resume and os.path.isfile(MODEL_FILE):
            print('resuming saved model')
            self.model = tf.keras.models.load_model(MODEL_FILE, compile=False)
        else:
            print('building new model')
            self.model = self._build_model()

    def _build_model(self):
        kernel = self.conf.step
        model = tf.keras.Sequential()
        model.add(Input(self.conf.input_shape))
        self._conv_block(model, (16, 32), kernel, pool_reduc=(3, 3))
        self._conv_block(model, (64, 128), kernel, pool_reduc=(3, 1))
        self._conv_block(model, 256, kernel, pool_reduc=(2, 1))
        # Convolution implementation of sliding window:
        self._conv_block(model, self.conf.feature_size * 2, 1)
        self._conv_block(model, self.conf.feature_size, 1)
        return model

    @staticmethod
    def _conv_block(model, filter_sizes, kernel_size, pool_reduc=None, batch_norm=True):
        for filters in normalize_iterator(filter_sizes):
            model.add(Conv2D(filters=filters, kernel_size=kernel_size, padding='same', use_bias=not batch_norm, kernel_regularizer=tf.keras.regularizers.l2(0.0005)))
            model.add(LeakyReLU(alpha=0.1))
        if batch_norm:
            model.add(BatchNormalization())
        if pool_reduc is not None:
            pool_size, strides = pool_reduc
            if strides < pool_size:
                left_pad = (pool_size - strides) // 2
                right_pad = (pool_size - strides) - left_pad
                model.add(ZeroPadding2D(((0, 0), (left_pad, right_pad))))  # left-right half-padding
            model.add(MaxPooling2D(pool_size=(pool_size, pool_size), strides=(pool_size, strides))),

    def get_loss_fn(self, score_thresh=0.5):
        cce = tf.keras.losses.binary_crossentropy
        bce = tf.keras.losses.categorical_crossentropy
        mqe = tf.keras.losses.mean_squared_error

        def yolo_loss(y_true, y_pred):
            # y_pred: (batch_size, gridx, gridy, (pc, ...cls, x, w)) = N x 1 x 108 x 77
            # split features
            pc_pred, cls_pred, pos_pred, width_pred = tf.split(y_pred, (1, self.conf.classes, 1, 1), axis=-1)
            pc_true, cls_true, pos_true, width_true = tf.split(y_true, (1, self.conf.classes, 1, 1), axis=-1)
            # squeeze single dimensions
            pc_true = tf.squeeze(pc_true, axis=(1, 3))
            pc_pred = tf.squeeze(pc_pred, axis=(1, 3))
            cls_true = tf.squeeze(cls_true, axis=1)
            cls_pred = tf.squeeze(cls_pred, axis=1)
            pos_true = tf.squeeze(pos_true, axis=(1, 3))
            pos_pred = tf.squeeze(pos_pred, axis=(1, 3))
            width_true = tf.squeeze(width_true, axis=(1, 3))
            width_pred = tf.squeeze(width_pred, axis=(1, 3))
            # calculate all losses
            pc_loss = bce(pc_true, pc_pred)
            cls_loss = cce(cls_true, cls_pred)
            pos_loss = mqe(pos_true, pos_pred)
            width_loss = mqe(width_true, width_pred)
            cls_loss = tf.reduce_sum(cls_loss, axis=1)
            return 0.5 * (pc_loss + cls_loss + pos_loss + width_loss)

        return yolo_loss

    def fit(self, epochs):
        with open(TRAINING_FILE, 'r') as f:
            training_phrases = f.read()
        gen = YoloDatasetGenerator(self.conf)
        data = tf.data.Dataset.from_generator(gen.get_dataset_generator(training_phrases, list_fonts(), augment=True),
                                              (tf.float32, tf.float32), (tf.TensorShape(list(self.conf.input_shape)), tf.TensorShape([1, self.conf.grid_count, self.conf.feature_size])))
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
        model.fit(train_dataset.batch(64), epochs=epochs)
        # evaluate on test set
        print("Evaluating test set")
        outs = model.evaluate(test_dataset.batch(64))
        print("loss:", outs[0])
        print("acc:", outs[1])
        # Save model + parameters
        model.save(MODEL_FILE)
        # Save configuration
        encode_conf(self.conf)


def train(epochs, resume=False):
    if resume:
        conf = decode_conf()
    else:
        conf = YoloConfiguration(FEATURES, (FULL_SIZE, 18 * FULL_SIZE, 1), 3)
    trainer = YoloTrainer(conf, 2, resume)
    trainer.fit(epochs)


if __name__ == "__main__":
    train(50, True)
