import json

CONF_FILE = '../datasets/yolo_model.conf'


def encode_conf(conf):
    with open(CONF_FILE, 'w') as f:
        json.dump(conf.__dict__, f)


def decode_conf():
    with open(CONF_FILE, 'r') as f:
        data = json.load(f)
    conf = object.__new__(YoloConfiguration)
    for key, value in data.items():
        setattr(conf, key, value)
    return conf


class YoloConfiguration:
    def __init__(self, features, input_shape, step):
        self.features = features
        self.input_shape = input_shape
        self.step = step
        self.grid_count = self.columns // step
        self.feature_dict = {c: idx for idx, c in enumerate(features)}
        self.feature_size = len(features) + 3  # +3 for pc, pos & size

    def __str__(self):
        return f"Yolo configuration will generate {self.input_shape} images with step {self.step} => {self.grid_count} cells"

    @property
    def lines(self):
        return self.input_shape[0]

    @property
    def classes(self):
        return len(self.features)

    @property
    def columns(self):
        return self.input_shape[1]

    @property
    def image_shape(self):
        return self.columns, self.lines