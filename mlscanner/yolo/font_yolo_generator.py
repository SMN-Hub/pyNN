import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mlscanner.font_generator import FULL_SIZE, ReScale, TEXT_SIZE, get_font_text_size, RePlace, place_font_image
from mlscanner.splitchar.char_trainer import FEATURES
from mlscanner.splitchar.image_splitter import ImageSplitter


class YoloConfiguration:
    def __init__(self, features, input_shape, step):
        self.features = features
        self.input_shape = input_shape
        self.step = step
        self.grid_count = int(self.columns / step)
        self.feature_dict = {c: idx for idx, c in enumerate(features)}
        self.feature_size = len(features) + 3  # +3 for pc, pos & size

    def __str__(self):
        return f"Yolo configuration will generate {self.input_shape} images with step {self.step} => {self.grid_count} calls"

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


class YoloDatasetGenerator:
    def __init__(self, conf: YoloConfiguration):
        self.conf = conf
        self.current_fontname = None
        self.current_font = None

    def _set_current_font(self, fontname):
        if fontname is not self.current_fontname:
            self.current_fontname = fontname
            self.current_font = ImageFont.truetype(fontname, TEXT_SIZE)
            print(fontname)

    @staticmethod
    def _generate_font_image(text, image_size, norm_size, font, text_pos):
        # Draw with scale
        image = Image.new("L", image_size, "white")
        draw = ImageDraw.Draw(image)
        draw.text((0, text_pos), text, font=font, fill="black")
        if image_size != norm_size:
            image = image.resize(norm_size, Image.ANTIALIAS)

        return image

    def _generate_font_image_bbox(self, text, scale=ReScale.Normal):
        # Draw with scale
        image_height = FULL_SIZE
        text_size = TEXT_SIZE
        if scale == ReScale.UpDown:
            image_height = FULL_SIZE + 4
            text_size = TEXT_SIZE + 4
        elif scale == ReScale.DownUp:
            image_height = FULL_SIZE - 4
            text_size = TEXT_SIZE - 4
        norm_width, _ = get_font_text_size(text, self.current_font)
        font = ImageFont.truetype(self.current_fontname, text_size) if text_size != TEXT_SIZE else self.current_font
        font_width, font_height = get_font_text_size(text, font)
        text_pos = (image_height-font_height)/2

        image = Image.new("L", (font_width, image_height), "white")
        image_data = np.ones((image_height, font_width), dtype=int) * 255
        temp_text = ""
        bboxes = []
        for idx, char in enumerate(text):
            temp_text += char
            if char == ' ':
                continue
            prev_image_data = image_data
            image = self._generate_font_image(temp_text, (font_width, image_height), (norm_width, FULL_SIZE), font, text_pos)
            image_data = np.array(image)
            # wipe out previous chars for bbox detection
            cleaned_image_data = image_data + (255 - prev_image_data) if idx > 0 else image_data
            # detect horizontal bbox
            splitter = ImageSplitter(cleaned_image_data, 0)
            (left, width, _) = splitter.full_section()
            bboxes.append((left+round(width/2), width+1, self.conf.feature_dict[char]))

        if norm_width < self.conf.columns:
            # resize canvas to fit minimum size
            canvas_image = Image.new("L", self.conf.image_shape, "white")
            canvas_image.paste(image)
            image = canvas_image
        return image, bboxes

    def print_bounding_boxes(self, bboxes):
        for (pos, width, idx) in bboxes:
            print(self.conf.features[idx], ':', f"[{pos},{width}]", end=', ')
        print('')

    def generate_augmented_font_image(self, text, fontname):
        self._set_current_font(fontname)
        for scale in ReScale:
            image, bboxes = self._generate_font_image_bbox(text, scale)
            for place in RePlace:
                yield (place_font_image(image, False, place), bboxes)

    def get_dataset_generator(self, training_text: str, fontnames, augment: True):
        print(self.conf)

        def generator():
            lines = training_text.splitlines()
            for fontname in fontnames:
                self._set_current_font(fontname)
                for line in lines:
                    data_set = self.generate_augmented_font_image(line, fontname) if augment else [self._generate_font_image_bbox(line)]
                    for image, bboxes in data_set:
                        bboxes_dict = self._build_bbox_grid(bboxes, self.conf.step)
                        image_data = np.array(image)
                        for grid_slide in range(int(image_data.shape[1]/self.conf.step)):
                            yield self.slide_window(image_data, bboxes_dict, grid_slide)
        return generator

    def slide_window(self, image_data, bboxes_dict, grid_slide):
        feature_size = self.conf.feature_size
        step = self.conf.step
        grid_count = self.conf.grid_count
        window = self.conf.columns

        start_pos = grid_slide * step
        window_data = np.roll(image_data, -start_pos, axis=1)[:, 0:window]
        x = window_data.reshape((FULL_SIZE, window, 1))
        y = np.zeros(feature_size * grid_count, dtype=float)
        for i in range(grid_count):
            # compute rolling grid number
            grid_id = grid_slide + i
            if grid_id >= grid_count:
                grid_id -= grid_count
            if grid_id in bboxes_dict:
                (pos, width, fidx) = bboxes_dict[grid_id]
                y[i * feature_size] = 1.  # pc
                y[i * feature_size + fidx + 1] = 1.  # classification
                y[(i+1) * feature_size - 2] = pos  # character position in cell
                y[(i+1) * feature_size - 1] = width  # character width in cell
        return x, y

    @staticmethod
    def _build_bbox_grid(bboxes, step):
        bboxes_dict = {}
        for (pos, width, fidx) in bboxes:
            grid_id = int(pos / step)
            bboxes_dict[grid_id] = (float((pos-grid_id)/step), float(width/step), fidx)
        return bboxes_dict


def test_augmented_image():
    conf = YoloConfiguration(FEATURES, (FULL_SIZE, 6 * FULL_SIZE, 1), 3)
    gen = YoloDatasetGenerator(conf)
    sample_text = "In the last video, you"
    gen = gen.get_dataset_generator(sample_text, ["../assets/OpenSans-Regular.ttf"], False)
    images = []
    for (x, y) in gen():
        image = Image.new("L", (conf.columns, conf.lines))
        image.putdata(x.reshape(-1))
        images.append(image)
        print(y)
    full_image = Image.new("L", (conf.columns, (conf.lines+1) * len(images)))
    for idx, image in enumerate(images):
        y = int((conf.lines+1) * idx)
        full_image.paste(image, (0, y))
        print('Dataset', idx)
    full_image.save("../out/generated_yolo_sample.png", "PNG")


def main():
    test_augmented_image()


if __name__ == "__main__":
    main()
