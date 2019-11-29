import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mlscanner.splitchar.char_trainer import FEATURES
from mlscanner.splitchar.image_splitter import ImageSplitter

from mlscanner.font_generator import FULL_SIZE, ReScale, TEXT_SIZE, get_font_text_size, list_fonts, RePlace, place_font_image


class YoloDatasetGenerator:
    def __init__(self, features):
        self.features = features
        self.feature_dict = {c: idx for idx, c in enumerate(features)}
        self.current_fontname = None
        self.current_font = None

    def _set_current_font(self, fontname):
        if fontname is not self.current_fontname:
            self.current_fontname = fontname
            self.current_font = ImageFont.truetype(fontname, TEXT_SIZE)

    @staticmethod
    def _generate_font_image(text, image_size, norm_size, font, text_pos):
        # Draw with scale
        image = Image.new("L", image_size, "white")
        draw = ImageDraw.Draw(image)
        draw.text((0, text_pos), text, font=font, fill="black")
        if image_size != norm_size:
            image = image.resize(norm_size, Image.ANTIALIAS)

        return image

    def _generate_font_image_bbox(self, text, window, scale=ReScale.Normal):
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
            bboxes.append((left+round(width/2), width+1, self.feature_dict[char]))

        if norm_width < window * FULL_SIZE:
            # resize canvas to fit minimum size
            canvas_image = Image.new("L", (window * FULL_SIZE, FULL_SIZE), "white")
            canvas_image.paste(image)
            image = canvas_image
        return image, bboxes

    def print_bounding_boxes(self, bboxes):
        for (pos, width, idx) in bboxes:
            print(self.features[idx], ':', f"[{pos},{width}]", end=', ')
        print('')

    def generate_augmented_font_image(self, text, fontname, window):
        self._set_current_font(fontname)
        for scale in ReScale:
            image, bboxes = self._generate_font_image_bbox(text, window, scale)
            for place in RePlace:
                yield (place_font_image(image, False, place), bboxes)

    def get_dataset_generator(self, training_text: str, fontnames, window, step):
        def generator():
            grid_count = int(window / step)
            fsize = grid_count * (len(self.features) + 2)  # +2 for pos, width
            lines = training_text.splitlines()
            for fontname in fontnames:
                print(fontname)
                for line in lines:
                    for image, bboxes in self.generate_augmented_font_image(line, fontname, window):
                        image_data = np.array(image)
                        for slide in range(grid_count):
                            yield self.slide_window(image_data, bboxes, window, slide)
        return generator

    def slide_window(self, image_data, bboxes, window, slide):
        x = image_data.reshape((FULL_SIZE, window, 1))
        y = np.zeros(fsize)


def test_augmented_image():
    gen = YoloDatasetGenerator(FEATURES)
    sample_text = "In the last video"
    full_image = Image.new("L", (FULL_SIZE * len(sample_text), (FULL_SIZE+1) * len(ReScale) * len(RePlace)))
    for idx, (image, bboxes) in enumerate(gen.generate_augmented_font_image(sample_text, "../assets/OpenSans-Regular.ttf", 6 * FULL_SIZE)):
        y = int((FULL_SIZE+1) * idx)
        full_image.paste(image, (0, y))
        print('Dataset', idx)
        gen.print_bounding_boxes(bboxes)
    full_image.save("../out/generated_yolo_sample.png", "PNG")


def main():
    test_augmented_image()


if __name__ == "__main__":
    main()
